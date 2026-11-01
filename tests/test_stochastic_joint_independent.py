"""Independent equations and adversarial state-boundary review.

The oracle consumes the actual dropped inputs: it checks affine differentiation
and momentum equations, not PyTorch's random-number or Dropout algorithm.
"""

import copy

import pytest
import torch

from pytorch_lab.checkpoint import CheckpointError, _content_digest
from pytorch_lab.stochastic_joint import (
    StochasticConfig,
    StochasticJointTrainer,
    load_stochastic_checkpoint,
    restore_snapshot,
    save_stochastic_checkpoint,
)


def close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("seed", [7, 42, 123])
@pytest.mark.parametrize("probability", [0.0, 0.35, 1.0])
@pytest.mark.parametrize("samples,microbatch,accumulation", [(23, 5, 3), (7, 3, 2)])
def test_independent_affine_gradient_momentum_and_safe_points(
    seed, probability, samples, microbatch, accumulation
):
    config = StochasticConfig(
        model_seed=seed,
        dropout_p=probability,
        samples=samples,
        microbatch_size=microbatch,
        accumulation_steps=accumulation,
    )
    trainer = StochasticJointTrainer(config)
    weights = {
        "weight": torch.zeros(3, 2, dtype=torch.float64),
        "bias": torch.zeros(2, dtype=torch.float64),
    }
    gradients = {key: torch.zeros_like(value) for key, value in weights.items()}
    momentum = {}
    elements = 0
    pending = 0
    steps = 2 * ((samples + microbatch - 1) // microbatch)
    for _ in range(steps):
        indices = trainer.step_microbatch()
        trace = trainer.last_trace
        dropped = torch.tensor(trace["dropped_input"], dtype=torch.float64)
        target = trainer._data.targets[list(indices)]
        prediction = dropped @ weights["weight"] + weights["bias"]
        residual = prediction - target
        close(torch.tensor(trace["output"], dtype=torch.float64), prediction)
        close(torch.tensor(trace["loss"], dtype=torch.float64), residual.square().sum())
        gradients["weight"] += 2 * dropped.T @ residual
        gradients["bias"] += 2 * residual.sum(dim=0)
        for name in weights:
            close(torch.tensor(trace["raw_gradients"][name], dtype=torch.float64), gradients[name])
        pending += 1
        elements += target.numel()
        if pending == accumulation or trainer.cursor == 0:
            for name in weights:
                direction = gradients[name] / elements + config.weight_decay * weights[name]
                momentum[name] = (
                    direction.clone()
                    if name not in momentum
                    else config.momentum * momentum[name] + direction
                )
                weights[name] = weights[name] - config.learning_rate * momentum[name]
                gradients[name].zero_()
            pending = elements = 0
        state = trainer.snapshot()
        for index, name in enumerate(weights):
            close(state["model"][name], weights[name])
            if momentum:
                close(state["optimizer"]["state"][index]["momentum_buffer"], momentum[name])
            if pending:
                close(state["pending_gradients"][name], gradients[name])
            else:
                assert state["pending_gradients"][name] is None
        assert state["pending_elements"] == elements
        # A structural equality gate is distinct from the equation tolerance.
        restored = restore_snapshot(state, config)
        assert _content_digest(restored.snapshot()) == _content_digest(state)
        left = restore_snapshot(state)
        right = restore_snapshot(state)
        left.step_microbatch()
        torch.rand(11)  # outside the trainer's owned model RNG
        right.step_microbatch()
        assert left.last_trace == right.last_trace
        assert _content_digest(left.snapshot()) == _content_digest(right.snapshot())
    assert trainer.epoch == 2
    if probability == 1.0:
        assert torch.count_nonzero(weights["weight"]) == 0
        assert torch.count_nonzero(weights["bias"]) > 0


@pytest.mark.parametrize("stage", ["forward", "backward", "update", "advance"])
def test_real_partial_mutation_failure_invalidates_owner_and_preserves_caller_rng(
    stage, monkeypatch, tmp_path
):
    trainer = StochasticJointTrainer(StochasticConfig(accumulation_steps=1))
    if stage == "advance":
        for _ in range(4):
            trainer.step_microbatch()
    snapshot = trainer.snapshot()
    before_rng = torch.get_rng_state().clone()

    def broken(original):
        def call(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("review injected after real operation")

        return call

    if stage == "forward":
        monkeypatch.setattr(trainer._dropout, "forward", broken(trainer._dropout.forward))
    elif stage == "backward":
        monkeypatch.setattr(torch.Tensor, "backward", broken(torch.Tensor.backward))
    elif stage == "update":
        monkeypatch.setattr(trainer._optimizer, "step", broken(trainer._optimizer.step))
    else:
        monkeypatch.setattr(trainer._stream, "advance_epoch", broken(trainer._stream.advance_epoch))
    with pytest.raises(RuntimeError, match="review injected"):
        trainer.step_microbatch()
    assert torch.equal(torch.get_rng_state(), before_rng)
    for action in (
        trainer.step_microbatch,
        trainer.snapshot,
        lambda: save_stochastic_checkpoint(trainer, tmp_path / "failed.pt"),
    ):
        with pytest.raises(CheckpointError, match="failed/in-progress"):
            action()
    monkeypatch.undo()
    resumed = restore_snapshot(snapshot)
    baseline = restore_snapshot(snapshot)
    resumed.step_microbatch()
    baseline.step_microbatch()
    assert _content_digest(resumed.snapshot()) == _content_digest(baseline.snapshot())


def test_constructor_save_load_and_rejected_restore_preserve_caller_rng(tmp_path):
    before = torch.get_rng_state().clone()
    trainer = StochasticJointTrainer()
    assert torch.equal(before, torch.get_rng_state())
    trainer.step_microbatch()
    assert torch.equal(before, torch.get_rng_state())
    path = tmp_path / "safe.pt"
    save_stochastic_checkpoint(trainer, path)
    assert torch.equal(before, torch.get_rng_state())
    loaded = load_stochastic_checkpoint(path, trainer.config)
    assert torch.equal(before, torch.get_rng_state())
    assert _content_digest(loaded.snapshot()) == _content_digest(trainer.snapshot())
    bad = trainer.snapshot()
    bad["data_sha256"] = "0" * 64
    bad["content_sha256"] = _content_digest({k: v for k, v in bad.items() if k != "content_sha256"})
    with pytest.raises(CheckpointError):
        restore_snapshot(bad)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize(
    "mutation",
    [
        "rng",
        "counter",
        "mode",
        "probability",
        "optimizer_order",
        "gradient",
        "version",
        "architecture",
    ],
)
def test_recomputed_checksum_does_not_bypass_semantic_validation(mutation):
    trainer = StochasticJointTrainer()
    for _ in range(4):
        trainer.step_microbatch()
    bad = copy.deepcopy(trainer.snapshot())
    if mutation == "rng":
        bad["model_rng"].zero_()
    elif mutation == "counter":
        bad["pending_elements"] += 2
    elif mutation == "mode":
        bad["training"] = 1
    elif mutation == "probability":
        bad["config"]["dropout_p"] = 1.1
    elif mutation == "optimizer_order":
        bad["optimizer"]["param_groups"][0]["params"].reverse()
    elif mutation == "gradient":
        bad["pending_gradients"]["weight"][0, 0] = float("inf")
    elif mutation == "version":
        bad["schema_version"] = True
    else:
        bad["architecture"] = "different-model"
    bad["content_sha256"] = _content_digest({k: v for k, v in bad.items() if k != "content_sha256"})
    before = torch.get_rng_state().clone()
    with pytest.raises(CheckpointError):
        restore_snapshot(bad)
    assert torch.equal(before, torch.get_rng_state())


def test_precondition_failure_keeps_owner_safe():
    trainer = StochasticJointTrainer()
    state = trainer.snapshot()
    with pytest.raises(CheckpointError, match="after_update"):
        trainer.step_microbatch(fault_at="after_update")
    assert _content_digest(state) == _content_digest(trainer.snapshot())
    trainer.step_microbatch()
