"""Independent affine equations, not an independent Dropout RNG implementation."""

import copy
from dataclasses import replace

import pytest
import torch

from pytorch_lab.checkpoint import CheckpointError, _content_digest
from pytorch_lab.stochastic_joint import (
    FAULTS,
    InjectedFailure,
    StochasticConfig,
    StochasticJointTrainer,
    load_stochastic_checkpoint,
    restore_snapshot,
    save_stochastic_checkpoint,
)


@pytest.mark.parametrize("seed", [7, 42, 123])
@pytest.mark.parametrize("probability", [0.0, 0.35, 1.0])
def test_independent_equations_and_every_safe_point(seed, probability):
    config = StochasticConfig(seed=seed, model_seed=seed, dropout_p=probability)
    trainer = StochasticJointTrainer(config)
    w = torch.zeros(3, 2, dtype=torch.float64)
    b = torch.zeros(2, dtype=torch.float64)
    gw, gb = torch.zeros_like(w), torch.zeros_like(b)
    mw = mb = None
    denominator = 0
    for step in range(15):
        indices = trainer.step_microbatch()
        trace = trainer.last_trace
        x = torch.tensor(trace["dropped_input"], dtype=torch.float64)
        y = trainer._data.targets[list(indices)]
        out = x @ w + b
        delta = 2 * (out - y)
        gw += x.T @ delta
        gb += delta.sum(0)
        denominator += y.numel()
        torch.testing.assert_close(
            out, torch.tensor(trace["output"], dtype=torch.float64), atol=1e-12, rtol=1e-12
        )
        assert abs((out - y).square().sum().item() - trace["loss"]) <= 1e-12
        torch.testing.assert_close(
            gw,
            torch.tensor(trace["raw_gradients"]["weight"], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )
        torch.testing.assert_close(
            gb,
            torch.tensor(trace["raw_gradients"]["bias"], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )
        update = step % 5 in (2, 4)
        if update:
            dw, db = (
                gw / denominator + config.weight_decay * w,
                gb / denominator + config.weight_decay * b,
            )
            mw = dw.clone() if mw is None else config.momentum * mw + dw
            mb = db.clone() if mb is None else config.momentum * mb + db
            w -= config.learning_rate * mw
            b -= config.learning_rate * mb
            gw.zero_()
            gb.zero_()
            denominator = 0
        snapshot = trainer.snapshot()
        torch.testing.assert_close(w, snapshot["model"]["weight"], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(b, snapshot["model"]["bias"], atol=1e-12, rtol=1e-12)
        if mw is not None:
            torch.testing.assert_close(
                mw, snapshot["optimizer"]["state"][0]["momentum_buffer"], atol=1e-12, rtol=1e-12
            )
            torch.testing.assert_close(
                mb, snapshot["optimizer"]["state"][1]["momentum_buffer"], atol=1e-12, rtol=1e-12
            )
        restored = restore_snapshot(snapshot)
        assert restored.snapshot()["content_sha256"] == snapshot["content_sha256"]
        if step < 14:
            a = restore_snapshot(snapshot)
            a.step_microbatch()
            restored.step_microbatch()
            assert a.snapshot()["content_sha256"] == restored.snapshot()["content_sha256"]


@pytest.mark.parametrize("fault", FAULTS)
def test_failure_restores_caller_rng_and_invalidates_owner(fault, tmp_path):
    trainer = StochasticJointTrainer()
    for _ in range(2):
        trainer.step_microbatch()
    saved = trainer.snapshot()
    caller = torch.get_rng_state().clone()
    with pytest.raises(InjectedFailure):
        trainer.step_microbatch(fault_at=fault)
    assert torch.equal(caller, torch.get_rng_state())
    for op in (
        trainer.step_microbatch,
        trainer.snapshot,
        lambda: save_stochastic_checkpoint(trainer, tmp_path / "bad.pt"),
    ):
        with pytest.raises(CheckpointError):
            op()
    good = restore_snapshot(saved)
    good.step_microbatch()
    assert good.committed_batches == 3


def test_external_rng_and_persistence_are_isolated(tmp_path):
    caller = torch.get_rng_state().clone()
    a = StochasticJointTrainer()
    b = StochasticJointTrainer()
    assert torch.equal(caller, torch.get_rng_state())
    for _ in range(8):
        caller = torch.get_rng_state().clone()
        a.step_microbatch()
        assert torch.equal(caller, torch.get_rng_state())
        torch.rand(13)
        b.step_microbatch()
        assert a.snapshot()["content_sha256"] == b.snapshot()["content_sha256"]
    caller = torch.get_rng_state().clone()
    save_stochastic_checkpoint(a, tmp_path / "saved.pt")
    c = load_stochastic_checkpoint(tmp_path / "saved.pt")
    assert torch.equal(caller, torch.get_rng_state())
    assert c.snapshot()["content_sha256"] == a.snapshot()["content_sha256"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_seed", True),
        ("model_seed", -1),
        ("dropout_p", 1),
        ("dropout_p", float("nan")),
        ("dropout_p", -0.1),
        ("dropout_p", 1.1),
    ],
)
def test_new_config_rejected(field, value):
    with pytest.raises(CheckpointError):
        StochasticJointTrainer(replace(StochasticConfig(), **{field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("training", False),
        ("training", 1),
        ("architecture", "other"),
        ("model_rng", torch.zeros(3, dtype=torch.uint8)),
        ("model_rng", None),
    ],
)
def test_new_snapshot_rejected_without_rng_pollution(field, value):
    state = StochasticJointTrainer().snapshot()
    state[field] = value
    state["content_sha256"] = _content_digest(
        {k: v for k, v in state.items() if k != "content_sha256"}
    )
    caller = torch.get_rng_state().clone()
    with pytest.raises(CheckpointError):
        restore_snapshot(state)
    assert torch.equal(caller, torch.get_rng_state())


@pytest.mark.parametrize("mutation", ["probability", "dropout_mode", "model_mode", "inplace"])
def test_live_mode_rejected(mutation):
    t = StochasticJointTrainer()
    if mutation == "probability":
        t._dropout.p = 0.2
    elif mutation == "dropout_mode":
        t._dropout.eval()
    elif mutation == "model_mode":
        t._model.eval()
    else:
        t._dropout.inplace = True
    with pytest.raises(CheckpointError):
        t.step_microbatch()
    with pytest.raises(CheckpointError):
        t.snapshot()


def test_precondition_error_keeps_safe_owner():
    t = StochasticJointTrainer()
    before = t.snapshot()["content_sha256"]
    with pytest.raises(CheckpointError):
        t.step_microbatch(fault_at="bad")
    assert t.snapshot()["content_sha256"] == before


def test_wrong_state_negative_controls():
    t = StochasticJointTrainer()
    for _ in range(4):
        t.step_microbatch()
    snapshot = t.snapshot()
    reference = restore_snapshot(snapshot)
    reference.step_microbatch()
    expected = reference.snapshot()["content_sha256"]
    for error in ("seed", "gradient", "momentum"):
        broken = restore_snapshot(snapshot)
        if error == "seed":
            broken._model_rng = torch.Generator().manual_seed(broken.config.model_seed).get_state()
        elif error == "gradient":
            for p in broken._model.parameters():
                p.grad.zero_()
        else:
            for state in broken._optimizer.state.values():
                state["momentum_buffer"].zero_()
        broken.step_microbatch()
        assert broken.snapshot()["content_sha256"] != expected
    missing = copy.deepcopy(snapshot)
    del missing["model_rng"]
    with pytest.raises(CheckpointError):
        restore_snapshot(missing)
