"""Independent mathematical and interruption review of the joint safe point.

The oracle uses explicit affine derivatives and a handwritten SGD recurrence:
no trainer, model forward, autograd, optimizer, or stream code is reused. The
fixture equation is the declared experiment input, not an implementation oracle.
"""

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import pytorch_lab.joint_recovery as joint
from pytorch_lab.checkpoint import CheckpointError, _content_digest
from pytorch_lab.joint_recovery import (
    JointConfig,
    JointTrainer,
    load_joint_checkpoint,
    restore_snapshot,
    save_joint_checkpoint,
)


def _equation_trace(config, epochs):
    fixture_rng = torch.Generator().manual_seed(config.seed)
    features = torch.randn(config.samples, 3, generator=fixture_rng, dtype=torch.float64) / 3
    truth = torch.tensor([[0.5, -0.25], [-0.75, 0.125], [0.25, 0.75]], dtype=torch.float64)
    targets = features @ truth + torch.linspace(
        -0.2, 0.4, config.samples, dtype=torch.float64
    ).unsqueeze(1)
    shuffle_rng = torch.Generator().manual_seed(config.shuffle_seed)
    order = torch.randperm(config.samples, generator=shuffle_rng).tolist()
    parameters = {
        "weight": torch.zeros(3, 2, dtype=torch.float64),
        "bias": torch.zeros(2, dtype=torch.float64),
    }
    gradients = {name: torch.zeros_like(value) for name, value in parameters.items()}
    momentum = {}
    pending_batches = pending_samples = pending_elements = updates = committed_batches = 0
    for epoch in range(epochs):
        for start in range(0, config.samples, config.microbatch_size):
            end = min(start + config.microbatch_size, config.samples)
            indices = order[start:end]
            x, y = features[indices], targets[indices]
            residual = x @ parameters["weight"] + parameters["bias"] - y
            gradients["weight"] += 2 * x.T @ residual
            gradients["bias"] += (2 * residual).sum(0)
            pending_batches += 1
            pending_samples += len(indices)
            pending_elements += y.numel()
            if pending_batches == config.accumulation_steps or end == config.samples:
                for name, parameter in parameters.items():
                    direction = gradients[name] / pending_elements + config.weight_decay * parameter
                    momentum[name] = (
                        direction.clone()
                        if name not in momentum
                        else config.momentum * momentum[name] + direction
                    )
                    parameter -= config.learning_rate * momentum[name]
                    gradients[name].zero_()
                updates += 1
                pending_batches = pending_samples = pending_elements = 0
            committed_batches += 1
            next_epoch, cursor = epoch, end
            if end == config.samples:
                next_epoch, cursor = epoch + 1, 0
                order = torch.randperm(config.samples, generator=shuffle_rng).tolist()
            yield {
                "indices": tuple(indices),
                "model": {name: value.clone() for name, value in parameters.items()},
                "momentum": {name: value.clone() for name, value in momentum.items()},
                "pending_gradients": {
                    name: value.clone() if pending_batches else None
                    for name, value in gradients.items()
                },
                "pending_batches": pending_batches,
                "pending_samples": pending_samples,
                "pending_elements": pending_elements,
                "updates": updates,
                "committed_batches": committed_batches,
                "epoch": next_epoch,
                "cursor": cursor,
                "permutation": order.copy(),
                "generator_state": shuffle_rng.get_state().tolist(),
            }


def _assert_equations(snapshot, expected):
    for field in (
        "pending_batches",
        "pending_samples",
        "pending_elements",
        "updates",
        "committed_batches",
        "epoch",
        "cursor",
    ):
        assert snapshot[field] == expected[field], field
    for field in ("epoch", "cursor", "permutation", "generator_state"):
        assert snapshot["stream"][field] == expected[field], field
    for index, name in enumerate(("weight", "bias")):
        torch.testing.assert_close(
            snapshot["model"][name], expected["model"][name], atol=1e-12, rtol=1e-12
        )
        gradient = expected["pending_gradients"][name]
        if gradient is None:
            assert snapshot["pending_gradients"][name] is None
        else:
            torch.testing.assert_close(
                snapshot["pending_gradients"][name], gradient, atol=1e-12, rtol=1e-12
            )
        if expected["momentum"]:
            torch.testing.assert_close(
                snapshot["optimizer"]["state"][index]["momentum_buffer"],
                expected["momentum"][name],
                atol=1e-12,
                rtol=1e-12,
            )
        else:
            assert snapshot["optimizer"]["state"] == {}


@pytest.mark.parametrize("seed,shuffle_seed", [(42, 19), (7, 123), (123, 7)])
@pytest.mark.parametrize(
    "samples,micro,window",
    [(23, 5, 3), (11, 4, 2), (10, 4, 3), (12, 3, 2), (7, 9, 4), (1, 1, 1), (9, 2, 1)],
)
def test_every_commit_and_restore_against_handwritten_equations(
    seed, shuffle_seed, samples, micro, window
):
    config = JointConfig(
        seed=seed,
        shuffle_seed=shuffle_seed,
        samples=samples,
        microbatch_size=micro,
        accumulation_steps=window,
    )
    trainer = JointTrainer(config)
    seen = []
    for expected in _equation_trace(config, 3):
        before = trainer.snapshot()
        resumed = restore_snapshot(before, config)
        issued = tuple(
            before["stream"]["permutation"][
                before["cursor"] : min(before["cursor"] + micro, samples)
            ]
        )
        assert issued == expected["indices"]
        seen.extend(issued)
        assert trainer.step_microbatch() == expected["indices"]
        assert resumed.step_microbatch() == expected["indices"]
        current = trainer.snapshot()
        _assert_equations(current, expected)
        assert resumed.snapshot()["content_sha256"] == current["content_sha256"]
    assert len(seen) == 3 * samples
    for epoch in range(3):
        assert sorted(seen[epoch * samples : (epoch + 1) * samples]) == list(range(samples))
    assert trainer.snapshot()["committed_batches"] == 3 * math.ceil(samples / micro)


def _assert_tainted(trainer, destination):
    for action in (
        trainer.snapshot,
        trainer.step_microbatch,
        lambda: trainer.train_to(3),
        lambda: save_joint_checkpoint(trainer, destination),
    ):
        with pytest.raises(CheckpointError):
            action()
    assert not destination.exists()


@pytest.mark.parametrize("stage", ["backward", "optimizer", "epoch_advance"])
@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_real_mutation_then_exception_cannot_publish_and_replays_equations(
    monkeypatch, tmp_path, stage, exception
):
    trainer = JointTrainer()
    cut = {"backward": 1, "optimizer": 2, "epoch_advance": 4}[stage]
    for _ in range(cut):
        trainer.step_microbatch()
    before = trainer.snapshot()
    checkpoint = tmp_path / "safe.pt"
    save_joint_checkpoint(trainer, checkpoint)
    if stage == "backward":
        target, name = torch.Tensor, "backward"
    elif stage == "optimizer":
        target, name = trainer._optimizer, "step"
    else:
        target, name = trainer._stream, "advance_epoch"
    original = getattr(target, name)
    mutated = []

    def mutate_then_raise(*args, **kwargs):
        original(*args, **kwargs)
        mutated.append(True)
        raise exception("independent post-mutation interruption")

    with monkeypatch.context() as patch:
        patch.setattr(target, name, mutate_then_raise)
        with pytest.raises(exception, match="independent post-mutation"):
            trainer.step_microbatch()
    assert mutated == [True]
    _assert_tainted(trainer, tmp_path / "must-not-publish.pt")
    restored = load_joint_checkpoint(checkpoint, trainer.config)
    assert restored.snapshot()["content_sha256"] == before["content_sha256"]
    trace = list(_equation_trace(trainer.config, 3))
    for expected in trace[cut:]:
        assert restored.step_microbatch() == expected["indices"]
        _assert_equations(restored.snapshot(), expected)


def test_fresh_isolated_process_restores_pending_window_and_momentum(tmp_path):
    trainer = JointTrainer()
    for _ in range(7):
        trainer.step_microbatch()
    snapshot = trainer.snapshot()
    assert snapshot["pending_batches"] == 2
    assert snapshot["optimizer"]["state"]
    checkpoint = tmp_path / "safe.pt"
    output = tmp_path / "recovered.pt"
    save_joint_checkpoint(trainer, checkpoint)
    # Explicit module root supports both source and installed-wheel execution;
    # -I and an unrelated cwd exclude inherited PYTHONPATH/cwd imports.
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from pytorch_lab.joint_recovery import "
        "JointConfig, load_joint_checkpoint, save_joint_checkpoint; "
        "trainer=load_joint_checkpoint(sys.argv[2], JointConfig()); "
        "trainer.train_to(4); save_joint_checkpoint(trainer, sys.argv[3]); "
        "import json; print(json.dumps({'digest':trainer.snapshot()['content_sha256']}))"
    )
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(Path(joint.__file__).resolve().parents[1]),
            str(checkpoint),
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    resumed = load_joint_checkpoint(output, trainer.config).snapshot()
    _assert_equations(resumed, list(_equation_trace(trainer.config, 4))[-1])
    trainer.train_to(4)
    assert json.loads(child.stdout)["digest"] == trainer.snapshot()["content_sha256"]
    assert resumed["content_sha256"] == trainer.snapshot()["content_sha256"]


def _recompute_checksum(payload):
    payload["content_sha256"] = _content_digest(
        {name: value for name, value in payload.items() if name != "content_sha256"}
    )
    return payload


def test_stale_stream_with_recomputed_outer_checksum_is_rejected():
    trainer = JointTrainer()
    trainer.step_microbatch()
    stale = trainer.snapshot()
    trainer.step_microbatch()
    mixed = trainer.snapshot()
    mixed["stream"] = stale["stream"]
    with pytest.raises(CheckpointError):
        restore_snapshot(_recompute_checksum(mixed))


def test_valid_shape_stale_model_is_detected_by_equations_not_authenticated_by_hash():
    trainer = JointTrainer()
    for _ in range(4):
        trainer.step_microbatch()
    stale = trainer.snapshot()
    for _ in range(5):
        trainer.step_microbatch()
    current = trainer.snapshot()
    mixed = copy.deepcopy(current)
    # Same within-epoch cursor/window, but model from an earlier epoch. Shape,
    # counters and a recomputed digest cannot authenticate computational history.
    assert stale["cursor"] == current["cursor"]
    mixed["model"] = stale["model"]
    accepted = restore_snapshot(_recompute_checksum(mixed))
    accepted.train_to(3)
    expected = list(_equation_trace(trainer.config, 3))[-1]
    with pytest.raises(AssertionError):
        _assert_equations(accepted.snapshot(), expected)
    assert (
        max(
            (accepted.snapshot()["model"][name] - expected["model"][name]).abs().max().item()
            for name in ("weight", "bias")
        )
        > 1e-6
    )
