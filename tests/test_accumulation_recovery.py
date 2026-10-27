"""Recovery invariants, not a claim of learner mastery or arbitrary-loader support."""

import copy
import random
from dataclasses import replace

import pytest
import torch

from pytorch_lab.accumulation_recovery import (
    WindowConfig,
    WindowTrainer,
    load_window_checkpoint,
    restore_snapshot,
    save_window_checkpoint,
)
from pytorch_lab.accumulation_recovery_cli import _negative_controls, _resign
from pytorch_lab.checkpoint import CheckpointError


@pytest.mark.parametrize(
    "samples,micro,window", [(23, 5, 3), (1, 1, 1), (7, 9, 4), (12, 3, 2), (11, 2, 9)]
)
@pytest.mark.parametrize("seed", [42, 7, 123])
def test_every_safe_point_replays_exactly(samples, micro, window, seed):
    config = WindowConfig(
        seed=seed, samples=samples, microbatch_size=micro, accumulation_steps=window
    )
    original = WindowTrainer(config)
    while original.epoch < 2:
        saved = original.snapshot()
        resumed = restore_snapshot(saved, config)
        original.step_microbatch()
        resumed.step_microbatch()
        assert original.snapshot()["content_sha256"] == resumed.snapshot()["content_sha256"]
    assert (
        restore_snapshot(original.snapshot()).snapshot()["content_sha256"]
        == original.snapshot()["content_sha256"]
    )


def test_pending_is_raw_sum_and_snapshot_is_independent():
    trainer = WindowTrainer()
    trainer.step_microbatch()
    saved = trainer.snapshot()
    x, y = trainer._data.features[:5], trainer._data.targets[:5]
    assert torch.equal(saved["pending_gradients"]["weight"], 2 * x.T @ (-y))
    assert torch.equal(saved["pending_gradients"]["bias"], (-2 * y).sum(0))
    assert saved["pending_elements"] == 10
    assert saved["updates"] == 0
    trainer.step_microbatch()
    assert saved["pending_samples"] == 5
    assert not torch.equal(
        saved["pending_gradients"]["weight"], trainer.snapshot()["pending_gradients"]["weight"]
    )


def test_rng_isolation():
    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    trainer = WindowTrainer()
    trainer.train_to(2)
    restore_snapshot(trainer.snapshot())
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", True),
        ("samples", 0),
        ("microbatch_size", 1.0),
        ("accumulation_steps", -1),
        ("learning_rate", 1),
        ("momentum", 0.0),
        ("weight_decay", float("nan")),
    ],
)
def test_invalid_config(field, value):
    with pytest.raises(CheckpointError):
        WindowTrainer(replace(WindowConfig(), **{field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("epoch", True),
        ("cursor", 1),
        ("updates", 2),
        ("pending_batches", 0),
        ("pending_samples", 4),
        ("pending_elements", 9),
        ("torch_version", "other"),
        ("data_sha256", "bad"),
        ("order", "shuffle"),
        ("training_rng", "global"),
    ],
)
def test_bad_metadata_rejected_even_with_recomputed_checksum(field, value):
    trainer = WindowTrainer()
    trainer.step_microbatch()
    saved = trainer.snapshot()
    saved[field] = value
    with pytest.raises(CheckpointError):
        restore_snapshot(_resign(saved))


@pytest.mark.parametrize(
    "fault", ["missing", "none", "nan", "dtype", "shape", "momentum", "group", "checksum"]
)
def test_bad_tensors_optimizer_or_integrity(fault):
    trainer = WindowTrainer()
    for _ in range(4):
        trainer.step_microbatch()
    saved = trainer.snapshot()
    if fault == "missing":
        del saved["pending_gradients"]
    elif fault == "none":
        saved["pending_gradients"]["bias"] = None
    elif fault == "nan":
        saved["model"]["bias"][0] = float("nan")
    elif fault == "dtype":
        saved["pending_gradients"]["bias"] = saved["pending_gradients"]["bias"].float()
    elif fault == "shape":
        saved["model"]["weight"] = torch.zeros(2, 3, dtype=torch.float64)
    elif fault == "momentum":
        del saved["optimizer"]["state"][0]["momentum_buffer"]
    elif fault == "group":
        saved["optimizer"]["param_groups"][0]["params"] = [1, 0]
    else:
        saved["content_sha256"] = "0" * 64
    if fault != "checksum":
        _resign(saved)
    with pytest.raises(CheckpointError):
        restore_snapshot(saved)


def test_no_clobber_and_disk_roundtrip(tmp_path):
    trainer = WindowTrainer()
    trainer.step_microbatch()
    path = tmp_path / "own.pt"
    save_window_checkpoint(trainer, path)
    before = path.read_bytes()
    assert (
        load_window_checkpoint(path).snapshot()["content_sha256"]
        == trainer.snapshot()["content_sha256"]
    )
    with pytest.raises(FileExistsError):
        save_window_checkpoint(trainer, path)
    assert path.read_bytes() == before
    with pytest.raises(CheckpointError):
        load_window_checkpoint(path, replace(trainer.config, seed=7))
    path.write_bytes(b"invalid own test fixture")
    with pytest.raises(CheckpointError):
        load_window_checkpoint(path)


def test_failure_invalidates_no_partial_snapshot(monkeypatch, tmp_path):
    trainer = WindowTrainer()
    trainer.step_microbatch()
    trainer.step_microbatch()
    original = trainer._optimizer.step

    def fail():
        original()
        raise RuntimeError("injected post-update failure")

    monkeypatch.setattr(trainer._optimizer, "step", fail)
    with pytest.raises(RuntimeError):
        trainer.step_microbatch()
    for action in (
        trainer.snapshot,
        trainer.step_microbatch,
        lambda: trainer.train_to(2),
        lambda: save_window_checkpoint(trainer, tmp_path / "bad.pt"),
    ):
        with pytest.raises(CheckpointError):
            action()
    assert not (tmp_path / "bad.pt").exists()


def test_valid_shape_faults_require_replay_not_checksum_authentication():
    controls = _negative_controls()
    assert all(controls["invalid_state_rejected"].values())
    assert all(v["detected"] for v in controls["valid_shape_fault_injection"].values())


def test_restore_does_not_alias_snapshot():
    trainer = WindowTrainer()
    trainer.step_microbatch()
    saved = trainer.snapshot()
    restored = restore_snapshot(saved)
    before = copy.deepcopy(restored.snapshot())
    saved["model"]["weight"].fill_(9)
    saved["pending_gradients"]["bias"].zero_()
    assert restored.snapshot()["content_sha256"] == before["content_sha256"]
