"""Contract and persistence tests for the original joint checkpoint API."""

import copy
from dataclasses import asdict, replace

import pytest
import torch

from pytorch_lab.checkpoint import MAX_CHECKPOINT_BYTES, CheckpointError, _content_digest
from pytorch_lab.joint_recovery import (
    JointConfig,
    JointTrainer,
    load_joint_checkpoint,
    restore_snapshot,
    save_joint_checkpoint,
)


def recompute_checksum(payload):
    payload["content_sha256"] = _content_digest(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    return payload


@pytest.fixture
def pending():
    trainer = JointTrainer()
    for _ in range(4):
        trainer.step_microbatch()
    return trainer


@pytest.mark.parametrize(
    "field,value",
    [
        ("shuffle_seed", True),
        ("shuffle_seed", -1),
        ("shuffle_seed", 2**32),
        ("shuffle_seed", 1.0),
        ("seed", True),
        ("samples", 0),
        ("samples", 1001),
        ("microbatch_size", 0),
        ("accumulation_steps", 0),
        ("learning_rate", 1),
        ("learning_rate", float("nan")),
        ("momentum", 0.0),
        ("weight_decay", -0.1),
    ],
)
def test_bad_config(field, value):
    with pytest.raises(CheckpointError):
        JointTrainer(replace(JointConfig(), **{field: value}))


@pytest.mark.parametrize("config", [False, {}, 1])
def test_config_exact_type(config):
    with pytest.raises(CheckpointError):
        JointTrainer(config)


@pytest.mark.parametrize("field", list(asdict(JointConfig())))
def test_expected_config_binds_every_field(pending, field):
    original = pending.config
    value = getattr(original, field)
    changed = value + (1 if type(value) is int else 0.01)
    with pytest.raises(CheckpointError):
        restore_snapshot(pending.snapshot(), replace(original, **{field: changed}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("format", "split"),
        ("torch_version", "other"),
        ("data_sha256", "0" * 64),
        ("epoch", True),
        ("epoch", -1),
        ("epoch", 101),
        ("cursor", 21),
        ("cursor", 23),
        ("updates", 2),
        ("committed_batches", 5),
        ("pending_batches", 2),
        ("pending_samples", 6),
        ("pending_elements", 5),
        ("pending_elements", True),
    ],
)
def test_rejects_checksum_recomputed_invalid_metadata(pending, field, value):
    payload = pending.snapshot()
    payload[field] = value
    with pytest.raises(CheckpointError):
        restore_snapshot(recompute_checksum(payload))


@pytest.mark.parametrize("field", ["model", "pending_gradients"])
@pytest.mark.parametrize("mutation", ["shape", "dtype", "nan", "missing", "extra", "none"])
def test_rejects_checksum_recomputed_invalid_tensor_fields(pending, field, mutation):
    payload = pending.snapshot()
    if mutation == "shape":
        payload[field]["weight"] = torch.zeros(2, 3, dtype=torch.float64)
    elif mutation == "dtype":
        payload[field]["weight"] = payload[field]["weight"].float()
    elif mutation == "nan":
        payload[field]["weight"][0, 0] = float("nan")
    elif mutation == "missing":
        del payload[field]["weight"]
    elif mutation == "extra":
        payload[field]["other"] = None
    else:
        payload[field]["weight"] = None
    with pytest.raises(CheckpointError):
        restore_snapshot(recompute_checksum(payload))


@pytest.mark.parametrize("mutation", ["lr", "lr_type", "momentum", "missing", "nan", "id_type"])
def test_rejects_checksum_recomputed_invalid_optimizer(pending, mutation):
    payload = pending.snapshot()
    optimizer = payload["optimizer"]
    if mutation == "lr":
        optimizer["param_groups"][0]["lr"] = 0.04
    elif mutation == "lr_type":
        optimizer["param_groups"][0]["dampening"] = False
    elif mutation == "momentum":
        optimizer["state"][0]["momentum_buffer"] = torch.zeros(2, dtype=torch.float64)
    elif mutation == "missing":
        optimizer["state"].pop(0)
    elif mutation == "nan":
        optimizer["state"][0]["momentum_buffer"][0, 0] = float("nan")
    else:
        optimizer["state"][False] = optimizer["state"].pop(0)
    with pytest.raises(CheckpointError):
        restore_snapshot(recompute_checksum(payload))


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "bad_digest", "model_without_recompute_checksum"]
)
def test_top_level_schema_and_integrity(pending, mutation):
    payload = pending.snapshot()
    if mutation == "missing":
        del payload["stream"]
    elif mutation == "extra":
        payload["external_ack"] = True
    elif mutation == "bad_digest":
        payload["content_sha256"] = "0" * 64
    else:
        payload["model"]["weight"].add_(1)
    with pytest.raises(CheckpointError):
        restore_snapshot(payload)


def test_snapshot_and_restore_are_detached(pending):
    before = pending.snapshot()
    restored = restore_snapshot(before, pending.config)
    digest = before["content_sha256"]
    before["model"]["weight"].add_(10)
    before["pending_gradients"]["bias"].zero_()
    before["optimizer"]["state"][0]["momentum_buffer"].zero_()
    before["stream"]["permutation"].reverse()
    assert pending.snapshot()["content_sha256"] == digest
    assert restored.snapshot()["content_sha256"] == digest
    restored.step_microbatch()
    assert pending.snapshot()["content_sha256"] == digest


def test_disk_roundtrip_and_no_clobber(pending, tmp_path):
    path = tmp_path / "joint.pt"
    save_joint_checkpoint(pending, path)
    encoded = path.read_bytes()
    loaded = load_joint_checkpoint(path, pending.config)
    assert loaded.snapshot()["content_sha256"] == pending.snapshot()["content_sha256"]
    pending.step_microbatch()
    with pytest.raises(FileExistsError):
        save_joint_checkpoint(pending, path)
    assert path.read_bytes() == encoded


def test_serialization_failure_does_not_publish_or_taint(pending, tmp_path, monkeypatch):
    digest = pending.snapshot()["content_sha256"]

    def fail_save(*args, **kwargs):
        raise OSError("injected serialization error")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="serialization"):
        save_joint_checkpoint(pending, tmp_path / "joint.pt")
    assert list(tmp_path.iterdir()) == []
    assert pending.snapshot()["content_sha256"] == digest


@pytest.mark.parametrize("kind", ["missing", "garbage", "oversize", "wrong_payload"])
def test_invalid_disk_files(tmp_path, kind):
    path = tmp_path / "bad.pt"
    if kind == "garbage":
        path.write_bytes(b"not a checkpoint")
    elif kind == "oversize":
        with path.open("wb") as handle:
            handle.truncate(MAX_CHECKPOINT_BYTES + 1)
    elif kind == "wrong_payload":
        torch.save({"model": {}}, path)
    with pytest.raises(CheckpointError):
        load_joint_checkpoint(path)


@pytest.mark.parametrize("argument", [True, -1, 101, 1.0])
def test_train_horizon_validation(argument):
    trainer = JointTrainer()
    before = trainer.snapshot()["content_sha256"]
    with pytest.raises(CheckpointError):
        trainer.train_to(argument)
    assert trainer.snapshot()["content_sha256"] == before


def test_horizon_boundary_and_empty_window():
    trainer = JointTrainer(JointConfig(samples=1, microbatch_size=3))
    trainer.train_to(100)
    payload = trainer.snapshot()
    assert (trainer.epoch, trainer.cursor, trainer.committed_batches) == (100, 0, 100)
    assert payload["pending_gradients"] == {"weight": None, "bias": None}
    restored = restore_snapshot(payload)
    restored.train_to(100)
    with pytest.raises(CheckpointError):
        restored.step_microbatch()
    payload["pending_gradients"]["bias"] = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(CheckpointError):
        restore_snapshot(recompute_checksum(payload))


@pytest.mark.parametrize("fault", ["unknown", True, [], "after_update"])
def test_invalid_failpoint_does_not_mutate_or_taint(fault):
    trainer = JointTrainer()
    digest = trainer.snapshot()["content_sha256"]
    with pytest.raises(CheckpointError):
        trainer.step_microbatch(fault_at=fault)
    assert trainer.snapshot()["content_sha256"] == digest
    trainer.step_microbatch()


def test_global_rng_unaffected_by_create_step_restore(pending):
    state = torch.get_rng_state().clone()
    candidate = JointTrainer()
    candidate.train_to(2)
    restore_snapshot(copy.deepcopy(pending.snapshot()))
    assert torch.equal(state, torch.get_rng_state())
