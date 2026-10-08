"""Epoch-boundary recovery contracts, including malformed and interrupted writes."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest
import torch

import pytorch_lab.checkpoint as checkpoint
from pytorch_lab.checkpoint import (
    CheckpointError,
    EpochTrainer,
    RecoveryConfig,
    load_checkpoint,
    save_checkpoint,
)


def assert_tree_equal(actual, expected):
    """Require exact CPU values, not an approximate training result."""
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        assert actual.device == expected.device
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            assert_tree_equal(actual[name], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            assert_tree_equal(value, reference)
    else:
        assert actual == expected


def write_payload(path, payload, *, refresh_digest=True):
    """Re-sign corruptions so structural validation, rather than the hash, runs."""
    payload = copy.deepcopy(payload)
    if refresh_digest:
        payload.pop("content_sha256", None)
        payload["content_sha256"] = checkpoint._content_digest(payload)
    torch.save(payload, path)


@pytest.fixture(scope="module")
def trained_snapshot():
    trainer = EpochTrainer(RecoveryConfig(seed=17, batch_size=19))
    trainer.train_to(3)
    return trainer.snapshot()


@pytest.fixture
def payload(trained_snapshot):
    return copy.deepcopy(trained_snapshot)


def test_default_config_is_frozen_and_explicit():
    config = RecoveryConfig()
    assert asdict(config) == {
        "seed": 42,
        "batch_size": 20,
        "learning_rate": 0.05,
        "momentum": 0.8,
        "noise_std": 0.05,
    }
    config.validate()
    with pytest.raises(FrozenInstanceError):
        config.seed = 1
    assert issubclass(CheckpointError, ValueError)


def test_live_configuration_cannot_be_reassigned_out_of_sync_with_loaders(tmp_path):
    trainer = EpochTrainer()
    trainer.train_to(2)
    original_config, before = trainer.config, trainer.snapshot()
    # Both sizes produce five batches, so history counts alone cannot catch this drift.
    with pytest.raises(AttributeError):
        trainer.config = replace(trainer.config, batch_size=21)
    assert trainer.config is original_config
    assert trainer._train_loader.batch_size == trainer.config.batch_size == 20
    assert_tree_equal(trainer.snapshot(), before)
    path = tmp_path / "unchanged.pt"
    save_checkpoint(trainer, path)
    resumed = load_checkpoint(path)
    trainer.train_to(3)
    resumed.train_to(3)
    assert_tree_equal(resumed.snapshot(), trainer.snapshot())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": 2**63},
        {"seed": True},
        {"seed": 1.5},
        {"batch_size": 0},
        {"batch_size": -1},
        {"batch_size": True},
        {"batch_size": 2.5},
        {"learning_rate": 0},
        {"learning_rate": -0.1},
        {"learning_rate": True},
        {"learning_rate": "0.1"},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"momentum": 0},
        {"momentum": 1},
        {"momentum": -0.5},
        {"momentum": True},
        {"momentum": "0.8"},
        {"momentum": float("nan")},
        {"momentum": float("inf")},
        {"noise_std": -0.1},
        {"noise_std": True},
        {"noise_std": "0.05"},
        {"noise_std": float("nan")},
        {"noise_std": float("inf")},
    ],
)
def test_invalid_config_is_rejected_before_training(kwargs):
    config = RecoveryConfig(**kwargs)
    with pytest.raises(ValueError):
        config.validate()
    with pytest.raises(ValueError):
        EpochTrainer(config)


@pytest.mark.parametrize("seed", [0, 42, 2**63 - 1])
def test_seed_boundaries_and_zero_noise_are_supported(seed):
    trainer = EpochTrainer(RecoveryConfig(seed=seed, noise_std=0))
    trainer.train_to(1)
    assert trainer.epoch == 1


def test_snapshot_has_versioned_plain_cpu_payload_and_real_momentum(payload):
    assert set(payload) == {
        "format",
        "schema_version",
        "torch_version",
        "config",
        "dataset_sha256",
        "epoch",
        "model",
        "optimizer",
        "generators",
        "history",
        "content_sha256",
    }
    assert payload["format"] == "pytorch-lab-epoch-checkpoint"
    assert type(payload["schema_version"]) is int and payload["schema_version"] == 1
    assert type(payload["torch_version"]) is str
    assert payload["epoch"] == 3
    assert len(payload["dataset_sha256"]) == len(payload["content_sha256"]) == 64
    assert set(payload["model"]) == {"weight", "bias"}
    assert payload["model"]["weight"].shape == (3, 1)
    assert payload["model"]["bias"].shape == (1,)
    for tensor in payload["model"].values():
        assert type(tensor) is torch.Tensor
        assert tensor.device.type == "cpu" and tensor.dtype == torch.float64
        assert tensor.grad_fn is None and not tensor.requires_grad
    assert set(payload["optimizer"]) == {"state", "param_groups"}
    assert len(payload["optimizer"]["state"]) == 2
    for state in payload["optimizer"]["state"].values():
        assert set(state) == {"momentum_buffer"}
        assert torch.count_nonzero(state["momentum_buffer"]).item() > 0
    assert set(payload["generators"]) == {"train", "metric"}
    for state in payload["generators"].values():
        assert state.dtype == torch.uint8 and state.device.type == "cpu"
        assert state.ndim == 1
    unsigned = {key: value for key, value in payload.items() if key != "content_sha256"}
    assert payload["content_sha256"] == checkpoint._content_digest(unsigned)


@pytest.mark.parametrize("seed", [0, 42, 991])
@pytest.mark.parametrize("cut", [0, 1, 4, 8])
def test_resume_matches_uninterrupted_training_exactly(tmp_path, seed, cut):
    config = RecoveryConfig(seed=seed, batch_size=19)
    uninterrupted = EpochTrainer(config)
    uninterrupted.train_to(8)
    interrupted = EpochTrainer(config)
    interrupted.train_to(cut)
    path = tmp_path / "epoch.pt"
    save_checkpoint(interrupted, path)
    resumed = load_checkpoint(path, expected_config=config)
    assert resumed is not interrupted
    assert resumed.epoch == cut
    assert_tree_equal(resumed.snapshot(), interrupted.snapshot())
    resumed.train_to(8)
    assert_tree_equal(resumed.snapshot(), uninterrupted.snapshot())
    assert_tree_equal(resumed.report(), uninterrupted.report())


def test_repeated_checkpoint_hops_preserve_the_same_trajectory(tmp_path):
    reference = EpochTrainer(RecoveryConfig(seed=93, batch_size=7))
    reference.train_to(9)
    resumed = EpochTrainer(RecoveryConfig(seed=93, batch_size=7))
    for epoch in [0, 1, 2, 5, 9]:
        resumed.train_to(epoch)
        path = tmp_path / f"epoch-{epoch}.pt"
        save_checkpoint(resumed, path)
        resumed = load_checkpoint(path)
    assert_tree_equal(resumed.snapshot(), reference.snapshot())


def test_checkpoint_continues_in_a_fresh_python_process(tmp_path):
    reference = EpochTrainer(RecoveryConfig(seed=24, batch_size=13))
    reference.train_to(7)
    interrupted = EpochTrainer(RecoveryConfig(seed=24, batch_size=13))
    interrupted.train_to(3)
    source, destination = tmp_path / "before.pt", tmp_path / "after.pt"
    save_checkpoint(interrupted, source)
    environment = dict(os.environ)
    source_root = str(Path(checkpoint.__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [source_root, environment.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pytorch_lab.checkpoint import load_checkpoint, save_checkpoint; "
            "trainer = load_checkpoint(sys.argv[1]); trainer.train_to(7); "
            "save_checkpoint(trainer, sys.argv[2])",
            str(source),
            str(destination),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert_tree_equal(load_checkpoint(destination).snapshot(), reference.snapshot())


def test_epoch_zero_checkpoint_keeps_empty_optimizer_and_initial_history(tmp_path):
    trainer = EpochTrainer()
    initial = trainer.snapshot()
    assert trainer.epoch == 0
    assert initial["optimizer"]["state"] == {}
    assert len(initial["history"]) == 1
    assert set(initial["history"][0]) == {"epoch", "train_mse"}
    assert initial["history"][0]["epoch"] == 0
    assert initial["history"][0]["train_mse"] > 0
    path = tmp_path / "initial.pt"
    save_checkpoint(trainer, path)
    restored = load_checkpoint(path)
    assert_tree_equal(restored.snapshot(), initial)
    trainer.train_to(3)
    restored.train_to(3)
    assert_tree_equal(restored.snapshot(), trainer.snapshot())


def test_history_counts_uneven_last_batch_correctly(payload):
    assert len(payload["history"]) == payload["epoch"] + 1
    for epoch, row in enumerate(payload["history"]):
        assert row["epoch"] == epoch
        assert row["train_mse"] >= 0
        if epoch:
            assert set(row) == {"epoch", "train_mse", "online_train_mse", "samples_seen", "batches"}
            assert row["samples_seen"] == 96
            assert row["batches"] == 6
            assert row["online_train_mse"] >= 0


def test_snapshot_is_deeply_independent_from_live_training():
    trainer = EpochTrainer()
    trainer.train_to(2)
    baseline = trainer.snapshot()
    changed = trainer.snapshot()
    changed["model"]["weight"].fill_(123)
    changed["model"]["bias"].fill_(456)
    for state in changed["optimizer"]["state"].values():
        state["momentum_buffer"].zero_()
    changed["generators"]["train"].zero_()
    changed["generators"]["metric"].zero_()
    changed["config"]["seed"] = 100
    changed["history"][0]["train_mse"] = 123
    changed["optimizer"]["param_groups"][0]["lr"] = 100
    assert_tree_equal(trainer.snapshot(), baseline)


def test_loaded_instances_do_not_share_model_optimizer_or_rng_storage(tmp_path):
    trainer = EpochTrainer()
    trainer.train_to(2)
    path = tmp_path / "state.pt"
    save_checkpoint(trainer, path)
    first, second = load_checkpoint(path), load_checkpoint(path)
    before = second.snapshot()
    first.train_to(5)
    assert_tree_equal(second.snapshot(), before)
    assert_tree_equal(trainer.snapshot(), before)
    second.train_to(5)
    assert_tree_equal(first.snapshot(), second.snapshot())


def test_training_loading_saving_and_reporting_preserve_global_rng(tmp_path):
    before = torch.random.get_rng_state().clone()
    trainer = EpochTrainer(RecoveryConfig(seed=123))
    trainer.train_to(2)
    trainer.report()
    trainer.report()
    path = tmp_path / "state.pt"
    save_checkpoint(trainer, path)
    loaded = load_checkpoint(path)
    loaded.train_to(5)
    loaded.report()
    assert torch.equal(torch.random.get_rng_state(), before)


def test_report_is_json_finite_and_does_not_advance_state():
    trainer = EpochTrainer()
    trainer.train_to(2)
    before = trainer.snapshot()
    first = trainer.report()
    assert isinstance(first, dict)
    json.dumps(first, allow_nan=False)
    assert_tree_equal(trainer.report(), first)
    assert_tree_equal(trainer.snapshot(), before)


@pytest.mark.parametrize("target", [-1, True, 1.5, "3", None, 10001])
def test_invalid_epoch_target_does_not_change_trainer(target):
    trainer = EpochTrainer()
    before = trainer.snapshot()
    with pytest.raises(ValueError):
        trainer.train_to(target)
    assert_tree_equal(trainer.snapshot(), before)


def test_training_target_is_absolute_and_cannot_rewind():
    trainer = EpochTrainer()
    trainer.train_to(3)
    before = trainer.snapshot()
    trainer.train_to(3)
    assert_tree_equal(trainer.snapshot(), before)
    with pytest.raises(ValueError):
        trainer.train_to(2)
    assert_tree_equal(trainer.snapshot(), before)
    trainer.train_to(5)
    reference = EpochTrainer()
    reference.train_to(5)
    assert_tree_equal(trainer.snapshot(), reference.snapshot())


@pytest.mark.parametrize("reset", ["optimizer", "shuffle"])
def test_omitting_real_optimizer_or_shuffle_state_changes_trajectory(tmp_path, reset):
    trainer = EpochTrainer(RecoveryConfig(seed=12, batch_size=11))
    trainer.train_to(3)
    path = tmp_path / "state.pt"
    save_checkpoint(trainer, path)
    wrong = load_checkpoint(path)
    if reset == "optimizer":
        wrong._optimizer.state.clear()
    else:
        wrong._train_loader.generator.manual_seed(12)
    trainer.train_to(4)
    wrong.train_to(4)
    assert not torch.equal(wrong._model.weight, trainer._model.weight)


@pytest.mark.parametrize(
    "field,value",
    [
        ("format", "some-other-format"),
        ("schema_version", 2),
        ("schema_version", True),
        ("torch_version", 123),
        ("torch_version", "0.0.0-incompatible"),
        ("dataset_sha256", "0" * 64),
        ("dataset_sha256", "invalid"),
        ("epoch", -1),
        ("epoch", True),
        ("epoch", 1.5),
        ("epoch", 10001),
    ],
)
def test_rejects_invalid_metadata_with_valid_checksum(tmp_path, payload, field, value):
    payload[field] = value
    path = tmp_path / "invalid.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


@pytest.mark.parametrize("field", ["config", "model", "generators", "optimizer"])
@pytest.mark.parametrize("change", ["missing", "extra"])
def test_nested_schema_rejects_missing_and_extra_keys(tmp_path, payload, field, change):
    if change == "missing":
        payload[field].pop(next(iter(payload[field])))
    else:
        payload[field]["unexpected"] = 1
    path = tmp_path / "invalid.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_top_level_schema_is_strict(tmp_path, payload, change):
    if change == "missing":
        del payload["format"]
    else:
        payload["unexpected"] = "field"
    path = tmp_path / "invalid.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", -1),
        ("seed", True),
        ("batch_size", 0),
        ("batch_size", True),
        ("learning_rate", -1.0),
        ("momentum", 1.0),
        ("noise_std", -1.0),
    ],
)
def test_load_validates_config_even_with_valid_checksum(tmp_path, payload, field, value):
    payload["config"][field] = value
    path = tmp_path / "invalid.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": 18},
        {"batch_size": 20},
        {"learning_rate": 0.06},
        {"momentum": 0.7},
        {"noise_std": 0.06},
    ],
)
def test_expected_config_must_match_every_field(tmp_path, payload, kwargs):
    path = tmp_path / "state.pt"
    write_payload(path, payload)
    expected = RecoveryConfig(**(payload["config"] | kwargs))
    with pytest.raises(CheckpointError):
        load_checkpoint(path, expected_config=expected)


@pytest.mark.parametrize("target", ["weight", "bias", "momentum"])
@pytest.mark.parametrize("damage", ["shape", "dtype", "nan", "inf"])
def test_tensor_shape_dtype_and_finiteness_are_validated(tmp_path, payload, target, damage):
    if target == "momentum":
        container = next(iter(payload["optimizer"]["state"].values()))
        key = "momentum_buffer"
    else:
        container, key = payload["model"], target
    original = container[key]
    if damage == "shape":
        container[key] = original.reshape(-1, 1, 1)
    elif damage == "dtype":
        container[key] = original.float()
    else:
        container[key].fill_(float(damage))
    path = tmp_path / "invalid.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


@pytest.mark.parametrize("field", ["train", "metric"])
@pytest.mark.parametrize("damage", ["dtype", "shape", "length"])
def test_rng_state_is_validated(tmp_path, payload, field, damage):
    state = payload["generators"][field]
    if damage == "dtype":
        state = state.float()
    elif damage == "shape":
        state = state.reshape(1, -1)
    else:
        state = state[:4]
    payload["generators"][field] = state
    path = tmp_path / "invalid.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


@pytest.mark.parametrize("field", ["train", "metric"])
def test_rng_state_must_be_accepted_by_the_generator(tmp_path, payload, field):
    payload["generators"][field].zero_()
    path = tmp_path / "invalid.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


@pytest.mark.parametrize(
    "damage",
    ["no_momentum", "extra_state", "extra_buffer", "parameter_order", "lr", "momentum"],
)
def test_optimizer_state_must_match_actual_sgd_contract(tmp_path, payload, damage):
    optimizer = payload["optimizer"]
    if damage == "no_momentum":
        optimizer["state"].clear()
    elif damage == "extra_state":
        optimizer["state"][99] = copy.deepcopy(next(iter(optimizer["state"].values())))
    elif damage == "extra_buffer":
        next(iter(optimizer["state"].values()))["unrecognized"] = torch.tensor([1])
    elif damage == "parameter_order":
        optimizer["param_groups"][0]["params"].reverse()
    elif damage == "lr":
        optimizer["param_groups"][0]["lr"] = 0.99
    else:
        optimizer["param_groups"][0]["momentum"] = 0.2
    path = tmp_path / "invalid.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_epoch_zero_rejects_populated_momentum_state(tmp_path, payload):
    initial = EpochTrainer(RecoveryConfig(**payload["config"])).snapshot()
    initial["optimizer"]["state"] = payload["optimizer"]["state"]
    path = tmp_path / "invalid.pt"
    write_payload(path, initial)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


@pytest.mark.parametrize("value", [True, 0.0])
def test_optimizer_parameter_ids_are_strict_integers(tmp_path, payload, value):
    state = payload["optimizer"]["state"]
    original_key = 1 if value is True else 0
    state[value] = state.pop(original_key)
    path = tmp_path / "invalid.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_row",
        "extra_row",
        "epoch_order",
        "bool_epoch",
        "missing_key",
        "extra_key",
        "negative_loss",
        "samples_seen",
        "batches",
        "nonfinite_loss",
    ],
)
def test_history_is_validated(tmp_path, payload, damage):
    history = payload["history"]
    if damage == "missing_row":
        history.pop()
    elif damage == "extra_row":
        history.append(copy.deepcopy(history[-1]))
    elif damage == "epoch_order":
        history[1]["epoch"] = 2
    elif damage == "bool_epoch":
        history[1]["epoch"] = True
    elif damage == "missing_key":
        del history[1]["online_train_mse"]
    elif damage == "extra_key":
        history[1]["validation_mse"] = 1.0
    elif damage == "negative_loss":
        history[1]["train_mse"] = -1.0
    elif damage == "samples_seen":
        history[1]["samples_seen"] = 95
    elif damage == "batches":
        history[1]["batches"] = 5
    else:
        history[1]["train_mse"] = float("nan")
    path = tmp_path / "invalid.pt"
    if damage == "nonfinite_loss":
        # A non-finite primitive is outside canonical JSON as well as the schema.
        write_payload(path, payload, refresh_digest=False)
    else:
        write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_checksum_detects_finite_weight_tampering(tmp_path, payload):
    payload["model"]["weight"][0, 0] += 0.01
    path = tmp_path / "tampered.pt"
    write_payload(path, payload, refresh_digest=False)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


@pytest.mark.parametrize("contents", [b"", b"not a checkpoint", b"PK\x03\x04truncated"])
def test_unreadable_or_truncated_checkpoint_raises_domain_error(tmp_path, contents):
    path = tmp_path / "invalid.pt"
    path.write_bytes(contents)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_missing_checkpoint_is_reported_without_modifying_live_trainer(tmp_path):
    trainer = EpochTrainer()
    trainer.train_to(2)
    before = trainer.snapshot()
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "absent.pt")
    assert_tree_equal(trainer.snapshot(), before)


def test_failed_deep_validation_does_not_modify_live_trainer_or_global_rng(tmp_path, payload):
    trainer = EpochTrainer()
    trainer.train_to(2)
    before, rng = trainer.snapshot(), torch.random.get_rng_state().clone()
    payload["model"]["weight"] = torch.zeros(8, 1, dtype=torch.float64)
    path = tmp_path / "invalid.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)
    assert_tree_equal(trainer.snapshot(), before)
    assert torch.equal(torch.random.get_rng_state(), rng)


class UnsafeCheckpointObject:
    """A harmless custom type which the weights-only loader must not permit."""


def test_successful_load_also_uses_restricted_cpu_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "valid.pt"
    save_checkpoint(EpochTrainer(), path)
    original_load = torch.load
    calls = []

    def observed_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_load(*args, **kwargs)

    monkeypatch.setattr(checkpoint.torch, "load", observed_load)
    assert load_checkpoint(path).epoch == 0
    assert len(calls) == 1
    assert calls[0]["weights_only"] is True
    assert calls[0]["map_location"] == "cpu"


def test_load_always_uses_weights_only_cpu_without_unsafe_retry(tmp_path, monkeypatch):
    path = tmp_path / "custom-object.pt"
    torch.save(UnsafeCheckpointObject(), path)
    original_load = torch.load
    calls = []

    def observed_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_load(*args, **kwargs)

    monkeypatch.setattr(checkpoint.torch, "load", observed_load)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)
    assert len(calls) == 1
    assert calls[0]["weights_only"] is True
    assert calls[0]["map_location"] == "cpu"


def test_oversize_checkpoint_is_rejected_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "oversize.pt"
    with path.open("wb") as stream:
        stream.truncate(checkpoint.MAX_CHECKPOINT_BYTES + 1)

    def forbidden_load(*args, **kwargs):
        pytest.fail("An oversized checkpoint reached torch.load")

    monkeypatch.setattr(checkpoint.torch, "load", forbidden_load)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_size_limit_also_prevents_publishing_an_oversize_save(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "MAX_CHECKPOINT_BYTES", 1)
    with pytest.raises(CheckpointError):
        save_checkpoint(EpochTrainer(), tmp_path / "too-big.pt")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("target_kind", ["file", "symlink"])
def test_save_never_clobbers_existing_destination(tmp_path, target_kind):
    target = tmp_path / "existing.pt"
    original = b"irreplaceable prior result"
    if target_kind == "file":
        target.write_bytes(original)
    else:
        other = tmp_path / "other.pt"
        other.write_bytes(original)
        target.symlink_to(other)
    before = set(tmp_path.iterdir())
    with pytest.raises((CheckpointError, FileExistsError)):
        save_checkpoint(EpochTrainer(), target)
    assert target.read_bytes() == original
    assert set(tmp_path.iterdir()) == before
    if target_kind == "symlink":
        assert target.is_symlink()


def test_serialization_failure_removes_temporary_file(tmp_path, monkeypatch):
    trainer = EpochTrainer()
    target = tmp_path / "new.pt"

    def failed_save(payload, destination, *args, **kwargs):
        if hasattr(destination, "write"):
            destination.write(b"partial serialization")
        else:
            with open(destination, "wb") as stream:
                stream.write(b"partial serialization")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(checkpoint.torch, "save", failed_save)
    with pytest.raises((CheckpointError, OSError)):
        save_checkpoint(trainer, target)
    assert list(tmp_path.iterdir()) == []


def test_link_failure_removes_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / "new.pt"

    def failed_link(*args, **kwargs):
        raise OSError("simulated link failure")

    monkeypatch.setattr(checkpoint.os, "link", failed_link)
    with pytest.raises((CheckpointError, OSError)):
        save_checkpoint(EpochTrainer(), target)
    assert list(tmp_path.iterdir()) == []


def test_competing_writer_wins_without_being_overwritten(tmp_path, monkeypatch):
    target = tmp_path / "race.pt"
    original_link = checkpoint.os.link
    competitor = b"another writer committed first"

    def racing_link(source, destination, *args, **kwargs):
        target.write_bytes(competitor)
        return original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(checkpoint.os, "link", racing_link)
    with pytest.raises((CheckpointError, FileExistsError)):
        save_checkpoint(EpochTrainer(), target)
    assert target.read_bytes() == competitor
    assert list(tmp_path.iterdir()) == [target]


def test_interrupted_training_is_not_a_saveable_epoch_boundary(tmp_path, monkeypatch):
    trainer = EpochTrainer()
    original_step = trainer._optimizer.step
    calls = 0

    def interrupted_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption after one update")
        return original_step(*args, **kwargs)

    monkeypatch.setattr(trainer._optimizer, "step", interrupted_step)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer.train_to(2)
    assert calls == 2
    assert trainer.epoch == 0
    with pytest.raises(CheckpointError):
        save_checkpoint(trainer, tmp_path / "partial.pt")
    with pytest.raises(CheckpointError):
        trainer.train_to(2)
    with pytest.raises(CheckpointError):
        trainer.report()
    assert list(tmp_path.iterdir()) == []


def test_metric_failure_after_updates_does_not_commit_an_epoch(tmp_path, monkeypatch):
    trainer = EpochTrainer()
    trainer.train_to(1)
    valid = trainer.snapshot()
    checkpoint_path = tmp_path / "last-complete.pt"
    save_checkpoint(trainer, checkpoint_path)

    def interrupted_metric(*args, **kwargs):
        raise RuntimeError("simulated metric interruption")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, "evaluate", interrupted_metric)
        with pytest.raises(RuntimeError, match="metric interruption"):
            trainer.train_to(2)
        assert trainer.epoch == 1
        with pytest.raises(CheckpointError):
            save_checkpoint(trainer, tmp_path / "partial.pt")
    assert list(tmp_path.iterdir()) == [checkpoint_path]
    assert_tree_equal(load_checkpoint(checkpoint_path).snapshot(), valid)


def test_saving_from_inside_optimizer_step_is_rejected(tmp_path, monkeypatch):
    trainer = EpochTrainer()
    original_step = trainer._optimizer.step
    calls = 0

    def checked_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        with pytest.raises(CheckpointError):
            save_checkpoint(trainer, tmp_path / "mid-epoch.pt")
        return original_step(*args, **kwargs)

    monkeypatch.setattr(trainer._optimizer, "step", checked_step)
    trainer.train_to(1)
    assert calls == 5 and trainer.epoch == 1
    assert list(tmp_path.iterdir()) == []
    save_checkpoint(trainer, tmp_path / "boundary.pt")
    assert load_checkpoint(tmp_path / "boundary.pt").epoch == 1
