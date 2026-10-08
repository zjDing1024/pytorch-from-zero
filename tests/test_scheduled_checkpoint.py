"""Independent StepLR timing, exact recovery, and strict trusted-file contracts."""

import copy
import json
import math
from dataclasses import FrozenInstanceError, asdict, replace

import pytest
import torch

import pytorch_lab.checkpoint as checkpoint
import pytorch_lab.scheduled_checkpoint as scheduled
from pytorch_lab.checkpoint import CheckpointError, EpochTrainer, RecoveryConfig
from pytorch_lab.scheduled_checkpoint import (
    ScheduleConfig,
    ScheduledTrainer,
    load_scheduled_checkpoint,
    save_scheduled_checkpoint,
)


def assert_tree_equal(actual, expected):
    """A close loss is insufficient: require typed, bitwise-identical CPU state."""
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
    """Recompute the digest to independently exercise semantic validation."""
    payload = copy.deepcopy(payload)
    if refresh_digest:
        payload.pop("content_sha256", None)
        payload["content_sha256"] = checkpoint._content_digest(payload)
    torch.save(payload, path)


def expected_lrs(config, epochs):
    current = config.learning_rate
    result = [(None, current)]
    for epoch in range(1, epochs + 1):
        used = current
        if epoch % config.step_size == 0:
            current *= config.gamma
        result.append((used, current))
    return result


@pytest.fixture(scope="module")
def trained_snapshot():
    trainer = ScheduledTrainer(ScheduleConfig(seed=17, batch_size=19, step_size=3, gamma=0.7))
    trainer.train_to(7)
    return trainer.snapshot()


@pytest.fixture
def payload(trained_snapshot):
    return copy.deepcopy(trained_snapshot)


def test_config_is_frozen_explicit_and_keeps_recovery_projection():
    config = ScheduleConfig()
    assert asdict(config) == asdict(RecoveryConfig()) | {"step_size": 5, "gamma": 0.5}
    config.validate()
    assert type(config.recovery_config()) is RecoveryConfig
    assert config.recovery_config() == RecoveryConfig()
    with pytest.raises(FrozenInstanceError):
        config.gamma = 0.7
    trainer = ScheduledTrainer(config)
    assert trainer.schedule_config == config
    assert trainer.config == config.recovery_config()
    with pytest.raises(AttributeError):
        trainer.schedule_config = replace(config, gamma=0.6)
    with pytest.raises(AttributeError):
        trainer.config = replace(config.recovery_config(), learning_rate=0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": True},
        {"seed": 2**63},
        {"batch_size": 0},
        {"batch_size": True},
        {"batch_size": 2.5},
        {"learning_rate": 0},
        {"learning_rate": True},
        {"learning_rate": float("nan")},
        {"momentum": 0},
        {"momentum": 1},
        {"momentum": True},
        {"noise_std": -1},
        {"noise_std": float("inf")},
        {"step_size": 0},
        {"step_size": -1},
        {"step_size": True},
        {"step_size": 1.5},
        {"step_size": "3"},
        {"gamma": 0},
        {"gamma": -0.5},
        {"gamma": True},
        {"gamma": "0.5"},
        {"gamma": float("nan")},
        {"gamma": float("inf")},
    ],
)
def test_invalid_config_is_rejected_before_state_changes(kwargs):
    rng = torch.get_rng_state().clone()
    config = ScheduleConfig(**kwargs)
    with pytest.raises(ValueError):
        config.validate()
    with pytest.raises(ValueError):
        ScheduledTrainer(config)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("config", [RecoveryConfig(), {}, True, 3])
def test_trainer_rejects_wrong_config_type(config):
    with pytest.raises(ValueError):
        ScheduledTrainer(config)


@pytest.mark.parametrize("step_size,gamma", [(1, 0.1), (3, 0.7), (5, 0.5), (100, 0.9)])
def test_lr_history_matches_epoch_schedule_exactly(step_size, gamma):
    config = ScheduleConfig(step_size=step_size, gamma=gamma, batch_size=19)
    trainer = ScheduledTrainer(config)
    trainer.train_to(11)
    payload = trainer.snapshot()
    assert len(payload["history"]) == 12
    assert set(payload["history"][0]) == {"epoch", "train_mse", "next_lr"}
    for epoch, (row, (used, next_lr)) in enumerate(
        zip(payload["history"], expected_lrs(config, 11), strict=True)
    ):
        assert row["epoch"] == epoch
        assert row["next_lr"] == next_lr
        if epoch:
            assert set(row) == {
                "epoch",
                "train_mse",
                "online_train_mse",
                "samples_seen",
                "batches",
                "lr_used",
                "next_lr",
            }
            assert row["lr_used"] == used
            assert row["samples_seen"] == 96 and row["batches"] == 6
    assert payload["optimizer"]["param_groups"][0]["lr"] == next_lr
    assert payload["optimizer"]["param_groups"][0]["initial_lr"] == config.learning_rate
    assert payload["scheduler"]["last_epoch"] == 11
    assert payload["scheduler"]["_step_count"] == 12
    assert payload["scheduler"]["base_lrs"] == [config.learning_rate]
    assert payload["scheduler"]["_last_lr"] == [next_lr]


def test_scheduler_steps_once_after_all_batches_and_metric(monkeypatch):
    trainer = ScheduledTrainer(ScheduleConfig(step_size=2, gamma=0.5, batch_size=19))
    events = []
    real_optimizer_step = trainer._optimizer.step
    real_scheduler_step = trainer._scheduler.step
    real_evaluate = scheduled.evaluate

    def optimizer_step(*args, **kwargs):
        events.append(("optimizer", trainer._optimizer.param_groups[0]["lr"]))
        return real_optimizer_step(*args, **kwargs)

    def scheduler_step(*args, **kwargs):
        events.append(("scheduler", trainer._optimizer.param_groups[0]["lr"]))
        return real_scheduler_step(*args, **kwargs)

    def evaluate(*args, **kwargs):
        events.append(("evaluate", trainer._optimizer.param_groups[0]["lr"]))
        return real_evaluate(*args, **kwargs)

    # Preserve the flag used by PyTorch's scheduler wrapper warning check.
    optimizer_step._wrapped_by_lr_sched = True
    monkeypatch.setattr(trainer._optimizer, "step", optimizer_step)
    monkeypatch.setattr(trainer._scheduler, "step", scheduler_step)
    monkeypatch.setattr(scheduled, "evaluate", evaluate)
    trainer.train_to(3)
    assert events == sum(
        [
            [("optimizer", lr)] * 6 + [("evaluate", lr), ("scheduler", lr)]
            for lr in (0.05, 0.05, 0.025)
        ],
        [],
    )


@pytest.mark.parametrize("seed", [0, 42, 991])
@pytest.mark.parametrize("cut", [0, 1, 2, 3, 4, 5, 6, 7, 10])
def test_resume_is_bitwise_identical_before_at_and_after_decay(tmp_path, seed, cut):
    config = ScheduleConfig(seed=seed, batch_size=19, step_size=3, gamma=0.7)
    full = ScheduledTrainer(config)
    full.train_to(10)
    partial = ScheduledTrainer(config)
    partial.train_to(cut)
    path = tmp_path / "boundary.pt"
    save_scheduled_checkpoint(partial, path)
    resumed = load_scheduled_checkpoint(path, expected_config=config)
    assert resumed is not partial
    assert resumed.epoch == cut
    assert_tree_equal(resumed.snapshot(), partial.snapshot())
    resumed.train_to(10)
    assert_tree_equal(resumed.snapshot(), full.snapshot())
    assert_tree_equal(resumed.report(), full.report())


@pytest.mark.parametrize("cut", [2, 3, 4, 5, 6])
def test_restored_current_lr_is_used_for_first_optimizer_update(tmp_path, cut, monkeypatch):
    config = ScheduleConfig(step_size=3, gamma=0.7)
    trainer = ScheduledTrainer(config)
    trainer.train_to(cut)
    path = tmp_path / "state.pt"
    save_scheduled_checkpoint(trainer, path)
    restored = load_scheduled_checkpoint(path)
    observed = []
    original = restored._optimizer.step

    def step(*args, **kwargs):
        observed.append(restored._optimizer.param_groups[0]["lr"])
        return original(*args, **kwargs)

    step._wrapped_by_lr_sched = True
    monkeypatch.setattr(restored._optimizer, "step", step)
    restored.train_to(cut + 1)
    trainer.train_to(cut + 1)
    assert observed == [expected_lrs(config, cut)[-1][1]] * 5
    assert_tree_equal(restored.snapshot(), trainer.snapshot())


def test_epoch_zero_roundtrip_and_repeated_hops(tmp_path):
    config = ScheduleConfig(seed=99, batch_size=7, step_size=2, gamma=0.3)
    full, hopping = ScheduledTrainer(config), ScheduledTrainer(config)
    initial = hopping.snapshot()
    assert initial["epoch"] == initial["scheduler"]["last_epoch"] == 0
    assert initial["scheduler"]["_step_count"] == 1
    assert initial["optimizer"]["state"] == {}
    assert initial["history"][0]["next_lr"] == config.learning_rate
    full.train_to(9)
    for epoch in (0, 1, 2, 3, 4, 8, 9):
        hopping.train_to(epoch)
        path = tmp_path / f"epoch-{epoch}.pt"
        save_scheduled_checkpoint(hopping, path)
        hopping = load_scheduled_checkpoint(path)
    assert_tree_equal(hopping.snapshot(), full.snapshot())


def test_targets_are_total_epochs_and_zero_additional_is_observational():
    trainer = ScheduledTrainer()
    trainer.train_to(0)
    trainer.train_to(5)
    before = trainer.snapshot()
    trainer.train_to(5)
    assert_tree_equal(trainer.snapshot(), before)
    with pytest.raises(ValueError):
        trainer.train_to(4)
    assert_tree_equal(trainer.snapshot(), before)
    trainer.train_to(7)
    full = ScheduledTrainer()
    full.train_to(7)
    assert_tree_equal(trainer.snapshot(), full.snapshot())


@pytest.mark.parametrize("target", [-1, True, 1.5, "3", None, 10001])
def test_invalid_epoch_target_preserves_live_state_and_global_rng(target):
    trainer = ScheduledTrainer()
    before, rng = trainer.snapshot(), torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        trainer.train_to(target)
    assert_tree_equal(trainer.snapshot(), before)
    assert torch.equal(torch.get_rng_state(), rng)


def test_snapshots_and_reports_are_detached_from_all_live_state():
    trainer = ScheduledTrainer()
    trainer.train_to(6)
    before = trainer.snapshot()
    changed = trainer.snapshot()
    changed["model"]["weight"].zero_()
    changed["optimizer"]["state"][0]["momentum_buffer"].zero_()
    changed["optimizer"]["param_groups"][0]["lr"] = 123
    changed["scheduler"]["base_lrs"][0] = 123
    changed["scheduler"]["_last_lr"][0] = 456
    changed["generators"]["train"].zero_()
    changed["generators"]["metric"].zero_()
    changed["history"][1]["next_lr"] = 123
    changed["config"]["gamma"] = 0.9
    report = trainer.report()
    report["history"][1]["lr_used"] = 123
    report["config"]["step_size"] = 123
    assert_tree_equal(trainer.snapshot(), before)
    assert_tree_equal(trainer.report(), trainer.report())
    json.dumps(trainer.report(), allow_nan=False)


def test_loads_are_storage_independent_and_operations_preserve_caller_rng(tmp_path):
    rng = torch.get_rng_state().clone()
    trainer = ScheduledTrainer(ScheduleConfig(seed=123))
    trainer.train_to(5)
    trainer.report()
    path = tmp_path / "boundary.pt"
    save_scheduled_checkpoint(trainer, path)
    first, second = load_scheduled_checkpoint(path), load_scheduled_checkpoint(path)
    before = second.snapshot()
    first.train_to(8)
    assert_tree_equal(second.snapshot(), before)
    assert_tree_equal(trainer.snapshot(), before)
    second.train_to(8)
    assert_tree_equal(first.snapshot(), second.snapshot())
    assert torch.equal(torch.get_rng_state(), rng)


def test_snapshot_is_versioned_cpu_float64_with_real_momentum(payload):
    assert set(payload) == {
        "format",
        "schema_version",
        "torch_version",
        "config",
        "dataset_sha256",
        "epoch",
        "model",
        "optimizer",
        "scheduler",
        "generators",
        "history",
        "content_sha256",
    }
    assert payload["format"] == "pytorch-lab-step-lr-checkpoint"
    assert type(payload["schema_version"]) is int and payload["schema_version"] == 1
    for tensor in payload["model"].values():
        assert tensor.device.type == "cpu" and tensor.dtype == torch.float64
        assert tensor.grad_fn is None and not tensor.requires_grad
    for state in payload["optimizer"]["state"].values():
        assert state["momentum_buffer"].dtype == torch.float64
        assert state["momentum_buffer"].device.type == "cpu"
        assert torch.count_nonzero(state["momentum_buffer"]).item() > 0
    unsigned = {k: v for k, v in payload.items() if k != "content_sha256"}
    assert checkpoint._content_digest(unsigned) == payload["content_sha256"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("format", checkpoint.FORMAT),
        ("schema_version", 2),
        ("schema_version", True),
        ("torch_version", "unknown"),
        ("dataset_sha256", "0" * 64),
        ("epoch", -1),
        ("epoch", True),
        ("epoch", 1.5),
        ("epoch", 10001),
    ],
)
def test_metadata_mutations_with_fresh_digest_are_rejected(tmp_path, payload, field, value):
    payload[field] = value
    path = tmp_path / "bad.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path)


@pytest.mark.parametrize(
    "section", [None, "config", "model", "optimizer", "scheduler", "generators"]
)
@pytest.mark.parametrize("change", ["missing", "extra"])
def test_every_schema_rejects_missing_or_unknown_fields(tmp_path, payload, section, change):
    container = payload if section is None else payload[section]
    if change == "missing":
        container.pop(next(iter(container)))
    else:
        container["unrecognized"] = 1
    path = tmp_path / "bad.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("step_size", 4),
        ("step_size", True),
        ("gamma", 0.5),
        ("gamma", True),
        ("last_epoch", 6),
        ("last_epoch", 8),
        ("last_epoch", 7.0),
        ("_step_count", 7),
        ("_step_count", 9),
        ("_step_count", 8.0),
        ("base_lrs", []),
        ("base_lrs", [0.1]),
        ("base_lrs", [0.05, 0.05]),
        ("_last_lr", []),
        ("_last_lr", [0.05]),
        ("_last_lr", [True]),
        ("_get_lr_called_within_step", True),
        ("_get_lr_called_within_step", 0),
        ("_is_initial", True),
        ("_is_initial", 0),
    ],
)
def test_scheduler_state_must_match_exact_epoch_contract(tmp_path, payload, field, value):
    payload["scheduler"][field] = value
    path = tmp_path / "bad.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path)


@pytest.mark.parametrize(
    "epoch,field",
    [
        (0, "next_lr"),
        (1, "lr_used"),
        (3, "lr_used"),
        (3, "next_lr"),
        (4, "lr_used"),
        (7, "next_lr"),
    ],
)
@pytest.mark.parametrize("value", [0.123, 0, True, "0.05"])
def test_lr_history_is_validated_against_schedule(tmp_path, payload, epoch, field, value):
    payload["history"][epoch][field] = value
    path = tmp_path / "bad.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path)


@pytest.mark.parametrize("field", ["lr_used", "next_lr"])
@pytest.mark.parametrize("damage", ["missing", "extra", "nan", "inf"])
def test_lr_history_schema_and_nonfinite_values_are_rejected(tmp_path, payload, field, damage):
    if damage == "missing":
        del payload["history"][1][field]
    elif damage == "extra":
        payload["history"][1]["unrecognized"] = 1
    else:
        payload["history"][1][field] = float(damage)
    path = tmp_path / "bad.pt"
    write_payload(path, payload, refresh_digest=damage not in {"nan", "inf"})
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path)


@pytest.mark.parametrize(
    "damage",
    [
        "initial_lr",
        "missing_initial_lr",
        "current_lr",
        "groups",
        "group_key",
        "params",
        "momentum",
        "missing_momentum",
        "extra_buffer",
    ],
)
def test_optimizer_scheduler_consistency_is_strict(tmp_path, payload, damage):
    optimizer = payload["optimizer"]
    group = optimizer["param_groups"][0]
    if damage == "initial_lr":
        group["initial_lr"] = 0.1
    elif damage == "missing_initial_lr":
        del group["initial_lr"]
    elif damage == "current_lr":
        group["lr"] = payload["config"]["learning_rate"]
    elif damage == "groups":
        optimizer["param_groups"].append(copy.deepcopy(group))
    elif damage == "group_key":
        group["extra"] = 1
    elif damage == "params":
        group["params"].reverse()
    elif damage == "momentum":
        group["momentum"] = 0.7
    elif damage == "missing_momentum":
        optimizer["state"].clear()
    else:
        optimizer["state"][0]["extra"] = 1
    path = tmp_path / "bad.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": 18},
        {"batch_size": 20},
        {"learning_rate": 0.06},
        {"momentum": 0.7},
        {"noise_std": 0.06},
        {"step_size": 4},
        {"gamma": 0.5},
    ],
)
def test_expected_config_checks_every_field(tmp_path, payload, kwargs):
    path = tmp_path / "valid.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path, ScheduleConfig(**(payload["config"] | kwargs)))


@pytest.mark.parametrize("wrong", [RecoveryConfig(), {}, True])
def test_expected_config_rejects_wrong_type(tmp_path, payload, wrong):
    path = tmp_path / "valid.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path, wrong)


@pytest.mark.parametrize(
    "field,value",
    [("step_size", 0), ("step_size", True), ("gamma", 0), ("gamma", -0.1), ("gamma", True)],
)
def test_checkpoint_schedule_config_is_validated(tmp_path, payload, field, value):
    payload["config"][field] = value
    path = tmp_path / "bad.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path)


@pytest.mark.parametrize("target", ["weight", "bias", "momentum"])
@pytest.mark.parametrize("damage", ["shape", "dtype", "nan", "inf"])
def test_model_and_momentum_tensor_validation(tmp_path, payload, target, damage):
    container, key = payload["model"], target
    if target == "momentum":
        container, key = payload["optimizer"]["state"][0], "momentum_buffer"
    value = container[key]
    if damage == "shape":
        container[key] = value.reshape(1, -1, 1)
    elif damage == "dtype":
        container[key] = value.float()
    else:
        value.fill_(float(damage))
    path = tmp_path / "bad.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path)


@pytest.mark.parametrize("generator", ["train", "metric"])
@pytest.mark.parametrize("damage", ["dtype", "shape", "length", "zero"])
def test_rng_states_are_validated(tmp_path, payload, generator, damage):
    state = payload["generators"][generator]
    if damage == "dtype":
        state = state.float()
    elif damage == "shape":
        state = state.reshape(1, -1)
    elif damage == "length":
        state = state[:4]
    else:
        state.zero_()
    payload["generators"][generator] = state
    path = tmp_path / "bad.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path)


@pytest.mark.parametrize("damage", ["scheduler", "checksum", "truncated", "missing"])
def test_failure_preserves_live_trainer_and_caller_rng(tmp_path, payload, damage):
    trainer = ScheduledTrainer()
    trainer.train_to(6)
    before, rng = trainer.snapshot(), torch.get_rng_state().clone()
    path = tmp_path / "bad.pt"
    if damage == "scheduler":
        payload["scheduler"]["last_epoch"] += 1
        write_payload(path, payload)
    elif damage == "checksum":
        payload["model"]["weight"][0, 0] += 0.01
        write_payload(path, payload, refresh_digest=False)
    elif damage == "truncated":
        write_payload(path, payload)
        path.write_bytes(path.read_bytes()[:128])
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path)
    assert_tree_equal(trainer.snapshot(), before)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("contents", [b"", b"not a checkpoint", b"PK\x03\x04truncated"])
def test_invalid_files_raise_domain_error(tmp_path, contents):
    path = tmp_path / "bad.pt"
    path.write_bytes(contents)
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path)


def test_old_and_scheduled_formats_are_not_interchangeable(tmp_path):
    old, new = tmp_path / "old.pt", tmp_path / "new.pt"
    checkpoint.save_checkpoint(EpochTrainer(), old)
    save_scheduled_checkpoint(ScheduledTrainer(), new)
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(old)
    with pytest.raises(CheckpointError):
        checkpoint.load_checkpoint(new)


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling_symlink"])
def test_saving_never_clobbers_any_existing_name(tmp_path, kind):
    target, other = tmp_path / "target.pt", tmp_path / "other.pt"
    if kind == "file":
        target.write_bytes(b"preserve")
    else:
        if kind == "symlink":
            other.write_bytes(b"preserve")
        target.symlink_to(other)
    before = set(tmp_path.iterdir())
    with pytest.raises((CheckpointError, FileExistsError)):
        save_scheduled_checkpoint(ScheduledTrainer(), target)
    assert set(tmp_path.iterdir()) == before
    if kind != "dangling_symlink":
        assert target.read_bytes() == b"preserve"
    if kind != "file":
        assert target.is_symlink()


def test_partial_serialization_is_not_published(tmp_path, monkeypatch):
    def failed_save(payload, stream, *args, **kwargs):
        stream.write(b"partial")
        raise OSError("simulated failure")

    monkeypatch.setattr(torch, "save", failed_save)
    with pytest.raises((CheckpointError, OSError)):
        save_scheduled_checkpoint(ScheduledTrainer(), tmp_path / "new.pt")
    assert list(tmp_path.iterdir()) == []


def test_failed_epoch_cannot_be_saved_retried_or_reported(tmp_path, monkeypatch):
    trainer = ScheduledTrainer(ScheduleConfig(step_size=1))
    original = trainer._optimizer.step
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted optimizer")
        return original(*args, **kwargs)

    interrupted._wrapped_by_lr_sched = True
    monkeypatch.setattr(trainer._optimizer, "step", interrupted)
    with pytest.raises(RuntimeError, match="interrupted optimizer"):
        trainer.train_to(1)
    assert trainer.epoch == 0
    assert trainer._scheduler.last_epoch == 0
    for operation in (
        trainer.snapshot,
        trainer.report,
        lambda: trainer.train_to(1),
        lambda: save_scheduled_checkpoint(trainer, tmp_path / "bad.pt"),
    ):
        with pytest.raises(CheckpointError):
            operation()
    assert list(tmp_path.iterdir()) == []


def test_in_epoch_snapshot_is_rejected_and_scheduler_cannot_advance_early(monkeypatch):
    trainer = ScheduledTrainer(ScheduleConfig(step_size=1))
    original = trainer._optimizer.step
    calls = 0

    def step(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert trainer._scheduler.last_epoch == 0
        with pytest.raises(CheckpointError):
            trainer.snapshot()
        return original(*args, **kwargs)

    step._wrapped_by_lr_sched = True
    monkeypatch.setattr(trainer._optimizer, "step", step)
    trainer.train_to(1)
    assert calls == 5 and trainer.epoch == trainer._scheduler.last_epoch == 1


def test_metric_failure_does_not_advance_scheduler(tmp_path, monkeypatch):
    trainer = ScheduledTrainer(ScheduleConfig(step_size=2))
    trainer.train_to(1)
    good = trainer.snapshot()
    path = tmp_path / "good.pt"
    save_scheduled_checkpoint(trainer, path)

    def broken(*args, **kwargs):
        raise RuntimeError("metric failed")

    with monkeypatch.context() as patch:
        patch.setattr(scheduled, "evaluate", broken)
        with pytest.raises(RuntimeError, match="metric failed"):
            trainer.train_to(2)
        assert trainer.epoch == trainer._scheduler.last_epoch == 1
        with pytest.raises(CheckpointError):
            trainer.snapshot()
    assert_tree_equal(load_scheduled_checkpoint(path).snapshot(), good)


class UnsafeCheckpointObject:
    """A harmless nonallowlisted object used to check restricted deserialization."""


@pytest.mark.parametrize("unsafe", [False, True])
def test_cpu_weights_only_load_is_never_retried_unsafely(tmp_path, monkeypatch, unsafe):
    path = tmp_path / "state.pt"
    if unsafe:
        torch.save(UnsafeCheckpointObject(), path)
    else:
        save_scheduled_checkpoint(ScheduledTrainer(), path)
    original = torch.load
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", observed)
    if unsafe:
        with pytest.raises(CheckpointError):
            load_scheduled_checkpoint(path)
    else:
        assert load_scheduled_checkpoint(path).epoch == 0
    assert len(calls) == 1
    assert calls[0]["weights_only"] is True
    assert calls[0]["map_location"] == "cpu"


def test_oversized_file_is_rejected_before_torch_load(tmp_path, monkeypatch):
    path = tmp_path / "oversized.pt"
    with path.open("wb") as stream:
        stream.truncate(checkpoint.MAX_CHECKPOINT_BYTES + 1)

    def forbidden(*args, **kwargs):
        pytest.fail("Oversized checkpoint reached deserialization")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path)


def test_many_decays_keep_iterative_lr_rounding_on_resume(tmp_path):
    config = ScheduleConfig(step_size=1, gamma=0.7, batch_size=96)
    trainer = ScheduledTrainer(config)
    trainer.train_to(51)
    iterative = expected_lrs(config, 51)[-1][1]
    assert not math.isclose(iterative, 0.0, abs_tol=0.0)
    path = tmp_path / "rounding.pt"
    save_scheduled_checkpoint(trainer, path)
    restored = load_scheduled_checkpoint(path)
    assert restored.snapshot()["optimizer"]["param_groups"][0]["lr"] == iterative
    trainer.train_to(52)
    restored.train_to(52)
    assert_tree_equal(restored.snapshot(), trainer.snapshot())


def test_gamma_one_reproduces_original_fixed_lr_training_exactly():
    config = ScheduleConfig(seed=912, step_size=2, gamma=1.0, batch_size=19)
    fixed = EpochTrainer(config.recovery_config())
    scheduled_fixed = ScheduledTrainer(config)
    fixed.train_to(9)
    scheduled_fixed.train_to(9)
    expected, actual = fixed.snapshot(), scheduled_fixed.snapshot()
    for field in ("model", "generators", "dataset_sha256", "epoch"):
        assert_tree_equal(actual[field], expected[field])
    group = copy.deepcopy(actual["optimizer"])
    del group["param_groups"][0]["initial_lr"]
    assert_tree_equal(group, expected["optimizer"])
    history = [
        {k: v for k, v in row.items() if k not in {"lr_used", "next_lr"}}
        for row in actual["history"]
    ]
    assert_tree_equal(history, expected["history"])
    assert all(row["next_lr"] == config.learning_rate for row in actual["history"])


def test_validated_payload_never_aliases_new_trainer(payload):
    trainer = scheduled._validate_scheduled_payload(payload)
    before = trainer.snapshot()
    payload["scheduler"]["_last_lr"][0] = 999
    payload["scheduler"]["base_lrs"][0] = 999
    payload["optimizer"]["param_groups"][0]["lr"] = 999
    payload["optimizer"]["state"][0]["momentum_buffer"].fill_(999)
    payload["model"]["weight"].fill_(999)
    payload["generators"]["train"].zero_()
    payload["history"][1]["lr_used"] = 999
    assert_tree_equal(trainer.snapshot(), before)


@pytest.mark.parametrize("gamma", [1, 1.0, 0.5, 0.7])
def test_boundary_rate_helper_supports_zero_and_max_epochs(gamma):
    config = ScheduleConfig(step_size=3, gamma=gamma)
    assert scheduled.learning_rates(config, 0) == [config.learning_rate]
    rates = scheduled.learning_rates(config, 10000)
    assert len(rates) == 10001
    assert rates == [next_lr for _, next_lr in expected_lrs(config, 10000)]


@pytest.mark.parametrize("bad", [1.1, 2, 100])
def test_gamma_above_one_is_rejected(bad):
    with pytest.raises(ValueError):
        ScheduledTrainer(ScheduleConfig(gamma=bad))


def test_subnormal_and_zero_underflow_learning_rates_remain_valid(tmp_path):
    config = ScheduleConfig(step_size=1, gamma=1e-100, batch_size=96)
    trainer = ScheduledTrainer(config)
    trainer.train_to(5)
    assert trainer.snapshot()["optimizer"]["param_groups"][0]["lr"] == 0.0
    path = tmp_path / "underflow.pt"
    save_scheduled_checkpoint(trainer, path)
    restored = load_scheduled_checkpoint(path)
    trainer.train_to(6)
    restored.train_to(6)
    assert_tree_equal(restored.snapshot(), trainer.snapshot())


@pytest.mark.parametrize(
    "damage", ["missing", "extra", "order", "bool_epoch", "counts", "negative_loss", "integer_loss"]
)
def test_non_lr_history_contract_stays_strict(tmp_path, payload, damage):
    history = payload["history"]
    if damage == "missing":
        history.pop()
    elif damage == "extra":
        history.append(copy.deepcopy(history[-1]))
    elif damage == "order":
        history[1]["epoch"] = 2
    elif damage == "bool_epoch":
        history[1]["epoch"] = True
    elif damage == "counts":
        history[1]["samples_seen"] = 95
    elif damage == "negative_loss":
        history[1]["train_mse"] = -1.0
    else:
        history[1]["train_mse"] = 0
    path = tmp_path / "bad.pt"
    write_payload(path, payload)
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path)


def test_epoch_zero_rejects_nonempty_optimizer_state(tmp_path, payload):
    initial = ScheduledTrainer(ScheduleConfig(**payload["config"])).snapshot()
    initial["optimizer"]["state"] = payload["optimizer"]["state"]
    path = tmp_path / "bad.pt"
    write_payload(path, initial)
    with pytest.raises(CheckpointError):
        load_scheduled_checkpoint(path)


def test_save_rejects_oversized_serialization_and_removes_temporary_file(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduled, "MAX_CHECKPOINT_BYTES", 1)
    with pytest.raises(CheckpointError):
        save_scheduled_checkpoint(ScheduledTrainer(), tmp_path / "too-big.pt")
    assert list(tmp_path.iterdir()) == []


def test_failed_publish_does_not_leave_temporary_file(tmp_path, monkeypatch):
    def failed_link(*args, **kwargs):
        raise OSError("simulated link failure")

    monkeypatch.setattr(checkpoint.os, "link", failed_link)
    with pytest.raises((CheckpointError, OSError)):
        save_scheduled_checkpoint(ScheduledTrainer(), tmp_path / "new.pt")
    assert list(tmp_path.iterdir()) == []


def test_racing_publication_preserves_winner(tmp_path, monkeypatch):
    target = tmp_path / "race.pt"
    real_link = checkpoint.os.link

    def race(source, destination, *args, **kwargs):
        target.write_bytes(b"first writer")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(checkpoint.os, "link", race)
    with pytest.raises((CheckpointError, FileExistsError)):
        save_scheduled_checkpoint(ScheduledTrainer(), target)
    assert target.read_bytes() == b"first writer"
    assert list(tmp_path.iterdir()) == [target]


def test_scheduler_failure_is_not_a_completed_boundary(tmp_path, monkeypatch):
    trainer = ScheduledTrainer(ScheduleConfig(step_size=1))

    def failed_step(*args, **kwargs):
        raise RuntimeError("scheduler failed")

    monkeypatch.setattr(trainer._scheduler, "step", failed_step)
    with pytest.raises(RuntimeError, match="scheduler failed"):
        trainer.train_to(1)
    assert trainer.epoch == 0
    with pytest.raises(CheckpointError):
        save_scheduled_checkpoint(trainer, tmp_path / "bad.pt")
    assert list(tmp_path.iterdir()) == []
