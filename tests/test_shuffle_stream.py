"""Original tests of the explicit CPU index-issuance contract, not training."""

import copy
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from pytorch_lab.shuffle_stream import (
    MAX_EPOCH,
    MAX_SNAPSHOT_BYTES,
    ShuffledBatchStream,
    ShuffleStateError,
    _payload_digest,
    load_shuffle_checkpoint,
    restore_snapshot,
    save_shuffle_checkpoint,
)
from pytorch_lab.shuffle_stream_cli import _negative_controls, fixture_config


def config(**kwargs):
    return replace(fixture_config(), **kwargs)


def resign(payload):
    payload["content_sha256"] = _payload_digest(payload)
    return payload


def drain(stream):
    batches = []
    while not stream.exhausted:
        batches.append(stream.next_batch())
    return batches


@pytest.mark.parametrize("seed", [42, 7, 123])
@pytest.mark.parametrize("samples,batch", [(23, 5), (12, 3), (3, 7), (0, 3), (1, 1), (1, 4)])
@pytest.mark.parametrize("drop", [False, True])
def test_every_cutpoint_and_future_epoch_match_independent_oracle(seed, samples, batch, drop):
    cfg = config(seed=seed, samples=samples, batch_size=batch, drop_last=drop)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    stream = ShuffledBatchStream(cfg)
    for epoch in range(3):
        permutation = torch.randperm(samples, generator=generator).tolist()
        limit = samples - samples % batch if drop else samples
        expected = [tuple(permutation[i : min(i + batch, limit)]) for i in range(0, limit, batch)]
        assert stream.epoch == epoch
        assert stream.snapshot()["permutation"] == permutation
        for cut in range(len(expected) + 1):
            saved = stream.snapshot()
            restored = restore_snapshot(saved, cfg)
            assert drain(restored) == expected[cut:]
            assert restored.exhausted
            assert restored.snapshot()["generator_state"] == generator.get_state().tolist()
            if cut < len(expected):
                assert stream.next_batch() == expected[cut]
        assert len({i for b in expected for i in b}) == limit
        before = stream.snapshot()
        for _ in range(2):
            with pytest.raises(StopIteration):
                stream.next_batch()
            assert stream.snapshot() == before
        if epoch < 2:
            stream.advance_epoch()


@pytest.mark.parametrize(
    "field,value",
    [
        ("samples", True),
        ("samples", -1),
        ("samples", 10001),
        ("batch_size", 0),
        ("batch_size", 10001),
        ("batch_size", 1.0),
        ("seed", True),
        ("seed", -1),
        ("seed", 2**32),
        ("drop_last", 1),
        ("dataset_sha256", "x"),
        ("dataset_sha256", "A" * 64),
    ],
)
def test_invalid_config_rejected(field, value):
    with pytest.raises(ShuffleStateError):
        ShuffledBatchStream(config(**{field: value}))


def test_required_typed_config_and_expected_config():
    with pytest.raises(ShuffleStateError):
        ShuffledBatchStream({})
    stream = ShuffledBatchStream(config())
    with pytest.raises(ShuffleStateError):
        restore_snapshot(stream.snapshot(), {})
    with pytest.raises(ShuffleStateError):
        save_shuffle_checkpoint({}, "unused.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("format", "unknown"),
        ("version", True),
        ("version", 2),
        ("torch_version", "other"),
        ("epoch", True),
        ("epoch", -1),
        ("epoch", MAX_EPOCH + 1),
        ("cursor", True),
        ("cursor", 1),
        ("cursor", -1),
        ("cursor", 24),
        ("permutation", []),
        ("permutation", list(range(22)) + [21]),
        ("permutation", list(range(22)) + [23]),
        ("permutation", [False] + list(range(1, 23))),
        ("generator_state", []),
        ("generator_state", "bad"),
        ("content_sha256", "bad"),
    ],
)
def test_malformed_snapshot_rejected_even_when_resigned(field, value):
    saved = ShuffledBatchStream(config()).snapshot()
    saved[field] = value
    if field != "content_sha256":
        resign(saved)
    with pytest.raises(ShuffleStateError):
        restore_snapshot(saved)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "config_extra",
        "config_missing",
        "rng_bool",
        "rng_range",
        "rng_invalid",
        "rng_wrong_seed",
        "digest",
    ],
)
def test_schema_rng_and_integrity_rejected(fault):
    saved = ShuffledBatchStream(config()).snapshot()
    if fault == "missing":
        del saved["cursor"]
    elif fault == "extra":
        saved["unexpected"] = 1
    elif fault == "config_extra":
        saved["config"]["unexpected"] = 1
    elif fault == "config_missing":
        del saved["config"]["seed"]
    elif fault == "rng_bool":
        saved["generator_state"][0] = True
    elif fault == "rng_range":
        saved["generator_state"][0] = 256
    elif fault == "rng_invalid":
        saved["generator_state"] = [0] * len(saved["generator_state"])
    elif fault == "rng_wrong_seed":
        saved["generator_state"] = torch.Generator().manual_seed(7).get_state().tolist()
    else:
        saved["permutation"][0], saved["permutation"][1] = (
            saved["permutation"][1],
            saved["permutation"][0],
        )
    if fault != "digest":
        resign(saved)
    with pytest.raises(ShuffleStateError):
        restore_snapshot(saved)


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 7},
        {"samples": 22},
        {"batch_size": 4},
        {"drop_last": True},
        {"dataset_sha256": hashlib.sha256(b"different ordered dataset").hexdigest()},
    ],
)
def test_expected_config_binds_dataset_and_policy(change):
    saved = ShuffledBatchStream(config()).snapshot()
    with pytest.raises(ShuffleStateError):
        restore_snapshot(saved, config(**change))


def test_snapshot_and_restoration_own_all_mutable_values():
    original = ShuffledBatchStream(config())
    saved = original.snapshot()
    frozen = copy.deepcopy(saved)
    restored = restore_snapshot(saved)
    original.next_batch()
    assert saved == frozen
    saved["permutation"].reverse()
    saved["generator_state"][0] ^= 1
    saved["config"]["seed"] = 0
    assert restored.snapshot() == frozen
    detached = restored.snapshot()
    detached["permutation"][0] = -1
    detached["generator_state"][0] = 0
    assert restored.snapshot() == frozen


def test_global_rng_and_unrelated_generator_are_untouched():
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state().clone()
    unrelated = torch.Generator().manual_seed(9)
    other_state = unrelated.get_state().clone()
    stream = ShuffledBatchStream(config())
    for _ in range(3):
        drain(stream)
        stream = restore_snapshot(stream.snapshot())
        stream.advance_epoch()
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert torch.equal(unrelated.get_state(), other_state)


def test_early_advance_and_epoch_limit_leave_state_unchanged():
    stream = ShuffledBatchStream(config())
    before = stream.snapshot()
    with pytest.raises(ShuffleStateError):
        stream.advance_epoch()
    assert stream.snapshot() == before
    drain(stream)
    saved = stream.snapshot()
    saved["epoch"] = MAX_EPOCH
    stream = restore_snapshot(resign(saved))
    before = stream.snapshot()
    with pytest.raises(ShuffleStateError):
        stream.advance_epoch()
    assert stream.snapshot() == before


def test_failed_next_epoch_draw_is_atomic(monkeypatch):
    stream = ShuffledBatchStream(config())
    drain(stream)
    before = stream.snapshot()
    original = torch.randperm

    def fail(*args, **kwargs):
        original(*args, **kwargs)  # Consume candidate RNG then fail.
        raise RuntimeError("injected draw failure")

    monkeypatch.setattr(torch, "randperm", fail)
    with pytest.raises(RuntimeError):
        stream.advance_epoch()
    assert stream.snapshot() == before


def test_issuance_is_not_downstream_training_acknowledgment():
    stream = ShuffledBatchStream(config())
    first = stream.next_batch()
    # Consumer discards first without doing work; the stream cannot know that.
    after = stream.snapshot()
    assert after["cursor"] == len(first)
    restored = restore_snapshot(after)
    assert not set(first) & {i for batch in drain(restored) for i in batch}


def test_tail_drop_policy_retains_full_order_and_can_drop_entire_epoch():
    keep, drop = ShuffledBatchStream(config()), ShuffledBatchStream(config(drop_last=True))
    assert keep.snapshot()["permutation"] == drop.snapshot()["permutation"]
    assert list(map(len, drain(keep))) == [5, 5, 5, 5, 3]
    assert list(map(len, drain(drop))) == [5, 5, 5, 5]
    assert drop.cursor == 20
    saved = drop.snapshot()
    saved["cursor"] = 23
    with pytest.raises(ShuffleStateError):
        restore_snapshot(resign(saved))
    empty = ShuffledBatchStream(config(samples=3, batch_size=5, drop_last=True))
    assert empty.exhausted and empty.cursor == 0
    empty.advance_epoch()
    assert empty.epoch == 1 and empty.exhausted


def test_file_roundtrip_no_clobber_and_failed_publication(tmp_path, monkeypatch):
    stream = ShuffledBatchStream(config())
    stream.next_batch()
    path = tmp_path / "checkpoint.json"
    save_shuffle_checkpoint(stream, path)
    before = path.read_bytes()
    assert load_shuffle_checkpoint(path, stream.config).snapshot() == stream.snapshot()
    with pytest.raises(FileExistsError):
        save_shuffle_checkpoint(stream, path)
    assert path.read_bytes() == before
    import os

    def fail(*args):
        raise OSError("injected publication failure")

    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(OSError):
        save_shuffle_checkpoint(stream, tmp_path / "failed.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint.json"]
    assert stream.cursor == 5


@pytest.mark.parametrize("content", [b"not-json", b' {"x":1,"x":2}', b"\xff", b"[]", b"[" * 2000])
def test_invalid_file_fails_closed(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_bytes(content)
    with pytest.raises(ShuffleStateError):
        load_shuffle_checkpoint(path)


def test_missing_and_oversized_files_rejected(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(ShuffleStateError):
        load_shuffle_checkpoint(path)
    path.write_bytes(b" " * (MAX_SNAPSHOT_BYTES + 1))
    with pytest.raises(ShuffleStateError):
        load_shuffle_checkpoint(path)


def test_meaningful_wrong_cursor_seed_and_rng_controls():
    controls = _negative_controls()
    assert controls["all_detected"]
    assert controls["rewound_cursor"]["duplicate_count"] == 10
    assert controls["advanced_cursor"]["missing_count"] == 5
    assert all(controls["reset_same_seed_rng"].values())


def test_cli_fresh_process_matrix_stdout_and_no_overwrite(tmp_path):
    path = tmp_path / "evidence.json"
    command = [sys.executable, "-I", "-m", "pytorch_lab.shuffle_stream_cli", "--output", str(path)]
    run = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    report = json.loads(path.read_text())
    assert report == json.loads(run.stdout)
    assert report["all_checks_passed"] and report["resume_checkpoints"] == 99
    assert len(report["cases"]) == 6
    original = path.read_bytes()
    retry = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    assert retry.returncode == 2 and path.read_bytes() == original
