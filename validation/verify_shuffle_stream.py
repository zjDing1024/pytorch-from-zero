"""Independent CPU index-stream oracle and adversarial review; JSON on stdout."""

import copy
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from pytorch_lab import shuffle_stream as implementation
from pytorch_lab.shuffle_stream import (
    MAX_EPOCH,
    MAX_SAMPLES,
    MAX_SNAPSHOT_BYTES,
    ShuffleConfig,
    ShuffledBatchStream,
    ShuffleStateError,
    load_shuffle_checkpoint,
    restore_snapshot,
    save_shuffle_checkpoint,
)

IDENTITY = "3" * 64


def oracle(n, batch, drop, seed, epochs=5):
    """Independent direct primitive + slicing, never stream implementation output."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    expected = []
    for _ in range(epochs):
        full = torch.randperm(n, generator=generator).tolist()
        usable = (n // batch) * batch if drop else n
        chunks = [
            tuple(full[start : min(start + batch, usable)]) for start in range(0, usable, batch)
        ]
        expected.append((full, chunks, generator.get_state().tolist()))
    return expected


def drain(stream):
    batches = []
    while True:
        try:
            batches.append(stream.next_batch())
        except StopIteration:
            return batches


def must_reject(call, error_type=ShuffleStateError):
    try:
        call()
    except error_type:
        return
    raise AssertionError("invalid operation accepted")


def grid():
    count = points = future_checks = 0
    global_before = torch.get_rng_state().clone()
    for n, batch, drop, seed in itertools.product(
        (0, 1, 2, 7, 11, 23), (1, 2, 5, 30), (False, True), (0, 7, 42)
    ):
        config = ShuffleConfig(
            samples=n, batch_size=batch, drop_last=drop, seed=seed, dataset_sha256=IDENTITY
        )
        stream = ShuffledBatchStream(config)
        expected = oracle(n, batch, drop, seed)
        for epoch in range(3):
            chunks = expected[epoch][1]
            issued = []
            for cut in range(len(chunks) + 1):
                snapshot = stream.snapshot()
                frozen = copy.deepcopy(snapshot)
                assert stream.epoch == epoch
                assert stream.cursor == sum(map(len, chunks[:cut]))
                assert snapshot["permutation"] == expected[epoch][0]
                assert snapshot["generator_state"] == expected[epoch][2]
                restored = restore_snapshot(snapshot, config)
                assert drain(restored) == chunks[cut:], (n, batch, drop, seed, epoch, cut)
                assert snapshot == frozen
                # Exhaustion must not silently create or consume next epoch.
                at_end = restored.snapshot()
                assert drain(restored) == []
                assert restored.snapshot() == at_end
                for future in (epoch + 1, epoch + 2):
                    restored.advance_epoch()
                    assert drain(restored) == expected[future][1]
                    future_checks += 1
                assert snapshot == frozen
                points += 1
                if cut < len(chunks):
                    before_rejected = stream.snapshot()
                    must_reject(stream.advance_epoch)
                    assert stream.snapshot() == before_rejected
                    one = stream.next_batch()
                    assert one == chunks[cut]
                    issued.extend(one)
            assert issued == list(itertools.chain.from_iterable(chunks))
            assert len(issued) == len(set(issued))
            assert set(issued) == set(expected[epoch][0][: len(issued)])
            ended = stream.snapshot()
            assert drain(stream) == []
            assert stream.snapshot() == ended
            if epoch < 2:
                stream.advance_epoch()
        count += 1
    assert torch.equal(torch.get_rng_state(), global_before)
    return {
        "oracle_configurations": count,
        "recovery_cutpoints": points,
        "future_epoch_comparisons": future_checks,
        "exact_expected_retained_index_order": True,
        "global_rng_unchanged": True,
    }


def seal(snapshot):
    """Recompute documented JSON checksum without production digest helper."""
    body = {key: value for key, value in snapshot.items() if key != "content_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    snapshot["content_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return snapshot


def adversarial():
    config = ShuffleConfig(dataset_sha256=IDENTITY)
    stream = ShuffledBatchStream(config)
    stream.next_batch()
    stream.next_batch()
    snapshot = stream.snapshot()
    expected = oracle(23, 5, False, 42)
    failures = []

    def mutation(field, value):
        candidate = copy.deepcopy(snapshot)
        candidate[field] = value
        failures.append(seal(candidate))

    for field, value in (
        ("format", "unrecognized"),
        ("format", 1),
        ("version", True),
        ("version", 2),
        ("version", 1.0),
        ("torch_version", "different-runtime"),
        ("torch_version", 1),
        ("config", []),
        ("epoch", True),
        ("epoch", -1),
        ("epoch", MAX_EPOCH + 1),
        ("cursor", True),
        ("cursor", -1),
        ("cursor", 1),
        ("cursor", 21),
        ("cursor", 24),
        ("permutation", list(range(22))),
        ("permutation", [0] * 23),
        ("permutation", [23] + list(range(1, 23))),
        ("permutation", [0.0] + list(range(1, 23))),
        ("permutation", [False] + list(range(1, 23))),
        ("permutation", tuple(range(23))),
        ("generator_state", []),
        ("generator_state", tuple(snapshot["generator_state"])),
        ("generator_state", [0] * len(snapshot["generator_state"])),
        ("generator_state", torch.Generator().manual_seed(7).get_state().tolist()),
    ):
        mutation(field, value)
    for bad in (-1, 256, 1.0, True):
        state = snapshot["generator_state"].copy()
        state[30] = bad
        mutation("generator_state", state)
    for field, value in (
        ("samples", True),
        ("samples", -1),
        ("samples", MAX_SAMPLES + 1),
        ("batch_size", 0),
        ("batch_size", True),
        ("batch_size", MAX_SAMPLES + 1),
        ("seed", -1),
        ("seed", True),
        ("seed", 2**32),
        ("drop_last", 1),
        ("dataset_sha256", "A" * 64),
        ("dataset_sha256", "3" * 63),
        ("dataset_sha256", 3),
    ):
        malformed = copy.deepcopy(snapshot["config"])
        malformed[field] = value
        mutation("config", malformed)
    for target in (None, "config"):
        missing = copy.deepcopy(snapshot)
        extra = copy.deepcopy(snapshot)
        if target is None:
            del missing["epoch"]
            extra["unknown"] = 1
        else:
            del missing[target]["seed"]
            extra[target]["unknown"] = 1
        failures.extend((seal(missing), seal(extra)))
    bad_digest = copy.deepcopy(snapshot)
    bad_digest["content_sha256"] = "0" * 64
    failures.extend((bad_digest, [], None))
    for candidate in failures:
        must_reject(lambda candidate=candidate: restore_snapshot(candidate))
    mismatch_count = 0
    for field, value in (
        ("samples", 22),
        ("batch_size", 4),
        ("seed", 7),
        ("drop_last", True),
        ("dataset_sha256", "4" * 64),
    ):
        must_reject(
            lambda field=field, value=value: restore_snapshot(
                snapshot, replace(config, **{field: value})
            )
        )
        mismatch_count += 1
    must_reject(lambda: restore_snapshot(snapshot, {}))
    drop_config = replace(config, drop_last=True)
    invalid_drop_tail = ShuffledBatchStream(drop_config).snapshot()
    invalid_drop_tail["cursor"] = 23
    must_reject(lambda: restore_snapshot(seal(invalid_drop_tail), drop_config))
    failures.append(invalid_drop_tail)

    # Both outgoing snapshots and incoming restore payloads are detached.
    detached = stream.snapshot()
    detached["permutation"].reverse()
    detached["generator_state"][30] ^= 1
    detached["config"]["seed"] = 7
    assert stream.snapshot() == snapshot
    incoming = copy.deepcopy(snapshot)
    restored = restore_snapshot(incoming, config)
    incoming["permutation"].reverse()
    incoming["generator_state"][30] ^= 1
    incoming["config"]["seed"] = 7
    assert restored.snapshot() == snapshot
    assert drain(restored) == expected[0][1][2:]

    # Discarded batches have still been issued. An earlier snapshot, not a later
    # snapshot, is needed to reissue them. No consumer/training commit is implied.
    before = stream.snapshot()
    abandoned = stream.next_batch()
    assert abandoned == expected[0][1][2]
    assert restore_snapshot(before, config).next_batch() == abandoned
    assert restore_snapshot(stream.snapshot(), config).next_batch() == expected[0][1][3]

    # Checksum-recomputed, structurally valid corruption controls must actually fail the
    # expected-sequence oracle; checksums cannot authenticate processing history.
    reset = copy.deepcopy(snapshot)
    reset["cursor"] = 0
    repeated = drain(restore_snapshot(seal(reset), config))
    assert repeated != expected[0][1][2:]
    assert set(expected[0][1][0]).issubset(set(itertools.chain.from_iterable(repeated)))
    shifted = copy.deepcopy(snapshot)
    shifted["cursor"] = 15
    skipped = drain(restore_snapshot(seal(shifted), config))
    assert skipped != expected[0][1][2:]
    assert not set(expected[0][1][2]).intersection(itertools.chain.from_iterable(skipped))
    rng_reset = copy.deepcopy(snapshot)
    rng_reset["generator_state"] = torch.Generator().manual_seed(42).get_state().tolist()
    wrong_rng = restore_snapshot(seal(rng_reset), config)
    assert drain(wrong_rng) == expected[0][1][2:]
    wrong_rng.advance_epoch()
    assert drain(wrong_rng) != expected[1][1]

    # Interleaved unrelated draws may change the process-global RNG, but not
    # this stream's next-epoch sequence. fork_rng restores our test's own edits.
    independent = restore_snapshot(snapshot, config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8128)
        torch.rand(137)
        assert drain(independent) == expected[0][1][2:]
        torch.rand(29)
        independent.advance_epoch()
        assert drain(independent) == expected[1][1]

    # Restore must not regenerate a permutation. A failed advance must preserve
    # both old permutation and RNG even if the failing draw consumed local RNG.
    with patch("torch.randperm", side_effect=AssertionError("unexpected random draw")):
        assert restore_snapshot(snapshot, config).snapshot() == snapshot
    exhausted = restore_snapshot(snapshot, config)
    drain(exhausted)
    ended = exhausted.snapshot()
    real_randperm = torch.randperm

    def failing_draw(*args, **kwargs):
        real_randperm(*args, **kwargs)
        raise RuntimeError("injected post-draw failure")

    with patch("torch.randperm", side_effect=failing_draw):
        must_reject(exhausted.advance_epoch, RuntimeError)
    assert exhausted.snapshot() == ended
    exhausted.advance_epoch()
    assert drain(exhausted) == expected[1][1]
    maxed = copy.deepcopy(ended)
    maxed["epoch"] = MAX_EPOCH
    limited = restore_snapshot(seal(maxed), config)
    must_reject(limited.advance_epoch)
    assert limited.snapshot() == maxed
    for n, batch, drop in ((0, 5, False), (1, 5, True), (23, 5, False)):
        with patch("torch.randperm", wraps=real_randperm) as draw:
            counted = ShuffledBatchStream(
                replace(config, samples=n, batch_size=batch, drop_last=drop)
            )
            for epoch in range(3):
                assert draw.call_count == epoch + 1
                drain(counted)
                restored = restore_snapshot(counted.snapshot(), counted.config)
                assert drain(restored) == []
                assert draw.call_count == epoch + 1
                if epoch < 2:
                    counted.advance_epoch()
    return {
        "malformed_snapshots_rejected": len(failures),
        "expected_config_mismatches_rejected": mismatch_count + 1,
        "snapshot_isolation": True,
        "abandoned_batch_is_still_issued": True,
        "wrong_cursor_reset_detected": True,
        "wrong_cursor_forward_shift_detected": True,
        "wrong_rng_reset_detected_in_future_epoch": True,
        "unrelated_global_rng_draws_do_not_change_stream": True,
        "restore_does_not_draw_rng": True,
        "failed_advance_is_atomic": True,
        "epoch_limit_enforced": True,
        "exactly_one_randperm_call_per_explicit_epoch": True,
    }


def files_and_fresh_process():
    config = ShuffleConfig(dataset_sha256=IDENTITY)
    stream = ShuffledBatchStream(config)
    drain(stream)
    stream.advance_epoch()
    stream.next_batch()
    stream.next_batch()
    snapshot = stream.snapshot()
    malformed_files = 0
    with tempfile.TemporaryDirectory(prefix="shuffle-review-") as directory:
        directory = Path(directory)
        checkpoint = directory / "trusted-own.json"
        save_shuffle_checkpoint(stream, checkpoint)
        original = checkpoint.read_bytes()
        must_reject(lambda: save_shuffle_checkpoint(stream, checkpoint), FileExistsError)
        assert checkpoint.read_bytes() == original
        assert list(directory.iterdir()) == [checkpoint]
        assert load_shuffle_checkpoint(checkpoint, config).snapshot() == snapshot
        for index, content in enumerate(
            (
                b"{",
                b"[]",
                b'{"version":1,"version":1}',
                json.dumps(snapshot)
                .replace('"samples": 23', '"samples": 23, "samples": 23')
                .encode(),
                ("[" * 1100 + "0" + "]" * 1100).encode(),
                b" " * (MAX_SNAPSHOT_BYTES + 1),
                b"\xff\xfe",
            )
        ):
            invalid = directory / f"invalid-{index}.json"
            invalid.write_bytes(content)
            must_reject(lambda invalid=invalid: load_shuffle_checkpoint(invalid, config))
            malformed_files += 1
        must_reject(lambda: load_shuffle_checkpoint(directory / "missing.json", config))
        malformed_files += 1
        child = """
import json, sys, torch
from pytorch_lab.shuffle_stream import ShuffleConfig, load_shuffle_checkpoint
before = torch.get_rng_state().clone()
cfg = ShuffleConfig(dataset_sha256="3" * 64)
stream = load_shuffle_checkpoint(sys.argv[1], cfg)
outputs = []
for offset in range(3):
    batches = []
    while True:
        try:
            batches.append(stream.next_batch())
        except StopIteration:
            break
    outputs.append(batches)
    if offset < 2:
        stream.advance_epoch()
assert torch.equal(before, torch.get_rng_state())
print(json.dumps(outputs))
"""
        result = subprocess.run(
            [sys.executable, "-c", child, str(checkpoint)],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        actual = json.loads(result.stdout)
        reference = oracle(23, 5, False, 42)
        wanted = [reference[1][1][2:], reference[2][1], reference[3][1]]
        assert actual == json.loads(json.dumps(wanted))
    return {
        "invalid_files_rejected": malformed_files,
        "no_clobber_save": True,
        "fresh_process_recovery": True,
    }


def run():
    torch.set_num_threads(1)
    global_before = torch.get_rng_state().clone()
    report = grid()
    report.update(adversarial())
    report.update(files_and_fresh_process())
    assert torch.equal(torch.get_rng_state(), global_before)
    report["all_checks_passed"] = True
    report["scope"] = "same-runtime CPU index issuance only; no training recovery"
    report["torch_version"] = str(torch.__version__)
    report["python_version"] = platform.python_version()
    report["source_sha256"] = {
        "src/pytorch_lab/shuffle_stream.py": hashlib.sha256(
            Path(implementation.__file__).read_bytes()
        ).hexdigest(),
        "validation/verify_shuffle_stream.py": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
