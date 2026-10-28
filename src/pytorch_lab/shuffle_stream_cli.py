"""Verify shuffled index issuance, not training, across separate CPU processes."""

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import torch

from pytorch_lab.shuffle_stream import (
    ShuffleConfig,
    ShuffledBatchStream,
    ShuffleStateError,
    _payload_digest,
    load_shuffle_checkpoint,
    restore_snapshot,
    save_shuffle_checkpoint,
)

EPOCHS = 3
SEEDS = (42, 7, 123)


def fixture_config(seed=42, drop_last=False):
    identity = hashlib.sha256(
        b"original-index-fixture-v1:ordered-integers-0-through-22"
    ).hexdigest()
    return ShuffleConfig(identity, seed=seed, drop_last=drop_last)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _oracle(config):
    """Expected sequence directly from randperm plus Python slicing, no stream."""
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    records = []
    limit = (
        config.samples - config.samples % config.batch_size if config.drop_last else config.samples
    )
    for epoch in range(EPOCHS):
        order = torch.randperm(config.samples, generator=generator, device="cpu").tolist()
        for start in range(0, limit, config.batch_size):
            records.append(
                {"epoch": epoch, "indices": order[start : min(start + config.batch_size, limit)]}
            )
    return records


def _remaining(stream):
    trace = []
    while True:
        if stream.exhausted:
            if stream.epoch == EPOCHS - 1:
                return trace
            stream.advance_epoch()
        else:
            epoch = stream.epoch
            trace.append({"epoch": epoch, "indices": list(stream.next_batch())})


def _produce(directory):
    fixtures = []
    for seed in SEEDS:
        for drop_last in (False, True):
            config = fixture_config(seed, drop_last)
            stream = ShuffledBatchStream(config)
            trace, cuts = [], []
            while True:
                name = f"s{seed}-d{int(drop_last)}-e{stream.epoch}-c{stream.cursor}.json"
                save_shuffle_checkpoint(stream, directory / name)
                cuts.append({"name": name, "offset": len(trace)})
                if stream.exhausted:
                    if stream.epoch == EPOCHS - 1:
                        break
                    stream.advance_epoch()
                else:
                    epoch = stream.epoch
                    trace.append({"epoch": epoch, "indices": list(stream.next_batch())})
            fixtures.append(
                {
                    "config": asdict(config),
                    "trace": trace,
                    "cuts": cuts,
                    "final_digest": stream.snapshot()["content_sha256"],
                }
            )
    return fixtures


def _consume(directory):
    fixtures = json.loads((directory / "fixtures.json").read_text())
    results = {}
    for fixture in fixtures:
        config = ShuffleConfig(**fixture["config"])
        for cut in fixture["cuts"]:
            stream = load_shuffle_checkpoint(directory / cut["name"], config)
            trace = _remaining(stream)
            results[cut["name"]] = {
                "trace": trace,
                "final_digest": stream.snapshot()["content_sha256"],
            }
    return results


def _invoke(mode, directory):
    run = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "pytorch_lab.shuffle_stream_cli",
            "--worker",
            mode,
            "--directory",
            str(directory),
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    return json.loads(run.stdout)


def _resign(payload):
    payload["content_sha256"] = _payload_digest(payload)
    return payload


def _negative_controls():
    config = fixture_config()
    stream = ShuffledBatchStream(config)
    prefix = list(stream.next_batch()) + list(stream.next_batch())
    saved = stream.snapshot()
    expected = _remaining(restore_snapshot(saved))
    controls = {}
    for name, cursor in (("rewound_cursor", 0), ("advanced_cursor", 15)):
        bad = copy.deepcopy(saved)
        bad["cursor"] = cursor
        trace = _remaining(restore_snapshot(_resign(bad)))
        first_epoch = prefix + [i for row in trace if row["epoch"] == 0 for i in row["indices"]]
        controls[name] = {
            "sequence_diverged": trace != expected,
            "duplicate_count": len(first_epoch) - len(set(first_epoch)),
            "missing_count": len(set(range(config.samples)) - set(first_epoch)),
        }
    bad = copy.deepcopy(saved)
    bad["generator_state"] = (
        torch.Generator(device="cpu").manual_seed(config.seed).get_state().tolist()
    )
    changed = _remaining(restore_snapshot(_resign(bad)))
    controls["reset_same_seed_rng"] = {
        "current_epoch_unchanged": [r for r in changed if r["epoch"] == 0]
        == [r for r in expected if r["epoch"] == 0],
        "next_epoch_diverged": [r for r in changed if r["epoch"] == 1]
        != [r for r in expected if r["epoch"] == 1],
    }
    try:
        restore_snapshot(saved, replace(config, seed=7))
    except ShuffleStateError:
        controls["wrong_expected_seed_rejected"] = True
    else:
        controls["wrong_expected_seed_rejected"] = False
    wrong_seed = _remaining(ShuffledBatchStream(replace(config, seed=7)))
    controls["wrong_seed_restart_diverged"] = wrong_seed != _oracle(config)
    controls["all_detected"] = (
        controls["rewound_cursor"]["sequence_diverged"]
        and controls["rewound_cursor"]["duplicate_count"] == 10
        and controls["advanced_cursor"]["sequence_diverged"]
        and controls["advanced_cursor"]["missing_count"] == 5
        and all(controls["reset_same_seed_rng"].values())
        and controls["wrong_expected_seed_rejected"]
        and controls["wrong_seed_restart_diverged"]
    )
    return controls


def run_experiment():
    cases = []
    with tempfile.TemporaryDirectory(prefix="shuffle-index-check-") as temporary:
        directory = Path(temporary)
        produced = _invoke("produce", directory)
        (directory / "fixtures.json").write_text(json.dumps(produced))
        resumed = _invoke("consume", directory)
        for fixture in produced:
            config = ShuffleConfig(**fixture["config"])
            oracle = _oracle(config)
            replay = all(
                resumed[cut["name"]]["trace"] == oracle[cut["offset"] :]
                and resumed[cut["name"]]["final_digest"] == fixture["final_digest"]
                for cut in fixture["cuts"]
            )
            conservation = all(
                len(indices) == config.usable_samples and len(set(indices)) == len(indices)
                for indices in (
                    [i for row in oracle if row["epoch"] == epoch for i in row["indices"]]
                    for epoch in range(EPOCHS)
                )
            )
            cases.append(
                {
                    "config": asdict(config),
                    "issued_batches": len(oracle),
                    "resume_checkpoints": len(fixture["cuts"]),
                    "expected_trace_sha256": _digest(oracle),
                    "uninterrupted_matches_oracle": fixture["trace"] == oracle,
                    "every_resume_matches_expected_suffix_and_final_state": replay,
                    "no_unintended_skips_or_duplicates": conservation,
                    "intentional_drops_per_epoch": config.samples - config.usable_samples,
                }
            )
    controls = _negative_controls()
    return {
        "protocol": {
            "scope": "single-owner CPU shuffled index issuance only; no training integration",
            "processes": "one fresh producer and one distinct fresh consumer for all checkpoints",
            "epochs": EPOCHS,
            "seeds": list(SEEDS),
            "cutpoints": "every batch boundary and explicit epoch start/end",
            "oracle": "independent local torch.randperm calls plus Python list slicing",
            "comparison": (
                "exact index sequences and final snapshot content digest in same environment"
            ),
            "limits": (
                "no workers/prefetch/distributed; no stock sampler or cross-version parity claim"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
        },
        "cases": cases,
        "resume_checkpoints": sum(case["resume_checkpoints"] for case in cases),
        "negative_controls": controls,
        "all_checks_passed": controls["all_detected"]
        and all(
            case["uninterrupted_matches_oracle"]
            and case["every_resume_matches_expected_suffix_and_final_state"]
            and case["no_unintended_skips_or_duplicates"]
            for case in cases
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("produce", "consume"), help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output is not None and (args.output.exists() or args.output.is_symlink()):
        parser.exit(2, "Output exists; choose a new path.\n")
    try:
        if args.worker:
            if args.directory is None:
                raise ShuffleStateError("worker requires directory")
            result = (
                _produce(args.directory) if args.worker == "produce" else _consume(args.directory)
            )
        else:
            result = run_experiment()
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                output.write(encoded)
        print(encoded, end="")
        if not args.worker and not result["all_checks_passed"]:
            parser.exit(1, "Index recovery checks failed; evidence retained.\n")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        parser.exit(2, "Index recovery check failed; no success report written.\n")


if __name__ == "__main__":
    main()
