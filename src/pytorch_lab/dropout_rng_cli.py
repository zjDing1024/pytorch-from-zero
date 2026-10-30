"""Fresh-process CPU Dropout replay and deliberately incomplete restoration controls."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from .dropout_rng import DropoutConfig, DropoutReplay, cpu_seed_state, mode_probe

SEEDS = (7, 42, 123)
CUTS = (1, 3, 5)
TOTAL = 8


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _worker(kind: str, seed: int, directory: Path) -> dict:
    config = DropoutConfig(seed=seed)
    if kind == "baseline":
        replay = DropoutReplay(config)
        traces = []
        for _ in range(TOTAL):
            traces.append(replay.step())
            if replay.steps in CUTS:
                replay.save(directory / f"{seed}-{replay.steps}.json")
        return {"traces": traces, "final": replay.snapshot()}
    result = {}
    for cut in CUTS:
        replay = DropoutReplay.load(directory / f"{seed}-{cut}.json", expected_config=config)
        traces = [replay.step() for _ in range(TOTAL - cut)]
        result[str(cut)] = {"traces": traces, "final": replay.snapshot()}
    return result


def _run_worker(kind, seed, directory):
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytorch_lab.dropout_rng_cli",
            "--worker",
            kind,
            "--seed",
            str(seed),
            "--directory",
            str(directory),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(run.stdout)


def run_experiment() -> dict:
    torch.set_num_threads(1)
    caller = torch.get_rng_state().clone()
    cases, negatives = [], []
    with tempfile.TemporaryDirectory(prefix="dropout-replay-") as temp:
        directory = Path(temp)
        for seed in SEEDS:
            baseline = _run_worker("baseline", seed, directory)
            restored = _run_worker("recover", seed, directory)
            for cut in CUTS:
                expected = baseline["traces"][cut:]
                actual = restored[str(cut)]
                cases.append(
                    {
                        "seed": seed,
                        "cut": cut,
                        "compared_steps": len(expected),
                        "trace_exact": expected == actual["traces"],
                        "final_snapshot_exact": baseline["final"] == actual["final"],
                        "baseline_trace_sha256": _hash(expected),
                        "restored_trace_sha256": _hash(actual["traces"]),
                    }
                )
            state = DropoutReplay.load(directory / f"{seed}-3.json").snapshot()
            expected = baseline["traces"][3]
            seed_only = DropoutReplay.from_snapshot(state)
            seed_only._rng = cpu_seed_state(seed)
            seed_trace = seed_only.step()
            wrong_order = DropoutReplay.from_snapshot(state)
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(wrong_order._rng)
                # Intentionally wrong: a constructor consumes the captured default RNG.
                torch.nn.Linear(4, 8, dtype=torch.float64)
                wrong_order._rng = torch.get_rng_state().clone()
            order_trace = wrong_order.step()
            good = DropoutReplay.from_snapshot(state)
            with torch.random.fork_rng(devices=[]):
                torch.rand(137)  # Unrelated global draws cannot alter the owned state.
                isolated = good.step()
            negatives.append(
                {
                    "seed": seed,
                    "seed_only_output_differs": seed_trace["output"] != expected["output"],
                    "restore_before_init_output_differs": order_trace["output"]
                    != expected["output"],
                    "unrelated_global_draws_preserve_replay": isolated == expected,
                    "seed_only_loss_abs_difference": abs(seed_trace["loss"] - expected["loss"]),
                    "restore_before_init_loss_abs_difference": abs(
                        order_trace["loss"] - expected["loss"]
                    ),
                }
            )
    modes = mode_probe()
    checks = {
        "fresh_process_traces_exact": all(x["trace_exact"] for x in cases),
        "fresh_process_final_snapshots_exact": all(x["final_snapshot_exact"] for x in cases),
        "seed_only_controls_detected": all(x["seed_only_output_differs"] for x in negatives),
        "restore_order_controls_detected": all(
            x["restore_before_init_output_differs"] for x in negatives
        ),
        "global_noise_isolated": all(
            x["unrelated_global_draws_preserve_replay"] for x in negatives
        ),
        "no_grad_same_random_forward": modes["train"]["output"] == modes["no_grad"]["output"],
        "no_grad_still_consumes_rng": modes["no_grad"]["rng_changed"],
        "no_grad_disables_recording": not modes["no_grad"]["requires_grad"],
        "eval_does_not_consume_rng": not modes["eval"]["rng_changed"],
        "eval_can_record_gradients": modes["eval"]["requires_grad"],
        "caller_cpu_rng_preserved": torch.equal(caller, torch.get_rng_state()),
    }
    return {
        "protocol": "cpu-dropout-forward-backward-replay-v1",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "dtype": "float64",
        "device": "cpu",
        "optimizer_updates": 0,
        "worker_processes": 6,
        "recovery_cases": len(cases),
        "compared_steps": sum(x["compared_steps"] for x in cases),
        "cases": cases,
        "negative_controls": negatives,
        "mode_probe": modes,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "limitations": [
            "fixed buffer-free model, no optimizer updates",
            "single-threaded CPU only",
            "trusted own snapshots only; checksum is not authentication",
            "no cross-version, cross-platform, CUDA, worker, or joint-training claim",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("baseline", "recover"), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42, help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.worker:
        if args.directory is None:
            parser.error("worker requires directory")
        report = _worker(args.worker, args.seed, args.directory)
    else:
        if args.output and args.output.exists():
            parser.error("output already exists")
        report = run_experiment()
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")
    if not args.worker and not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
