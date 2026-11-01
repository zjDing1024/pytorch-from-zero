"""Fresh-process input-Dropout joint-training recovery evidence.

Six workers compare full committed-state and trace suffixes. Explicit Python
failpoints do not simulate process killing, power loss or external transactions.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from pytorch_lab.checkpoint import CheckpointError
from pytorch_lab.stochastic_joint import (
    InjectedFailure,
    StochasticConfig,
    StochasticJointTrainer,
    load_stochastic_checkpoint,
    save_stochastic_checkpoint,
)

SEEDS = (7, 42, 123)
EPOCHS = 3
CUTS = (1, 3, 5)
FAULT_CASES = (
    ("after_fetch", 1),
    ("after_forward", 1),
    ("after_backward", 3),
    ("after_update", 4),
    ("before_commit", 7),
)


def _json_values(value):
    if isinstance(value, torch.Tensor):
        return value.detach().tolist()
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_values(item) for item in value]
    return value


def _file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace(trainer):
    states = [trainer.snapshot()["content_sha256"]]
    batches, traces = [], []
    while trainer.epoch < EPOCHS:
        batches.append(list(trainer.step_microbatch()))
        traces.append(_json_values(trainer.last_trace))
        states.append(trainer.snapshot()["content_sha256"])
    return {"states": states, "batches": batches, "traces": traces}


def _baseline(config, directory):
    trainer = StochasticJointTrainer(config)
    states, batches, traces = [], [], []
    cuts = set(CUTS) | {cut for _, cut in FAULT_CASES}
    while True:
        states.append(trainer.snapshot()["content_sha256"])
        if trainer.committed_batches in cuts:
            save_stochastic_checkpoint(trainer, directory / f"cut-{trainer.committed_batches}.pt")
        if trainer.epoch == EPOCHS:
            break
        batches.append(list(trainer.step_microbatch()))
        traces.append(_json_values(trainer.last_trace))
    return {"states": states, "batches": batches, "traces": traces}


def _resume(config, directory):
    normal = [
        {"cut": cut, **_trace(load_stochastic_checkpoint(directory / f"cut-{cut}.pt", config))}
        for cut in CUTS
    ]
    failures = []
    for fault, cut in FAULT_CASES:
        path = directory / f"cut-{cut}.pt"
        before_file = _file_digest(path)
        trainer = load_stochastic_checkpoint(path, config)
        before_rng = torch.get_rng_state().clone()
        observed = False
        try:
            trainer.step_microbatch(fault_at=fault)
        except InjectedFailure:
            observed = True
        caller_rng_restored = torch.equal(before_rng, torch.get_rng_state())
        refused = {}
        attempted = directory / f"unsafe-{fault}.pt"
        for name, operation in (
            ("step", trainer.step_microbatch),
            ("snapshot", trainer.snapshot),
            ("save", lambda t=trainer, p=attempted: save_stochastic_checkpoint(t, p)),
            ("train", lambda t=trainer: t.train_to(EPOCHS)),
        ):
            try:
                operation()
            except CheckpointError:
                refused[name] = True
            else:
                refused[name] = False
        failures.append(
            {
                "fault": fault,
                "cut": cut,
                "injected_failure_observed": observed,
                "caller_rng_restored": caller_rng_restored,
                "unsafe_operations_refused": refused,
                "unsafe_file_absent": not attempted.exists(),
                "saved_checkpoint_unchanged": before_file == _file_digest(path),
                **_trace(load_stochastic_checkpoint(path, config)),
            }
        )
    return {"normal": normal, "failures": failures}


def _worker(mode, seed, directory):
    torch.set_num_threads(1)
    config = StochasticConfig(seed=seed, model_seed=seed)
    return {"baseline": _baseline, "resume": _resume}[mode](config, directory)


def _invoke(mode, seed, directory):
    command = [
        sys.executable,
        "-I",
        "-m",
        "pytorch_lab.stochastic_joint_cli",
        "--worker",
        mode,
        "--seed",
        str(seed),
        "--directory",
        str(directory),
    ]
    run = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if run.returncode != 0:
        raise RuntimeError(f"{mode} worker exited {run.returncode}; expected 0")
    return json.loads(run.stdout)


def _compare(baseline, resumed):
    cut = resumed["cut"]
    return {
        "every_safe_point_exact": resumed["states"] == baseline["states"][cut:],
        "committed_indices_exact": resumed["batches"] == baseline["batches"][cut:],
        "every_output_loss_raw_gradient_trace_exact": resumed["traces"] == baseline["traces"][cut:],
    }


def run_experiment():
    recoveries, failures, coverage = [], [], []
    with tempfile.TemporaryDirectory(prefix="stochastic-joint-") as temp:
        for seed in SEEDS:
            directory = Path(temp) / str(seed)
            directory.mkdir()
            baseline = _invoke("baseline", seed, directory)
            resumed = _invoke("resume", seed, directory)
            if [case["cut"] for case in resumed["normal"]] != list(CUTS):
                raise RuntimeError("recovery cut identity mismatch")
            if [(case["fault"], case["cut"]) for case in resumed["failures"]] != list(FAULT_CASES):
                raise RuntimeError("failure identity mismatch")
            batches = baseline["batches"]
            coverage.append(
                {
                    "seed": seed,
                    "committed_microbatches": len(batches),
                    "committed_sample_visits": sum(map(len, batches)),
                    "each_epoch_all_samples_once": [
                        sorted(i for batch in batches[start : start + 5] for i in batch)
                        == list(range(23))
                        for start in (0, 5, 10)
                    ],
                    "final_state_sha256": baseline["states"][-1],
                    "reference_trajectory": baseline,
                }
            )
            for case in resumed["normal"]:
                checks = _compare(baseline, case)
                recoveries.append(
                    {
                        "seed": seed,
                        "cut": case["cut"],
                        "compared_safe_points": len(case["states"]),
                        "compared_steps": len(case["traces"]),
                        "checks": checks,
                        "all_checks_passed": all(checks.values()),
                    }
                )
            for case in resumed["failures"]:
                checks = {
                    **_compare(baseline, case),
                    "injected_failure_observed": case["injected_failure_observed"],
                    "caller_rng_restored": case["caller_rng_restored"],
                    "all_unsafe_operations_refused": all(
                        case["unsafe_operations_refused"].values()
                    ),
                    "unsafe_file_absent": case["unsafe_file_absent"],
                    "saved_checkpoint_unchanged": case["saved_checkpoint_unchanged"],
                }
                failures.append(
                    {
                        "seed": seed,
                        "fault": case["fault"],
                        "cut": case["cut"],
                        "compared_safe_points": len(case["states"]),
                        "compared_steps": len(case["traces"]),
                        "checks": checks,
                        "all_checks_passed": all(checks.values()),
                    }
                )
    checks = {
        "all_9_recovery_cases_exact": len(recoveries) == 9
        and all(case["all_checks_passed"] for case in recoveries),
        "all_15_failure_recovery_cases_exact": len(failures) == 15
        and all(case["all_checks_passed"] for case in failures),
        "all_epochs_cover_every_sample_once": all(
            all(case["each_epoch_all_samples_once"]) for case in coverage
        ),
        "all_baselines_complete_15_microbatches_69_visits": all(
            case["committed_microbatches"] == 15 and case["committed_sample_visits"] == 69
            for case in coverage
        ),
    }
    return {
        "schema_version": 1,
        "experiment": "input_dropout_joint_training_recovery",
        "executed_by": "AI assistant; learner mastery unassessed",
        "environment": {"python": platform.python_version(), "pytorch": str(torch.__version__)},
        "protocol": {
            "seeds": list(SEEDS),
            "model_rng_seed": "same as each data seed",
            "shuffle_seed": 19,
            "samples": 23,
            "microbatch_size": 5,
            "accumulation_steps": 3,
            "dropout_p": 0.35,
            "epochs": EPOCHS,
            "dtype": "float64",
            "device": "cpu",
            "worker_processes": {"baseline": 3, "resume_and_fault": 3},
            "comparison": "exact typed-state digest, index suffix and full per-step trace",
            "fault_mode": "explicit Python exceptions; workers then reload old whole checkpoints",
            "excluded": [
                "process kill or power loss",
                "cross-runtime snapshot exchange",
                "concurrent RNG consumers",
                "external effects exactly once",
                "GPU or AMP or DDP",
            ],
        },
        "recovery_cases": recoveries,
        "failure_recovery_cases": failures,
        "baseline_coverage": coverage,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("baseline", "resume"), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42, help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output is not None and (args.output.exists() or args.output.is_symlink()):
        parser.exit(2, "Output already exists; choose a new path.\n")
    if args.worker and args.directory is None:
        parser.error("worker directory required")
    try:
        result = (
            _worker(args.worker, args.seed, args.directory) if args.worker else run_experiment()
        )
        encoded = json.dumps(result, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                output.write(encoded + "\n")
        print(encoded)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Stochastic joint experiment failed: {type(error).__name__}.\n")
    if not args.worker and not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
