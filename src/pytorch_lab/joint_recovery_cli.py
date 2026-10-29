"""Fresh-process fail-closed joint stream/trainer recovery experiment.

Each fault worker checks four injected Python exceptions, then exits nonzero.
This is not a process-kill, concurrent-consumer, or storage durability test.
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

from pytorch_lab.checkpoint import CheckpointError, _content_digest
from pytorch_lab.joint_recovery import (
    InjectedFailure,
    JointConfig,
    JointTrainer,
    load_joint_checkpoint,
    save_joint_checkpoint,
)

# Frozen before execution: 23 samples -> 5 microbatches, 2 updates per epoch.
# These cuts exercise a pending window, existing momentum, a short tail update,
# and an update in epoch 1. Recovery is exact within the same CPU runtime only.
SEEDS = (42, 7, 123)
EPOCHS = 3
FAULT_CASES = (
    ("after_fetch", 1),
    ("after_backward", 3),
    ("after_update", 4),
    ("before_commit", 7),
)
FAULT_EXIT = 23


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence_digest(batches: list) -> str:
    return hashlib.sha256(json.dumps(batches, separators=(",", ":")).encode()).hexdigest()


def _model_values(snapshot: dict) -> list[float]:
    return [
        value
        for name in ("weight", "bias")
        for value in snapshot["model"][name].reshape(-1).tolist()
    ]


def _trace(trainer: JointTrainer) -> dict:
    states = [trainer.snapshot()["content_sha256"]]
    batches = []
    while trainer.epoch < EPOCHS:
        batches.append(list(trainer.step_microbatch()))
        states.append(trainer.snapshot()["content_sha256"])
    return {
        "states": states,
        "batches": batches,
        "final_model": _model_values(trainer.snapshot()),
    }


def _baseline(config: JointConfig, directory: Path) -> dict:
    trainer = JointTrainer(config)
    states, batches = [], []
    cuts = {cut for _, cut in FAULT_CASES}
    while True:
        states.append(trainer.snapshot()["content_sha256"])
        if trainer.committed_batches in cuts:
            save_joint_checkpoint(trainer, directory / f"cut-{trainer.committed_batches}.pt")
        if trainer.epoch == EPOCHS:
            break
        batches.append(list(trainer.step_microbatch()))
    return {
        "states": states,
        "batches": batches,
        "final_model": _model_values(trainer.snapshot()),
    }


def _faults(config: JointConfig, directory: Path) -> dict:
    cases = []
    for fault, cut in FAULT_CASES:
        path = directory / f"cut-{cut}.pt"
        original = _file_digest(path)
        trainer = load_joint_checkpoint(path, config)
        saved = trainer.snapshot()
        attempted = directory / f"unsafe-{fault}.pt"
        observed = False
        try:
            trainer.step_microbatch(fault_at=fault)
        except InjectedFailure:
            observed = True
        # Diagnostic reads only: these private, partially mutated values are
        # never accepted as a recovery snapshot or combined with another save.
        stream_after_failure = trainer._stream.snapshot()
        current_gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in trainer._model.named_parameters()
        }
        model_changed = _content_digest(dict(trainer._model.state_dict())) != _content_digest(
            saved["model"]
        )
        gradients_changed = _content_digest(current_gradients) != _content_digest(
            saved["pending_gradients"]
        )
        stage_effects = {
            "stream_issuance_advanced": (
                stream_after_failure["epoch"],
                stream_after_failure["cursor"],
            )
            != (saved["epoch"], saved["cursor"]),
            "trainer_progress_not_committed": (
                trainer.epoch,
                trainer.cursor,
                trainer.committed_batches,
            )
            == (saved["epoch"], saved["cursor"], cut),
            "model_changed": model_changed,
            "gradients_changed": gradients_changed,
        }
        expected_stage = {
            "after_fetch": not model_changed and not gradients_changed,
            "after_backward": not model_changed and gradients_changed,
            "after_update": model_changed,
            "before_commit": model_changed,
        }[fault]
        refused = {}
        for name, operation in (
            ("snapshot", trainer.snapshot),
            ("save", lambda t=trainer, p=attempted: save_joint_checkpoint(t, p)),
            ("step", trainer.step_microbatch),
            ("train", lambda t=trainer: t.train_to(EPOCHS)),
        ):
            try:
                operation()
            except CheckpointError:
                refused[name] = True
            else:
                refused[name] = False
        cursor = saved["cursor"]
        issued = saved["stream"]["permutation"][cursor : cursor + config.microbatch_size]
        cases.append(
            {
                "fault": fault,
                "cut": cut,
                "injected_failure_observed": observed,
                "stage_effects": stage_effects,
                "expected_stage_effect_observed": expected_stage
                and stage_effects["stream_issuance_advanced"]
                and stage_effects["trainer_progress_not_committed"],
                "unsafe_operations_refused": refused,
                "unsafe_file_absent": not attempted.exists(),
                "saved_checkpoint_unchanged": _file_digest(path) == original,
                "attempted_indices": issued,
                "saved_epoch": saved["epoch"],
                "saved_cursor": cursor,
                "saved_pending_samples": saved["pending_samples"],
                "saved_updates": saved["updates"],
            }
        )
    return {"cases": cases}


def _resumed(config: JointConfig, directory: Path) -> dict:
    return {
        "cases": [
            {
                "fault": fault,
                "cut": cut,
                **_trace(load_joint_checkpoint(directory / f"cut-{cut}.pt", config)),
            }
            for fault, cut in FAULT_CASES
        ]
    }


def _worker(mode: str, seed: int, directory: Path) -> dict:
    torch.set_num_threads(1)
    config = JointConfig(seed=seed)
    return {"baseline": _baseline, "fault": _faults, "resume": _resumed}[mode](config, directory)


def _invoke(mode: str, seed: int, directory: Path) -> dict:
    command = [
        sys.executable,
        "-I",
        "-m",
        "pytorch_lab.joint_recovery_cli",
        "--worker",
        mode,
        "--seed",
        str(seed),
        "--directory",
        str(directory),
    ]
    run = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    expected = FAULT_EXIT if mode == "fault" else 0
    if run.returncode != expected:
        raise RuntimeError(f"{mode} worker exited {run.returncode}; expected {expected}")
    return json.loads(run.stdout)


def run_experiment() -> dict:
    cases, coverage = [], []
    with tempfile.TemporaryDirectory(prefix="joint-recovery-") as temp:
        for seed in SEEDS:
            directory = Path(temp) / str(seed)
            directory.mkdir()
            baseline = _invoke("baseline", seed, directory)
            faults = _invoke("fault", seed, directory)["cases"]
            resumed = _invoke("resume", seed, directory)["cases"]
            batches = baseline["batches"]
            epoch_coverage = [
                sorted(index for batch in batches[start : start + 5] for index in batch)
                == list(range(23))
                for start in (0, 5, 10)
            ]
            coverage.append(
                {
                    "seed": seed,
                    "committed_microbatches": len(batches),
                    "committed_sample_visits": sum(map(len, batches)),
                    "each_epoch_all_samples_once": epoch_coverage,
                    "committed_sequence_sha256": _sequence_digest(batches),
                }
            )
            for (fault, cut), failed, recovery in zip(FAULT_CASES, faults, resumed, strict=True):
                if failed["fault"] != fault or recovery["fault"] != fault:
                    raise RuntimeError("worker fault-case identity mismatch")
                if failed["cut"] != cut or recovery["cut"] != cut:
                    raise RuntimeError("worker checkpoint identity mismatch")
                checks = {
                    "injected_failure_observed": failed["injected_failure_observed"],
                    "expected_stage_effect_observed": failed["expected_stage_effect_observed"],
                    "all_unsafe_operations_refused": all(
                        failed["unsafe_operations_refused"].values()
                    ),
                    "unsafe_file_absent": failed["unsafe_file_absent"],
                    "saved_checkpoint_unchanged": failed["saved_checkpoint_unchanged"],
                    "every_remaining_state_exact": recovery["states"] == baseline["states"][cut:],
                    "committed_indices_exact": recovery["batches"] == batches[cut:],
                    "first_replayed_batch_is_attempted_batch": recovery["batches"][0]
                    == failed["attempted_indices"],
                }
                max_difference = max(
                    abs(a - b)
                    for a, b in zip(baseline["final_model"], recovery["final_model"], strict=True)
                )
                checks["final_model_exact"] = max_difference == 0
                cases.append(
                    {
                        "seed": seed,
                        "fault": fault,
                        "completed_microbatches_at_save": cut,
                        "epoch_at_save": failed["saved_epoch"],
                        "cursor_at_save": failed["saved_cursor"],
                        "pending_samples_at_save": failed["saved_pending_samples"],
                        "updates_at_save": failed["saved_updates"],
                        "failure_stage_observation": failed["stage_effects"],
                        "compared_safe_points": len(recovery["states"]),
                        "replayed_uncommitted_indices": failed["attempted_indices"],
                        "final_parameter_max_absolute_difference": max_difference,
                        "final_state_sha256": recovery["states"][-1],
                        "checks": checks,
                        "all_checks_passed": all(checks.values()),
                    }
                )
    checks = {
        "all_12_fault_recovery_cases_exact": len(cases) == 12
        and all(case["all_checks_passed"] for case in cases),
        "all_baseline_epochs_cover_every_sample_once": all(
            all(case["each_epoch_all_samples_once"]) for case in coverage
        ),
        "all_baselines_complete_15_microbatches_69_sample_visits": all(
            case["committed_microbatches"] == 15 and case["committed_sample_visits"] == 69
            for case in coverage
        ),
    }
    return {
        "schema_version": 1,
        "experiment": "joint_shuffled_accumulation_recovery",
        "executed_by": "AI assistant; learner mastery unassessed",
        "environment": {"python": platform.python_version(), "pytorch": str(torch.__version__)},
        "protocol": {
            "seeds": list(SEEDS),
            "shuffle_seed": 19,
            "epochs": EPOCHS,
            "samples": 23,
            "microbatch_size": 5,
            "accumulation_steps": 3,
            "drop_last": False,
            "dtype": "float64",
            "device": "cpu",
            "worker_processes": {"baseline": 3, "fault": 3, "recovery": 3},
            "fault_worker_expected_exit_code": FAULT_EXIT,
            "fault_mode": "injected Python exceptions followed by controlled nonzero worker exit",
            "safe_point": (
                "whole microbatch committed; epoch tail flushed and next permutation prepared"
            ),
            "comparison": "exact full typed-state digest and committed batch-index suffix",
            "recovery": "discard failed object; load the previous whole joint checkpoint",
            "excluded": [
                "exactly-once external side effects",
                "process-kill handling",
                "power-loss durability",
                "workers or prefetch",
                "GPU or AMP or DDP",
                "cross-runtime snapshot exchange",
            ],
        },
        "recovered_checkpoints": len(cases),
        "baseline_coverage": coverage,
        "cases": cases,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("baseline", "fault", "resume"), help=argparse.SUPPRESS)
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
        parser.exit(2, f"Joint recovery experiment failed: {type(error).__name__}.\n")
    if args.worker == "fault":
        raise SystemExit(FAULT_EXIT)
    if not args.worker and not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
