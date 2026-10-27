"""Fresh-process recovery experiment for own trusted microbatch checkpoints."""

import argparse
import copy
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from pytorch_lab.accumulation_recovery import (
    WindowConfig,
    WindowTrainer,
    load_window_checkpoint,
    restore_snapshot,
    save_window_checkpoint,
)
from pytorch_lab.checkpoint import CheckpointError, _content_digest

# Declared before execution: bit-exact *state* at each same-environment safe point,
# not byte-identical torch.save archives, cross-hardware/version or arbitrary loaders.
CUTS = (0, 1, 3, 4, 5, 6, 15)
EPOCHS = 3


def _trace(trainer):
    result = [trainer.snapshot()["content_sha256"]]
    while trainer.epoch < EPOCHS:
        trainer.step_microbatch()
        result.append(trainer.snapshot()["content_sha256"])
    return result


def _worker(seed: int, directory: Path, resume: int | None):
    torch.set_num_threads(1)
    config = WindowConfig(seed=seed)
    if resume is not None:
        return {"trace": _trace(load_window_checkpoint(directory / f"cut-{resume}.pt", config))}
    trainer = WindowTrainer(config)
    trace = []
    for point in range(16):
        trace.append(trainer.snapshot()["content_sha256"])
        if point in CUTS:
            save_window_checkpoint(trainer, directory / f"cut-{point}.pt")
        if point < 15:
            trainer.step_microbatch()
    return {"trace": trace}


def _invoke(seed, directory, resume=None):
    command = [
        sys.executable,
        "-I",
        "-m",
        "pytorch_lab.accumulation_recovery_cli",
        "--worker",
        "--seed",
        str(seed),
        "--directory",
        str(directory),
    ]
    if resume is not None:
        command += ["--resume", str(resume)]
    run = subprocess.run(command, text=True, capture_output=True, check=True, timeout=60)
    return json.loads(run.stdout)


def _resign(payload):
    payload["content_sha256"] = _content_digest(
        {k: v for k, v in payload.items() if k != "content_sha256"}
    )
    return payload


def _negative_controls():
    trainer = WindowTrainer()
    for _ in range(4):
        trainer.step_microbatch()
    original = trainer.snapshot()  # Cursor 20: momentum exists and 5 samples pending.
    rejected = {}
    for fault in ("missing_pending_gradients", "wrong_element_count", "wrong_cursor"):
        payload = copy.deepcopy(original)
        if fault == "missing_pending_gradients":
            del payload["pending_gradients"]
        elif fault == "wrong_element_count":
            payload["pending_elements"] += 1
        else:
            payload["cursor"] += 1
        try:
            restore_snapshot(_resign(payload))
        except CheckpointError:
            rejected[fault] = True
        else:
            rejected[fault] = False
    expected = restore_snapshot(original)
    expected.step_microbatch()
    good = expected.snapshot()
    drifts = {}
    for fault in ("zero_pending_gradients", "zero_momentum"):
        payload = copy.deepcopy(original)
        if fault == "zero_pending_gradients":
            for value in payload["pending_gradients"].values():
                value.zero_()
        else:
            for value in payload["optimizer"]["state"].values():
                value["momentum_buffer"].zero_()
        changed = restore_snapshot(_resign(payload))
        changed.step_microbatch()
        bad = changed.snapshot()
        gap = max((good["model"][k] - bad["model"][k]).abs().max().item() for k in good["model"])
        drifts[fault] = {"next_parameter_max_absolute_difference": gap, "detected": gap > 0}
    return {
        "invalid_state_rejected": rejected,
        "valid_shape_fault_injection": drifts,
        "boundary": "checksum is not authentication; valid-shape forged values may load",
    }


def run_experiment():
    torch.set_num_threads(1)
    cases = []
    with tempfile.TemporaryDirectory(prefix="accumulation-recovery-") as temp:
        for seed in (42, 7, 123):
            directory = Path(temp) / str(seed)
            directory.mkdir()
            baseline = _invoke(seed, directory)["trace"]
            for cut in CUTS:
                resumed = _invoke(seed, directory, cut)["trace"]
                saved = load_window_checkpoint(directory / f"cut-{cut}.pt").snapshot()
                cases.append(
                    {
                        "seed": seed,
                        "completed_microbatches_at_save": cut,
                        "epoch_at_save": saved["epoch"],
                        "cursor_at_save": saved["cursor"],
                        "pending_samples_at_save": saved["pending_samples"],
                        "pending_target_elements_at_save": saved["pending_elements"],
                        "compared_safe_points": len(resumed),
                        "every_remaining_state_exact": resumed == baseline[cut:],
                        "final_state_sha256": resumed[-1],
                    }
                )
    controls = _negative_controls()
    checks = {
        "all_21_fresh_process_resume_cases_exact": len(cases) == 21
        and all(c["every_remaining_state_exact"] for c in cases),
        "malformed_state_controls_rejected": all(controls["invalid_state_rejected"].values()),
        "omitted_gradient_and_momentum_effect_detected": all(
            c["detected"] for c in controls["valid_shape_fault_injection"].values()
        ),
    }
    return {
        "schema_version": 1,
        "experiment": "accumulation_window_recovery",
        "executed_by": "AI assistant; learner mastery unassessed",
        "environment": {"python": platform.python_version(), "pytorch": str(torch.__version__)},
        "protocol": {
            "seeds": [42, 7, 123],
            "epochs": EPOCHS,
            "samples": 23,
            "microbatch_size": 5,
            "accumulation_steps": 3,
            "input_output_width": [3, 2],
            "order": "fixed-sequential",
            "safe_point": "after a fully completed microbatch operation",
            "comparison": "exact typed state digest at every remaining safe point",
            "training_rng": "none; isolated seed regenerates fingerprinted fixture",
            "excluded": [
                "shuffled loaders",
                "workers",
                "AMP",
                "DDP",
                "GPU",
                "mid-operation crash recovery",
                "power-loss durability",
            ],
        },
        "cases": cases,
        "negative_controls": controls,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42, help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--resume", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.directory is None:
            parser.error("worker directory required")
        result = _worker(args.seed, args.directory, args.resume)
    else:
        result = run_experiment()
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    print(encoded)
    if not args.worker and not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
