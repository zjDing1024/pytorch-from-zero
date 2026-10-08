"""Measured momentum, StepLR order, equal-budget and fresh-process recovery evidence."""

import copy
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import warnings
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from pytorch_lab.checkpoint import CheckpointError, _epoch
from pytorch_lab.checkpoint_experiment import _parameter_gap, _same
from pytorch_lab.scheduled_checkpoint import (
    ScheduleConfig,
    ScheduledTrainer,
    learning_rates,
    load_scheduled_checkpoint,
    save_scheduled_checkpoint,
)


def scheduler_order_demo() -> dict:
    """An actual torch reference: constant gradient isolates LR call-order effects."""
    runs = {}
    for order in ("optimizer_then_scheduler", "scheduler_then_optimizer"):
        parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        used = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                parameter.grad = torch.ones_like(parameter)
                if order == "scheduler_then_optimizer":
                    scheduler.step()
                used.append(optimizer.param_groups[0]["lr"])
                optimizer.step()
                if order == "optimizer_then_scheduler":
                    scheduler.step()
        runs[order] = {
            "lr_used": used,
            "final_parameter": parameter.item(),
            "order_warning_observed": any("before" in str(w.message) for w in caught),
        }
    runs["checks"] = {
        "correct_rates": runs["optimizer_then_scheduler"]["lr_used"] == [0.1, 0.05, 0.025],
        "early_step_skips_initial_rate": runs["scheduler_then_optimizer"]["lr_used"]
        == [0.05, 0.025, 0.0125],
        "wrong_order_warned": runs["scheduler_then_optimizer"]["order_warning_observed"],
        "correct_order_not_warned": not runs["optimizer_then_scheduler"]["order_warning_observed"],
        "parameters_differ": runs["optimizer_then_scheduler"]["final_parameter"]
        != runs["scheduler_then_optimizer"]["final_parameter"],
    }
    return runs


def _process(*arguments: str) -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "pytorch_lab.scheduler_cli", *arguments],
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode:
        raise RuntimeError(
            f"scheduler subprocess failed ({completed.returncode}): {completed.stderr}"
        )
    return json.loads(completed.stdout)


def _control_window(split_epoch: int, step_size: int) -> tuple[int, int]:
    """Early diagnostic state keeps omissions observable after late-run LR underflow."""
    start = min(split_epoch, 7)
    return start, start + step_size + 1


def run_scheduler_experiment(seed: int = 42, epochs: int = 40, split_epoch: int = 7) -> dict:
    from pytorch_lab.momentum import run_momentum_reference_trace

    config = ScheduleConfig(seed=seed)
    config.validate()
    _epoch(epochs)
    _epoch(split_epoch)
    if not 0 < split_epoch < epochs:
        raise ValueError("split_epoch must be strictly between zero and total epochs")
    # Diagnostics are deliberately early and separate from the requested recovery cut.
    control_start, control_epoch = _control_window(split_epoch, config.step_size)
    caller_rng = torch.get_rng_state().clone()
    momentum = run_momentum_reference_trace()
    order = scheduler_order_demo()
    with tempfile.TemporaryDirectory(prefix="pytorch-scheduler-") as directory:
        folder = Path(directory)
        full_path, cut_path, resumed_path, fixed_path = (
            folder / name for name in ("full.pt", "cut.pt", "resumed.pt", "fixed.pt")
        )
        full_run = _process(
            "train", "--seed", str(seed), "--epochs", str(epochs), "--checkpoint", str(full_path)
        )
        cut_run = _process(
            "train",
            "--seed",
            str(seed),
            "--epochs",
            str(split_epoch),
            "--checkpoint",
            str(cut_path),
        )
        resumed_run = _process(
            "resume",
            "--checkpoint",
            str(cut_path),
            "--epochs",
            str(epochs),
            "--save-checkpoint",
            str(resumed_path),
        )
        fixed_run = _process(
            "train",
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--gamma",
            "1",
            "--checkpoint",
            str(fixed_path),
        )
        full = load_scheduled_checkpoint(full_path, config)
        resumed = load_scheduled_checkpoint(resumed_path, config)
        fixed = load_scheduled_checkpoint(fixed_path, replace(config, gamma=1.0))
        cut = load_scheduled_checkpoint(cut_path, config)
        left, right = full.snapshot(), resumed.snapshot()
        diagnostic_path = cut_path
        if control_start != split_epoch:
            diagnostic_path = folder / "early-diagnostic.pt"
            diagnostic = ScheduledTrainer(config)
            diagnostic.train_to(control_start)
            save_scheduled_checkpoint(diagnostic, diagnostic_path)
        correct_next = load_scheduled_checkpoint(diagnostic_path)
        correct_next.train_to(control_epoch)
        missing_scheduler = load_scheduled_checkpoint(diagnostic_path)
        # Keep restored optimizer LR/momentum, but reset scheduler phase to epoch 0.
        missing_scheduler._scheduler = torch.optim.lr_scheduler.StepLR(
            missing_scheduler._optimizer, step_size=config.step_size, gamma=config.gamma
        )
        missing_scheduler.train_to(control_epoch)
        missing_momentum = load_scheduled_checkpoint(diagnostic_path)
        missing_momentum._optimizer.state.clear()
        missing_momentum.train_to(control_epoch)
        missing_shuffle = load_scheduled_checkpoint(diagnostic_path)
        missing_shuffle._train_loader.generator.manual_seed(config.seed)
        missing_shuffle.train_to(control_epoch)
        scheduler_gap = _parameter_gap(correct_next, missing_scheduler)
        momentum_gap = _parameter_gap(correct_next, missing_momentum)
        shuffle_gap = _parameter_gap(correct_next, missing_shuffle)
        damaged_path = folder / "truncated.pt"
        original_bytes = cut_path.read_bytes()
        damaged_path.write_bytes(original_bytes[:128])
        try:
            load_scheduled_checkpoint(damaged_path)
        except CheckpointError:
            corruption_rejected = True
        else:
            corruption_rejected = False
        try:
            load_scheduled_checkpoint(cut_path, replace(config, gamma=0.8))
        except CheckpointError:
            mismatch_rejected = True
        else:
            mismatch_rejected = False
        try:
            save_scheduled_checkpoint(full, cut_path)
        except FileExistsError:
            overwrite_rejected = cut_path.read_bytes() == original_bytes
        else:
            overwrite_rejected = False
        correct_rates = [row["lr_used"] for row in correct_next.report()["history"][1:]]
        omitted_rates = [row["lr_used"] for row in missing_scheduler.report()["history"][1:]]
        phase_misaligned = control_start % config.step_size != 0
        checks = {
            "manual_momentum_matches_torch": momentum["all_checks_passed"],
            "scheduler_order_demo_passed": all(order["checks"].values()),
            "fresh_processes_distinct": len(
                {run["process_id"] for run in (full_run, cut_run, resumed_run, fixed_run)}
            )
            == 4,
            "model_bitwise_equal": _same(left["model"], right["model"]),
            "optimizer_bitwise_equal": _same(left["optimizer"], right["optimizer"]),
            "scheduler_state_equal": _same(left["scheduler"], right["scheduler"]),
            "generator_states_equal": _same(left["generators"], right["generators"]),
            "entire_history_equal": left["history"] == right["history"],
            "checkpoint_content_equal": left["content_sha256"] == right["content_sha256"],
            "reports_equal": full_run["training"] == resumed_run["training"],
            "epoch_lr_matches_formula": [row["lr_used"] for row in left["history"][1:]]
            == learning_rates(config, epochs)[:-1],
            "fixed_and_scheduled_same_update_budget": full.report()["optimizer_updates"]
            == fixed.report()["optimizer_updates"],
            "fixed_and_scheduled_same_sample_budget": full.report()["samples_seen"]
            == fixed.report()["samples_seen"],
            "fixed_and_scheduled_same_shuffle_state": _same(
                left["generators"], fixed.snapshot()["generators"]
            ),
            "fixed_and_scheduled_same_data": left["dataset_sha256"]
            == fixed.snapshot()["dataset_sha256"],
            "fixed_rate_unchanged": all(
                row["lr_used"] == config.learning_rate for row in fixed.report()["history"][1:]
            ),
            "scheduler_reset_lr_effect_matches_phase": (correct_rates != omitted_rates)
            == phase_misaligned,
            "scheduler_reset_parameter_effect_matches_phase": (scheduler_gap > 1e-12)
            == phase_misaligned,
            "dropping_momentum_changes_control": momentum_gap > 1e-12,
            "resetting_shuffle_changes_control": shuffle_gap > 1e-12,
            "truncated_file_rejected": corruption_rejected,
            "config_mismatch_rejected": mismatch_rejected,
            "overwrite_rejected_bytes_unchanged": overwrite_rejected,
            "caller_global_rng_unchanged": torch.equal(caller_rng, torch.get_rng_state()),
        }
        result = {
            "schema_version": 1,
            "experiment": "momentum_step_lr_equal_budget_and_recovery",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "executed_by": "AI assistant; learner independent implementation remains pending",
            "environment": {
                "python": platform.python_version(),
                "pytorch": str(torch.__version__),
                "pytorch_git_version": torch.version.git_version,
                "platform": platform.platform(),
                "device": "cpu",
                "dtype": "float64",
                "num_workers": 0,
                "torch_num_threads": torch.get_num_threads(),
            },
            "config": asdict(config),
            "total_epochs": epochs,
            "split_epoch": split_epoch,
            "process_ids": {
                "full": full_run["process_id"],
                "interrupted": cut_run["process_id"],
                "resumed": resumed_run["process_id"],
                "fixed": fixed_run["process_id"],
            },
            "protocol": (
                "Four fresh interpreters: continuous StepLR, interrupted StepLR, resume, fixed LR. "
                "Same seed, split, initialization, batches, epochs, update and sample budgets; "
                "only gamma differs (0.5 vs 1). No hyperparameter tuning on holdouts. "
                "Negative controls use early diagnostic cut min(split_epoch,7), then step_size+1 "
                "epochs, separate from the requested recovery cut/budget; this avoids assuming "
                "parameter effects remain observable after very late LR underflow. "
                "Bitwise recovery is limited to this CPU/software environment."
            ),
            "checkpoint": {
                "bytes": len(original_bytes),
                "file_sha256": hashlib.sha256(original_bytes).hexdigest(),
                "content_sha256": cut.snapshot()["content_sha256"],
            },
            "momentum_reference": momentum,
            "lr_order": order,
            "learning_rate_sequence": [row["lr_used"] for row in left["history"][1:]],
            "scheduled_training": full_run["training"],
            "fixed_training": fixed_run["training"],
            "full_vs_resumed_max_parameter_gap": _parameter_gap(full, resumed),
            "resumed_content_sha256": right["content_sha256"],
            "negative_controls": {
                "start_epoch": control_start,
                "requested_recovery_split_epoch": split_epoch,
                "end_epoch": control_epoch,
                "scheduler_reset_phase_misaligned": phase_misaligned,
                "reset_scheduler_max_parameter_gap": scheduler_gap,
                "omitted_momentum_max_parameter_gap": momentum_gap,
                "reset_shuffle_max_parameter_gap": shuffle_gap,
                "correct_lr_used": correct_rates[control_start:],
                "reset_scheduler_lr_used": omitted_rates[control_start:],
                "caveat": (
                    "StepLR reset can preserve the LR trajectory at a step_size-aligned cut "
                    "if current optimizer LR is retained; counters still differ."
                ),
            },
            "checks": checks,
            "all_checks_passed": all(checks.values()),
        }
    return copy.deepcopy(result)
