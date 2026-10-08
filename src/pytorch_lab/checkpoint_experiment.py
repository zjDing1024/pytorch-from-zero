"""Fresh-process recovery evidence plus deliberately incomplete recovery controls."""

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from pytorch_lab.checkpoint import (
    CheckpointError,
    RecoveryConfig,
    _epoch,
    load_checkpoint,
    save_checkpoint,
)


def _same(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _parameter_gap(left, right) -> float:
    return max(
        (a.detach() - b.detach()).abs().max().item()
        for a, b in zip(left._model.parameters(), right._model.parameters(), strict=True)
    )


def _process(*arguments: str) -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "pytorch_lab.checkpoint_cli", *arguments],
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode:
        raise RuntimeError(
            f"checkpoint subprocess failed ({completed.returncode}): {completed.stderr}"
        )
    return json.loads(completed.stdout)


def run_checkpoint_experiment(seed: int = 42, epochs: int = 40, split_epoch: int = 7) -> dict:
    config = RecoveryConfig(seed=seed)
    config.validate()
    _epoch(epochs)
    _epoch(split_epoch)
    if not 0 < split_epoch < epochs:
        raise ValueError("split_epoch must be strictly between zero and total epochs")
    caller_rng = torch.get_rng_state().clone()
    with tempfile.TemporaryDirectory(prefix="pytorch-checkpoint-") as directory:
        folder = Path(directory)
        full_path, cut_path, resumed_path = (
            folder / name for name in ("full.pt", "cut.pt", "resumed.pt")
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
        full = load_checkpoint(full_path, config)
        resumed = load_checkpoint(resumed_path, config)
        left, right = full.snapshot(), resumed.snapshot()
        cut = load_checkpoint(cut_path, config)
        correct_next = load_checkpoint(cut_path, config)
        correct_next.train_to(split_epoch + 1)
        missing_momentum = load_checkpoint(cut_path, config)
        missing_momentum._optimizer = torch.optim.SGD(
            missing_momentum._model.parameters(), lr=config.learning_rate, momentum=config.momentum
        )
        missing_momentum.train_to(split_epoch + 1)
        missing_shuffle = load_checkpoint(cut_path, config)
        missing_shuffle._train_loader.generator.manual_seed(config.seed)
        missing_shuffle.train_to(split_epoch + 1)
        momentum_gap = _parameter_gap(correct_next, missing_momentum)
        shuffle_gap = _parameter_gap(correct_next, missing_shuffle)
        corrupted_path = folder / "truncated.pt"
        corrupted_path.write_bytes(cut_path.read_bytes()[:128])
        try:
            load_checkpoint(corrupted_path)
        except CheckpointError:
            corruption_rejected = True
        else:
            corruption_rejected = False
        try:
            load_checkpoint(cut_path, replace(config, momentum=0.7))
        except CheckpointError:
            mismatch_rejected = True
        else:
            mismatch_rejected = False
        original_bytes = cut_path.read_bytes()
        try:
            save_checkpoint(full, cut_path)
        except FileExistsError:
            overwrite_rejected = cut_path.read_bytes() == original_bytes
        else:
            overwrite_rejected = False
        checks = {
            "fresh_processes_distinct": len(
                {full_run["process_id"], cut_run["process_id"], resumed_run["process_id"]}
            )
            == 3,
            "model_bitwise_equal": _same(left["model"], right["model"]),
            "optimizer_bitwise_equal": _same(left["optimizer"], right["optimizer"]),
            "generator_states_equal": _same(left["generators"], right["generators"]),
            "entire_history_equal": left["history"] == right["history"],
            "checkpoint_content_equal": left["content_sha256"] == right["content_sha256"],
            "reports_equal": full_run["training"] == resumed_run["training"],
            "momentum_state_nonzero": any(
                torch.count_nonzero(v["momentum_buffer"]).item()
                for v in cut.snapshot()["optimizer"]["state"].values()
            ),
            "dropping_momentum_changes_next_epoch": momentum_gap > 1e-12,
            "resetting_shuffle_changes_next_epoch": shuffle_gap > 1e-12,
            "truncated_file_rejected": corruption_rejected,
            "config_mismatch_rejected": mismatch_rejected,
            "overwrite_rejected_bytes_unchanged": overwrite_rejected,
            "caller_global_rng_unchanged": torch.equal(caller_rng, torch.get_rng_state()),
            "holdout_beats_train_mean": full.report()["test_mse"]
            < full.report()["test_mean_baseline_mse"],
        }
        result = {
            "schema_version": 1,
            "experiment": "epoch_checkpoint_fresh_process_recovery",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "executed_by": "AI assistant; learner self-check remains pending",
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
            },
            "protocol": (
                "Three fresh interpreter processes: full, stop after a completed epoch, resume. "
                "Same CPU/software environment; no cross-platform bitwise guarantee. "
                "Negative controls compare one epoch after the cut."
            ),
            "checkpoint": {
                "bytes": len(original_bytes),
                "file_sha256": hashlib.sha256(original_bytes).hexdigest(),
                "content_sha256": cut.snapshot()["content_sha256"],
            },
            "training": full_run["training"],
            "resumed_content_sha256": right["content_sha256"],
            "full_vs_resumed_max_parameter_gap": _parameter_gap(full, resumed),
            "negative_controls": {
                "next_epoch": split_epoch + 1,
                "omitted_momentum_max_parameter_gap": momentum_gap,
                "reset_shuffle_max_parameter_gap": shuffle_gap,
            },
            "checks": checks,
            "all_checks_passed": all(checks.values()),
        }
    return result
