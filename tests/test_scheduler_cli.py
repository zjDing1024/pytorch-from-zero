"""Fresh-interpreter scheduler CLI and no-clobber publication checks."""

import json
import os
import subprocess
import sys

import pytest

from pytorch_lab.scheduled_checkpoint import (
    ScheduleConfig,
    ScheduledTrainer,
    load_scheduled_checkpoint,
    save_scheduled_checkpoint,
)


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "pytorch_lab.scheduler_cli", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_fresh_process_train_and_resume_match_custom_schedule(tmp_path):
    full, cut, resumed = (tmp_path / name for name in ("full.pt", "cut.pt", "resumed.pt"))
    output = tmp_path / "resumed.json"
    options = ("--seed", 7, "--batch-size", 19, "--step-size", 3, "--gamma", 0.7)
    first = run_cli("train", *options, "--epochs", 9, "--checkpoint", full)
    second = run_cli("train", *options, "--epochs", 3, "--checkpoint", cut)
    original = cut.read_bytes()
    third = run_cli(
        "resume",
        "--checkpoint",
        cut,
        "--epochs",
        9,
        "--save-checkpoint",
        resumed,
        "--output",
        output,
    )
    for result in (first, second, third):
        assert result.returncode == 0, result.stderr
    reports = [json.loads(result.stdout) for result in (first, second, third)]
    assert len({report["process_id"] for report in reports} | {os.getpid()}) == 4
    assert reports[0]["training"] == reports[2]["training"]
    assert json.loads(output.read_text()) == reports[2]
    assert cut.read_bytes() == original
    config = reports[2]["training"]["config"]
    assert config["step_size"] == 3 and config["gamma"] == 0.7 and config["batch_size"] == 19
    assert (
        load_scheduled_checkpoint(full).snapshot()["content_sha256"]
        == (load_scheduled_checkpoint(resumed).snapshot()["content_sha256"])
    )


def test_cli_epoch_zero_and_zero_additional_resume(tmp_path):
    initial, copied = tmp_path / "initial.pt", tmp_path / "copy.pt"
    first = run_cli("train", "--epochs", 0, "--checkpoint", initial)
    assert first.returncode == 0, first.stderr
    original = initial.read_bytes()
    second = run_cli("resume", "--checkpoint", initial, "--epochs", 0, "--save-checkpoint", copied)
    assert second.returncode == 0, second.stderr
    assert json.loads(first.stdout)["training"] == json.loads(second.stdout)["training"]
    assert load_scheduled_checkpoint(copied).epoch == 0
    assert initial.read_bytes() == original


@pytest.mark.parametrize(
    "option,value",
    [
        ("--epochs", -1),
        ("--epochs", 10001),
        ("--epochs", "1.5"),
        ("--step-size", 0),
        ("--step-size", -1),
        ("--step-size", "1.5"),
        ("--gamma", 0),
        ("--gamma", 1.1),
        ("--gamma", "nan"),
        ("--gamma", "inf"),
        ("--learning-rate", 0),
        ("--batch-size", 0),
        ("--momentum", 1),
        ("--seed", -1),
    ],
)
def test_invalid_train_args_create_no_artifacts(tmp_path, option, value):
    target, output = tmp_path / "bad.pt", tmp_path / "bad.json"
    result = run_cli("train", "--checkpoint", target, "--output", output, option, value)
    assert result.returncode == 2
    assert result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("arguments", [(), ("train",), ("resume",), ("unknown",)])
def test_missing_and_unknown_commands_are_parser_errors(arguments):
    result = run_cli(*arguments)
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_resume_requires_total_and_forbids_scheduler_override(tmp_path):
    path = tmp_path / "source.pt"
    save_scheduled_checkpoint(ScheduledTrainer(), path)
    for arguments in [(), ("--epochs", 2, "--gamma", 0.3)]:
        result = run_cli("resume", "--checkpoint", path, *arguments)
        assert result.returncode == 2


def test_resume_without_new_checkpoint_is_read_only(tmp_path):
    trainer = ScheduledTrainer(ScheduleConfig(step_size=2))
    trainer.train_to(2)
    path = tmp_path / "source.pt"
    save_scheduled_checkpoint(trainer, path)
    original = path.read_bytes()
    result = run_cli("resume", "--checkpoint", path, "--epochs", 4)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["training"]["epoch"] == 4
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("target_epoch", [-1, 2, 10001])
def test_resume_invalid_total_preserves_input_and_creates_nothing(tmp_path, target_epoch):
    trainer = ScheduledTrainer()
    trainer.train_to(3)
    source, target, output = (tmp_path / name for name in ("source.pt", "new.pt", "new.json"))
    save_scheduled_checkpoint(trainer, source)
    original = source.read_bytes()
    result = run_cli(
        "resume",
        "--checkpoint",
        source,
        "--epochs",
        target_epoch,
        "--save-checkpoint",
        target,
        "--output",
        output,
    )
    assert result.returncode == 2
    assert source.read_bytes() == original
    assert list(tmp_path.iterdir()) == [source]


def test_resume_corrupt_checkpoint_fails_cleanly(tmp_path):
    source, target = tmp_path / "bad.pt", tmp_path / "new.pt"
    source.write_bytes(b"PK truncated")
    result = run_cli("resume", "--checkpoint", source, "--epochs", 5, "--save-checkpoint", target)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert source.read_bytes() == b"PK truncated" and not target.exists()


@pytest.mark.parametrize("command", ["train", "resume", "verify"])
def test_existing_report_refused_before_work_or_checkpoint_publication(tmp_path, command):
    report, target = tmp_path / "keep.json", tmp_path / "new.pt"
    report.write_text("keep unchanged")
    if command == "train":
        args = ("--epochs", 1, "--checkpoint", target)
    elif command == "resume":
        args = ("--epochs", 1, "--checkpoint", tmp_path / "missing.pt", "--save-checkpoint", target)
    else:
        args = ("--epochs", 8, "--split-epoch", 5)
    result = run_cli(command, *args, "--output", report)
    assert result.returncode == 2
    assert "exists" in result.stderr
    assert report.read_text() == "keep unchanged"
    assert not target.exists()


def test_existing_checkpoint_is_never_overwritten_by_train_or_resume(tmp_path):
    path = tmp_path / "existing.pt"
    save_scheduled_checkpoint(ScheduledTrainer(), path)
    original = path.read_bytes()
    for arguments in [
        ("train", "--checkpoint", path),
        ("resume", "--checkpoint", path, "--save-checkpoint", path),
    ]:
        result = run_cli(*arguments, "--epochs", 1)
        assert result.returncode == 2
        assert path.read_bytes() == original


@pytest.mark.parametrize("command", ["train", "resume"])
def test_same_checkpoint_and_report_destination_is_rejected(tmp_path, command):
    target = tmp_path / "same"
    if command == "train":
        args = ("--checkpoint", target)
    else:
        source = tmp_path / "source.pt"
        save_scheduled_checkpoint(ScheduledTrainer(), source)
        args = ("--checkpoint", source, "--save-checkpoint", target)
    result = run_cli(command, *args, "--epochs", 1, "--output", target)
    assert result.returncode == 2
    assert not target.exists()


def test_dangling_output_symlink_is_not_replaced(tmp_path):
    target = tmp_path / "output.json"
    target.symlink_to(tmp_path / "absent.json")
    checkpoint = tmp_path / "new.pt"
    result = run_cli("train", "--epochs", 1, "--checkpoint", checkpoint, "--output", target)
    assert result.returncode == 2
    assert target.is_symlink() and not checkpoint.exists()


@pytest.mark.parametrize("cut,total", [(0, 5), (5, 5), (6, 5), (-1, 5), (1, 10001)])
def test_verify_invalid_split_or_total_returns_parser_error(tmp_path, cut, total):
    output = tmp_path / "bad.json"
    result = run_cli("verify", "--epochs", total, "--split-epoch", cut, "--output", output)
    assert result.returncode == 2
    assert not output.exists()
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("cut", [5, 7])
def test_verify_reports_real_fresh_process_recovery(tmp_path, cut):
    output = tmp_path / "evidence.json"
    result = run_cli(
        "verify", "--seed", 42, "--epochs", 10, "--split-epoch", cut, "--output", output
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(output.read_text())
    assert evidence == json.loads(result.stdout)
    assert evidence["all_checks_passed"] is True
    assert evidence["checks"] and all(evidence["checks"].values())
    assert evidence["full_vs_resumed_max_parameter_gap"] == 0
    assert len(set(evidence["process_ids"].values())) == 4
    assert evidence["scheduled_training"]["epoch"] == 10
    assert evidence["scheduled_training"]["config"]["gamma"] == 0.5
    assert evidence["fixed_training"]["config"]["gamma"] == 1
    assert evidence["learning_rate_sequence"] == [0.05] * 5 + [0.025] * 5
    assert evidence["scheduled_training"]["optimizer_updates"] == 50
    assert evidence["fixed_training"]["optimizer_updates"] == 50
    assert evidence["scheduled_training"]["samples_seen"] == 960
    assert evidence["fixed_training"]["samples_seen"] == 960
    assert evidence["lr_order"]["optimizer_then_scheduler"]["lr_used"] == [0.1, 0.05, 0.025]
    assert evidence["lr_order"]["scheduler_then_optimizer"]["lr_used"] == [0.05, 0.025, 0.0125]
    controls = evidence["negative_controls"]
    assert controls["start_epoch"] == cut
    assert controls["end_epoch"] > cut
    assert controls["scheduler_reset_phase_misaligned"] is (cut == 7)
    assert controls["omitted_momentum_max_parameter_gap"] > 1e-12
    assert controls["reset_shuffle_max_parameter_gap"] > 1e-12
    if cut == 5:
        assert controls["reset_scheduler_max_parameter_gap"] == 0
        assert controls["correct_lr_used"] == controls["reset_scheduler_lr_used"]
    else:
        assert controls["reset_scheduler_max_parameter_gap"] > 1e-12
        assert controls["correct_lr_used"] != controls["reset_scheduler_lr_used"]
        repeated = run_cli("verify", "--seed", 42, "--epochs", 10, "--split-epoch", cut)
        assert repeated.returncode == 0, repeated.stderr
        repeated_evidence = json.loads(repeated.stdout)
        assert repeated_evidence["all_checks_passed"] is True
        for field in (
            "checks",
            "scheduled_training",
            "fixed_training",
            "negative_controls",
            "learning_rate_sequence",
            "resumed_content_sha256",
            "lr_order",
            "momentum_reference",
        ):
            assert repeated_evidence[field] == evidence[field]


def test_late_recovery_cut_uses_separate_observable_early_controls():
    result = run_cli("verify", "--epochs", 240, "--split-epoch", 237)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["all_checks_passed"]
    assert report["full_vs_resumed_max_parameter_gap"] == 0
    assert report["split_epoch"] == 237
    assert report["negative_controls"]["requested_recovery_split_epoch"] == 237
    assert report["negative_controls"]["start_epoch"] == 7
    assert report["negative_controls"]["end_epoch"] == 13
    assert report["negative_controls"]["reset_scheduler_max_parameter_gap"] > 1e-12


def test_diagnostic_window_stays_bounded_at_maximum_recovery_cut():
    from pytorch_lab.scheduler_experiment import _control_window

    assert _control_window(9999, 5) == (7, 13)
    assert _control_window(5, 5) == (5, 11)
