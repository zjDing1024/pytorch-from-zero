"""Verify real child-process fault/recovery evidence and fail-closed CLI output."""

import json
import subprocess
import sys

import pytest

from pytorch_lab import joint_recovery_cli as cli
from pytorch_lab.joint_recovery import JointConfig


def test_predeclared_fault_points_cover_pending_momentum_tail_and_later_epoch():
    assert cli.FAULT_CASES == (
        ("after_fetch", 1),
        ("after_backward", 3),
        ("after_update", 4),
        ("before_commit", 7),
    )
    assert cli.SEEDS == (42, 7, 123)
    assert cli.EPOCHS == 3


def test_fault_worker_refuses_unsafe_operations_and_replays_saved_whole_state(tmp_path):
    config = JointConfig()
    baseline = cli._baseline(config, tmp_path)
    faults = cli._faults(config, tmp_path)["cases"]
    recoveries = cli._resumed(config, tmp_path)["cases"]
    assert len(faults) == len(recoveries) == 4
    for failed, recovered in zip(faults, recoveries, strict=True):
        cut = failed["cut"]
        assert failed["injected_failure_observed"]
        assert failed["unsafe_operations_refused"] == dict.fromkeys(
            ("snapshot", "save", "step", "train"), True
        )
        assert failed["saved_checkpoint_unchanged"] and failed["unsafe_file_absent"]
        assert recovered["states"] == baseline["states"][cut:]
        assert recovered["batches"] == baseline["batches"][cut:]
        assert recovered["batches"][0] == failed["attempted_indices"]
        assert recovered["final_model"] == baseline["final_model"]
    assert [case["saved_pending_samples"] for case in faults] == [5, 0, 5, 10]
    assert [case["saved_updates"] for case in faults] == [0, 1, 1, 2]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "cut-1.pt",
        "cut-3.pt",
        "cut-4.pt",
        "cut-7.pt",
    ]


@pytest.mark.parametrize("mode,exit_code", [("baseline", 0), ("fault", 23), ("resume", 0)])
def test_invoke_requires_declared_worker_exit_code(monkeypatch, tmp_path, mode, exit_code):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, exit_code, '{"ok":true}', "")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli._invoke(mode, 42, tmp_path) == {"ok": True}
    assert calls[0][0][:4] == [sys.executable, "-I", "-m", "pytorch_lab.joint_recovery_cli"]
    assert calls[0][1]["timeout"] == 60
    assert calls[0][1]["check"] is False


@pytest.mark.parametrize("mode,wrong_code", [("baseline", 23), ("fault", 0), ("resume", 1)])
def test_invoke_rejects_unexpected_exit_even_with_successful_json(
    monkeypatch, tmp_path, mode, wrong_code
):
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, wrong_code, '{"all_checks_passed":true}', ""
        ),
    )
    with pytest.raises(RuntimeError, match="worker exited"):
        cli._invoke(mode, 42, tmp_path)


def test_real_fresh_process_cli_stdout_protocol_and_no_overwrite(tmp_path):
    path = tmp_path / "evidence.json"
    command = [sys.executable, "-I", "-m", "pytorch_lab.joint_recovery_cli", "--output", str(path)]
    # This aggregate launches nine serial workers; each worker retains its 60s limit.
    # The outer integration deadline is not a numerical or production-latency contract.
    run = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    report = json.loads(path.read_text())
    assert report == json.loads(run.stdout)
    assert report["all_checks_passed"] and report["recovered_checkpoints"] == 12
    assert len(report["cases"]) == 12
    assert report["protocol"]["worker_processes"] == {"baseline": 3, "fault": 3, "recovery": 3}
    assert report["protocol"]["fault_worker_expected_exit_code"] == 23
    for case in report["cases"]:
        assert case["all_checks_passed"] and all(case["checks"].values())
        assert case["final_parameter_max_absolute_difference"] == 0
    original = path.read_bytes()
    retry = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    assert retry.returncode == 2 and path.read_bytes() == original


@pytest.mark.parametrize("arguments", [["--worker", "baseline"], ["--worker", "invalid"]])
def test_worker_argument_validation(arguments):
    run = subprocess.run(
        [sys.executable, "-I", "-m", "pytorch_lab.joint_recovery_cli", *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 2
    assert not run.stdout


def test_output_symlink_rejected_before_experiment(tmp_path):
    target = tmp_path / "absent.json"
    link = tmp_path / "output.json"
    link.symlink_to(target)
    run = subprocess.run(
        [sys.executable, "-I", "-m", "pytorch_lab.joint_recovery_cli", "--output", str(link)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 2 and link.is_symlink() and not target.exists()
