"""Real subprocess recovery and fail-closed output for the stochastic joint CLI."""

import json
import subprocess
import sys

import pytest

from pytorch_lab import stochastic_joint_cli as cli
from pytorch_lab.stochastic_joint import StochasticConfig


def test_predeclared_protocol_and_fault_boundaries():
    assert cli.SEEDS == (7, 42, 123)
    assert cli.CUTS == (1, 3, 5)
    assert cli.EPOCHS == 3
    assert cli.FAULT_CASES == (
        ("after_fetch", 1),
        ("after_forward", 1),
        ("after_backward", 3),
        ("after_update", 4),
        ("before_commit", 7),
    )


def test_same_process_worker_contract_before_real_process_test(tmp_path):
    config = StochasticConfig()
    baseline = cli._baseline(config, tmp_path)
    result = cli._resume(config, tmp_path)
    assert len(baseline["batches"]) == len(baseline["traces"]) == 15
    assert len(baseline["states"]) == 16
    assert len(result["normal"]) == 3
    assert len(result["failures"]) == 5
    for case in result["normal"] + result["failures"]:
        assert all(cli._compare(baseline, case).values())
    for case in result["failures"]:
        assert case["injected_failure_observed"]
        assert case["caller_rng_restored"]
        assert all(case["unsafe_operations_refused"].values())
        assert case["saved_checkpoint_unchanged"] and case["unsafe_file_absent"]


@pytest.mark.parametrize("field", ["states", "batches", "traces"])
def test_comparison_detects_full_suffix_tampering(field):
    baseline = {"states": ["a", "b"], "batches": [[1]], "traces": [{"loss": 2}]}
    resumed = {"cut": 0, "states": ["a", "b"], "batches": [[1]], "traces": [{"loss": 2}]}
    resumed[field] = []
    assert not all(cli._compare(baseline, resumed).values())


@pytest.mark.parametrize("mode", ["baseline", "resume"])
def test_invoke_uses_isolated_interpreter_and_rejects_wrong_exit(monkeypatch, tmp_path, mode):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 1, '{"all_checks_passed":true}', "")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="worker exited"):
        cli._invoke(mode, 42, tmp_path)
    assert calls[0][0][:4] == [sys.executable, "-I", "-m", "pytorch_lab.stochastic_joint_cli"]
    assert calls[0][1]["timeout"] == 60


def test_real_fresh_process_cli_and_no_overwrite(tmp_path):
    path = tmp_path / "result.json"
    command = [
        sys.executable,
        "-I",
        "-m",
        "pytorch_lab.stochastic_joint_cli",
        "--output",
        str(path),
    ]
    # Six serial child workers, each bounded at 60s. This is a supervisor budget,
    # not a performance result or numerical threshold.
    run = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    evidence = json.loads(path.read_text())
    assert evidence == json.loads(run.stdout)
    assert evidence["all_checks_passed"]
    assert len(evidence["recovery_cases"]) == 9
    assert len(evidence["failure_recovery_cases"]) == 15
    assert evidence["protocol"]["worker_processes"] == {"baseline": 3, "resume_and_fault": 3}
    assert sum(case["compared_steps"] for case in evidence["recovery_cases"]) == 108
    for case in evidence["recovery_cases"] + evidence["failure_recovery_cases"]:
        assert case["all_checks_passed"] and all(case["checks"].values())
    original = path.read_bytes()
    retry = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    assert retry.returncode == 2 and path.read_bytes() == original


@pytest.mark.parametrize("arguments", [["--worker", "baseline"], ["--worker", "invalid"]])
def test_worker_arguments_rejected(arguments):
    run = subprocess.run(
        [sys.executable, "-I", "-m", "pytorch_lab.stochastic_joint_cli", *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 2 and not run.stdout


def test_dangling_symlink_output_rejected(tmp_path):
    target = tmp_path / "absent.json"
    link = tmp_path / "result.json"
    link.symlink_to(target)
    run = subprocess.run(
        [sys.executable, "-I", "-m", "pytorch_lab.stochastic_joint_cli", "--output", str(link)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 2 and link.is_symlink() and not target.exists()
