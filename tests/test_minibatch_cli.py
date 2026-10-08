"""Subprocess checks for the additive minibatch experiment entry point."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args):
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    return subprocess.run(
        [sys.executable, "-m", "pytorch_lab.minibatch_cli", *args],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
        check=False,
    )


def test_cli_creates_machine_readable_minibatch_evidence(tmp_path):
    destination = tmp_path / "nested" / "minibatch.json"
    result = run_cli("--output", str(destination))
    assert result.returncode == 0, result.stderr
    saved = json.loads(destination.read_text())
    assert saved == json.loads(result.stdout)
    assert saved["experiment"] == "module_dataset_minibatch"
    assert saved["schema_version"] == 1
    assert saved["all_checks_passed"] is True
    assert all(saved["checks"].values())
    assert saved["environment"]["pytorch"]
    assert saved["environment"]["device"] == "cpu"
    assert saved["training"]["optimizer_steps"] == 200
    assert saved["training"]["training_sample_visits"] == 3840
    assert saved["training"]["last_batch_size"] == 16
    assert len(saved["training"]["history"]) == 41
    assert saved["comparison"]["full_batch_optimizer_steps"] == 40
    assert saved["comparison"]["module_full_batch_vs_handwritten_max_parameter_gap"] < 1e-12


def test_cli_accepts_all_training_options():
    result = run_cli(
        "--seed", "7", "--epochs", "10", "--batch-size", "13", "--learning-rate", "0.04"
    )
    assert result.returncode == 0, result.stderr
    saved = json.loads(result.stdout)
    assert saved["config"] == {
        "seed": 7,
        "epochs": 10,
        "batch_size": 13,
        "learning_rate": 0.04,
        "noise_std": 0.05,
    }
    assert saved["training"]["optimizer_steps"] == 80
    assert saved["training"]["training_sample_visits"] == 960
    assert saved["training"]["last_batch_size"] == 5


def test_cli_refuses_to_overwrite_existing_evidence(tmp_path):
    destination = tmp_path / "existing.json"
    destination.write_text("KEEP")
    result = run_cli("--output", str(destination))
    assert result.returncode == 2
    assert destination.read_text() == "KEEP"
    assert "already exists" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ("--epochs", "0"),
        ("--batch-size", "0"),
        ("--seed", "-1"),
        ("--learning-rate", "nan"),
        ("--learning-rate", "-0.1"),
    ],
)
def test_cli_rejects_invalid_configuration_without_writing_evidence(tmp_path, args):
    destination = tmp_path / "invalid.json"
    result = run_cli(*args, "--output", str(destination))
    assert result.returncode == 2
    assert "Experiment failed:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not destination.exists()
    assert not result.stdout


def test_cli_reports_overflow_without_a_traceback_or_output_file(tmp_path):
    destination = tmp_path / "overflow.json"
    result = run_cli("--epochs", "1", "--learning-rate", "1e200", "--output", str(destination))
    assert result.returncode == 2
    assert "Experiment failed:" in result.stderr
    assert "non-finite" in result.stderr
    assert "Traceback" not in result.stderr
    assert not destination.exists()
    assert not result.stdout


def test_cli_preserves_failed_check_evidence_and_exits_one(tmp_path):
    destination = tmp_path / "failed-checks.json"
    result = run_cli("--epochs", "1", "--learning-rate", "100", "--output", str(destination))
    assert result.returncode == 1, result.stderr
    saved = json.loads(destination.read_text())
    assert saved == json.loads(result.stdout)
    assert saved["all_checks_passed"] is False
    assert not all(saved["checks"].values())
    assert "checks failed" in result.stderr
    assert "Traceback" not in result.stderr
