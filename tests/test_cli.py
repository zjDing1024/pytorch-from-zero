import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args):
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    return subprocess.run(
        [sys.executable, "-m", "pytorch_lab", *args],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
        check=False,
    )


def test_cli_creates_valid_machine_readable_evidence(tmp_path):
    destination = tmp_path / "nested" / "report.json"
    result = run_cli("--output", str(destination))
    assert result.returncode == 0, result.stderr
    saved = json.loads(destination.read_text())
    assert saved == json.loads(result.stdout)
    assert saved["all_checks_passed"] is True
    assert all(saved["checks"].values())
    assert saved["environment"]["pytorch"]


def test_cli_refuses_to_overwrite_evidence(tmp_path):
    destination = tmp_path / "existing.json"
    destination.write_text("KEEP")
    result = run_cli("--output", str(destination))
    assert result.returncode == 2
    assert destination.read_text() == "KEEP"
    assert "already exists" in result.stderr


def test_cli_rejects_invalid_configuration():
    result = run_cli("--steps", "0")
    assert result.returncode == 2
    assert "positive integer" in result.stderr


def test_cli_reports_divergence_without_a_traceback():
    result = run_cli("--steps", "1", "--learning-rate", "1e200")
    assert result.returncode == 2
    assert "Experiment failed:" in result.stderr
    assert "non-finite" in result.stderr
    assert "Traceback" not in result.stderr
