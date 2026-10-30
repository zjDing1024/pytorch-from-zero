"""The public CLI must perform the experiment, not only parse its flags."""

import json
import subprocess
import sys


def test_cli_fresh_process_matrix_and_no_overwrite(tmp_path):
    output = tmp_path / "report.json"
    command = [sys.executable, "-m", "pytorch_lab.dropout_rng_cli", "--output", str(output)]
    run = subprocess.run(command, check=True, text=True, capture_output=True, timeout=120)
    report = json.loads(run.stdout)
    assert report == json.loads(output.read_text())
    assert report["all_checks_passed"] and all(report["checks"].values())
    assert report["worker_processes"] == 6
    assert report["recovery_cases"] == 9
    assert report["compared_steps"] == 45
    assert report["optimizer_updates"] == 0
    before = output.read_bytes()
    retry = subprocess.run(command, check=False, text=True, capture_output=True, timeout=15)
    assert retry.returncode != 0
    assert output.read_bytes() == before
