import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pytorch_lab import wine
from pytorch_lab import wine_pipeline_cli as pipeline
from pytorch_lab.wine_model_bundle import strict_json


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    output = tmp_path_factory.mktemp("pipeline") / "report.json"
    report = pipeline.run_pipeline(output)
    return output, report


def test_complete_path_and_same_model(completed):
    output, report = completed
    artifacts = output.with_name("report-artifacts")
    assert strict_json(output) == report
    assert report["all_checks_passed"]
    assert len(report["checks"]) == 5
    assert report["training"]["work"] == {"updates": 140, "training_visits": 2140}
    assert set(report["artifacts"]) == {
        "training.json",
        "evaluation.json",
        "model.json",
        "request.json",
        "predictions.json",
    }
    for name, digest in report["artifacts"].items():
        assert hashlib.sha256((artifacts / name).read_bytes()).hexdigest() == digest
    bundle = strict_json(artifacts / "model.json")
    assert report["evaluation"]["model_sha256"] == bundle["sha256"]
    assert report["evaluation"] == strict_json(artifacts / "evaluation.json")


def test_validation_only_and_train_fitted_scaler(completed):
    output, report = completed
    x, y = wine.load_wine()
    split = wine.build_split(y)
    evaluation = report["evaluation"]
    assert evaluation["metrics"]["row_ids"] == split["validation"]
    assert evaluation["test_rows_scored"] == 0
    assert evaluation["used_for_selection"] is False
    assert set(evaluation["metrics"]["row_ids"]).isdisjoint(split["test"])
    bundle = strict_json(output.with_name("report-artifacts") / "model.json")
    scaler = wine.Standardizer.fit(x[split["train"]])
    assert bundle["payload"]["mean"] == scaler.mean.tolist()
    assert bundle["payload"]["scale"] == scaler.scale.tolist()


def test_refuse_existing_report_and_directory(completed, tmp_path):
    output, _ = completed
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        pipeline.run_pipeline(output)
    assert output.read_bytes() == original
    target = tmp_path / "report.json"
    target.with_name("report-artifacts").mkdir()
    with pytest.raises(FileExistsError):
        pipeline.run_pipeline(target)
    assert not target.exists()


def test_refuse_dangling_report_link(tmp_path):
    target = tmp_path / "report.json"
    target.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        pipeline.run_pipeline(target)
    assert not target.with_name("report-artifacts").exists()


def test_child_failure_does_not_claim_success(monkeypatch, tmp_path):
    original_threads = torch.get_num_threads()

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(2, "predict")

    monkeypatch.setattr(pipeline.subprocess, "run", fail)
    output = tmp_path / "report.json"
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_pipeline(output)
    assert not output.exists()
    assert (tmp_path / "report-artifacts/model.json").exists()
    assert torch.get_num_threads() == original_threads


def test_real_cli_and_no_overwrite(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    output = tmp_path / "run.json"
    command = [sys.executable, "-m", "pytorch_lab.wine_pipeline_cli", "--output", str(output)]
    run = subprocess.run(command, env=env, check=True, capture_output=True, text=True, timeout=60)
    assert json.loads(run.stdout) == strict_json(output)
    assert json.loads(run.stdout)["all_checks_passed"]
    failed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert failed.returncode == 2


def test_ci_runs_source_and_installed_pipeline():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/tests.yml").read_text()
    assert workflow.count("-m pytorch_lab.wine_pipeline_cli") == 2
    assert "container-pipeline.json" in workflow
