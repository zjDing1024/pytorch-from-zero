"""Separate interpreter CLI checks; existing lab CLIs remain unchanged."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pytorch_lab.checkpoint import load_checkpoint


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "pytorch_lab.checkpoint_cli", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_separate_process_training_and_resume(tmp_path):
    full, cut, resumed = (tmp_path / name for name in ("full.pt", "cut.pt", "resumed.pt"))
    first = run_cli("train", "--seed", 7, "--epochs", 8, "--checkpoint", full)
    second = run_cli("train", "--seed", 7, "--epochs", 3, "--checkpoint", cut)
    third = run_cli("resume", "--checkpoint", cut, "--epochs", 8, "--save-checkpoint", resumed)
    for process in (first, second, third):
        assert process.returncode == 0, process.stderr
    full_report, resumed_report = json.loads(first.stdout), json.loads(third.stdout)
    assert full_report["process_id"] != resumed_report["process_id"] != os.getpid()
    assert full_report["training"] == resumed_report["training"]
    assert (
        load_checkpoint(full).snapshot()["content_sha256"]
        == load_checkpoint(resumed).snapshot()["content_sha256"]
    )


@pytest.mark.parametrize(
    "option,value",
    [
        ("--epochs", -1),
        ("--momentum", 0),
        ("--momentum", 1),
        ("--batch-size", 0),
        ("--learning-rate", "nan"),
        ("--seed", -1),
    ],
)
def test_train_bad_args_do_not_create_file(tmp_path, option, value):
    checkpoint = tmp_path / "output.pt"
    process = run_cli("train", "--checkpoint", checkpoint, option, value)
    assert process.returncode == 2
    assert not checkpoint.exists()


def test_cli_never_overwrites_checkpoint_or_json(tmp_path):
    checkpoint, report = tmp_path / "out.pt", tmp_path / "out.json"
    report.write_text("keep me")
    result = run_cli("train", "--epochs", 1, "--checkpoint", checkpoint, "--output", report)
    assert result.returncode == 2
    assert report.read_text() == "keep me"
    assert not checkpoint.exists()
    result = run_cli("train", "--epochs", 1, "--checkpoint", checkpoint)
    assert result.returncode == 0, result.stderr
    original = checkpoint.read_bytes()
    result = run_cli("train", "--epochs", 2, "--checkpoint", checkpoint)
    assert result.returncode == 2
    assert checkpoint.read_bytes() == original


def test_cli_rejects_same_destination(tmp_path):
    destination = tmp_path / "file"
    result = run_cli("train", "--checkpoint", destination, "--output", destination)
    assert result.returncode == 2
    assert not destination.exists()


def test_resume_corruption_and_backwards_rejected(tmp_path):
    checkpoint = tmp_path / "bad.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    result = run_cli("resume", "--checkpoint", checkpoint, "--epochs", 5)
    assert result.returncode == 2
    good = tmp_path / "good.pt"
    assert run_cli("train", "--epochs", 2, "--checkpoint", good).returncode == 0
    result = run_cli("resume", "--checkpoint", good, "--epochs", 1)
    assert result.returncode == 2


def test_verify_runs_fresh_process_comparison(tmp_path):
    output = tmp_path / "evidence.json"
    result = run_cli("verify", "--epochs", 5, "--split-epoch", 2, "--output", output)
    assert result.returncode == 0, result.stderr
    evidence = json.loads(output.read_text())
    assert evidence == json.loads(result.stdout)
    assert evidence["all_checks_passed"]
    assert all(evidence["checks"].values())
    assert evidence["full_vs_resumed_max_parameter_gap"] == 0
    assert evidence["negative_controls"]["omitted_momentum_max_parameter_gap"] > 1e-12
    assert len(set(evidence["process_ids"].values())) == 3


@pytest.mark.parametrize("cut,total", [(0, 5), (5, 5), (6, 5), (-1, 5)])
def test_verify_invalid_cut_returns_nonzero(cut, total):
    result = run_cli("verify", "--epochs", total, "--split-epoch", cut)
    assert result.returncode == 2


def test_readme_entrypoint_exists():
    # The module is importable without touching the original project's CLI.
    assert Path(__file__).parents[1].joinpath("src/pytorch_lab/checkpoint_cli.py").is_file()
