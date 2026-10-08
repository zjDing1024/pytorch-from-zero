"""CLI contracts: finite JSON, clear failures and atomic non-overwriting evidence."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pytorch_lab.checkpoint as checkpoint
import pytorch_lab.wine_cli as cli


def invoke(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["wine_cli", *map(str, arguments)])
    cli.main()


def test_fresh_process_one_epoch_cli_writes_complete_json_and_prints_same_bytes(tmp_path):
    destination = tmp_path / "wine.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(Path(cli.__file__).resolve().parents[1]), environment.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytorch_lab.wine_cli",
            "--epochs",
            "1",
            "--output",
            str(destination),
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert destination.read_text() == result.stdout
    report = json.loads(result.stdout)
    assert report["all_checks_passed"] and report["protocol"]["config"]["epochs"] == 1
    assert report["protocol"]["seeds"] == [42, 7, 123]
    assert set(report["models"]) == {"linear", "mlp16"}
    assert all(len(item["runs"]) == 3 for item in report["models"].values())
    assert sorted(path.name for path in tmp_path.iterdir()) == ["wine.json"]


def test_cli_rejects_malformed_and_out_of_range_epochs_before_training(monkeypatch, capsys):
    calls = []

    def validate_only(config):
        calls.append(config.epochs)
        config.validate()
        raise AssertionError("invalid epochs reached training")

    monkeypatch.setattr(cli, "run_wine_experiment", validate_only)
    for value in ("banana", "1.5", "true", "0", "-1", "1001"):
        with pytest.raises(SystemExit) as failure:
            invoke(monkeypatch, "--epochs", value)
        assert failure.value.code == 2
        captured = capsys.readouterr()
        assert captured.out == "" and "epochs" in captured.err
    assert calls == [0, -1, 1001]


def test_existing_file_directory_and_live_or_dangling_symlink_are_never_overwritten(
    tmp_path, monkeypatch, capsys
):
    original = tmp_path / "existing.json"
    original.write_bytes(b"original evidence\n")
    directory = tmp_path / "directory"
    directory.mkdir()
    live_link, dangling_link = tmp_path / "live.json", tmp_path / "dangling.json"
    live_link.symlink_to(original)
    dangling_link.symlink_to(tmp_path / "missing.json")

    def unexpected_training(config):
        raise AssertionError("output conflict must be detected before training")

    monkeypatch.setattr(cli, "run_wine_experiment", unexpected_training)
    for path in (original, directory, live_link, dangling_link):
        with pytest.raises(SystemExit) as failure:
            invoke(monkeypatch, "--output", path)
        assert failure.value.code == 2
        captured = capsys.readouterr()
        assert captured.out == "" and "already exists" in captured.err
    assert original.read_bytes() == b"original evidence\n"
    assert directory.is_dir() and live_link.is_symlink() and dangling_link.is_symlink()
    assert not (tmp_path / "missing.json").exists()


def test_cli_atomic_fsync_failure_leaves_no_partial_output_or_temporary_file(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "run_wine_experiment", lambda config: {"all_checks_passed": True})

    def fail_fsync(descriptor):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(checkpoint.os, "fsync", fail_fsync)
    destination = tmp_path / "new.json"
    with pytest.raises(SystemExit) as failure:
        invoke(monkeypatch, "--output", destination)
    assert failure.value.code == 2
    assert list(tmp_path.iterdir()) == []
    captured = capsys.readouterr()
    assert captured.out == "" and "simulated storage failure" in captured.err


def test_cli_atomic_publication_race_preserves_competing_writer_and_cleans_temporary(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "run_wine_experiment", lambda config: {"all_checks_passed": True})
    original_link = checkpoint.os.link
    destination = tmp_path / "racing.json"

    def concurrent_link(source, target):
        Path(target).write_bytes(b"other writer won\n")
        original_link(source, target)

    monkeypatch.setattr(checkpoint.os, "link", concurrent_link)
    with pytest.raises(SystemExit) as failure:
        invoke(monkeypatch, "--output", destination)
    assert failure.value.code == 2
    assert destination.read_bytes() == b"other writer won\n"
    assert list(tmp_path.iterdir()) == [destination]
    captured = capsys.readouterr()
    assert captured.out == "" and "Wine experiment failed" in captured.err


def test_cli_rejects_nonfinite_json_before_publication(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "run_wine_experiment", lambda config: {"all_checks_passed": True, "loss": float("nan")}
    )
    destination = tmp_path / "nonfinite.json"
    with pytest.raises(SystemExit) as failure:
        invoke(monkeypatch, "--output", destination)
    assert failure.value.code == 2
    assert list(tmp_path.iterdir()) == []
    captured = capsys.readouterr()
    assert captured.out == "" and "JSON" in captured.err


def test_cli_failed_protocol_keeps_complete_evidence_and_uses_exit_one(
    tmp_path, monkeypatch, capsys
):
    report = {"all_checks_passed": False, "checks": {"deliberate_failure": False}}
    monkeypatch.setattr(cli, "run_wine_experiment", lambda config: report)
    destination = tmp_path / "failed-protocol.json"
    with pytest.raises(SystemExit) as failure:
        invoke(monkeypatch, "--output", destination)
    assert failure.value.code == 1
    captured = capsys.readouterr()
    assert json.loads(destination.read_text()) == report
    assert destination.read_text() == captured.out
    assert "data-protocol checks failed" in captured.err


def test_cli_missing_parent_is_clear_failure_without_partial_stdout(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_wine_experiment", lambda config: {"all_checks_passed": True})
    with pytest.raises(SystemExit) as failure:
        invoke(monkeypatch, "--output", tmp_path / "missing" / "new.json")
    assert failure.value.code == 2
    assert list(tmp_path.iterdir()) == []
    captured = capsys.readouterr()
    assert captured.out == "" and "Wine experiment failed" in captured.err
