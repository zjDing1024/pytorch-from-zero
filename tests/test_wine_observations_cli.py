"""Public Wine observation CLI and installed-evidence delivery boundaries."""

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pytorch_lab import wine_observations_cli as cli


def test_real_subprocess(tmp_path):
    output = tmp_path / "new-directory" / "report.json"
    completed = subprocess.run(
        [sys.executable, "-m", "pytorch_lab.wine_observations_cli", "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert output.read_text() == completed.stdout
    report = json.loads(completed.stdout)
    assert report["all_checks_passed"] is True


@pytest.mark.parametrize("target_kind", ["file", "directory", "dangling", "live_symlink"])
def test_existing_target_prevents_experiment(tmp_path, monkeypatch, target_kind):
    output = tmp_path / "evidence"
    destination = tmp_path / "destination"
    if target_kind == "file":
        output.write_text("original")
    elif target_kind == "directory":
        output.mkdir()
    else:
        if target_kind == "live_symlink":
            destination.write_text("original")
        output.symlink_to(destination)

    def unexpected_experiment():
        raise AssertionError("refused output must not execute training")

    monkeypatch.setattr(cli, "run_experiment", unexpected_experiment)
    with pytest.raises(SystemExit) as error:
        cli.main(["--output", str(output)])
    assert error.value.code == 2
    if target_kind == "file":
        assert output.read_text() == "original"
    elif target_kind == "directory":
        assert output.is_dir()
    else:
        assert output.is_symlink()
        if target_kind == "live_symlink":
            assert destination.read_text() == "original"
        else:
            assert not destination.exists()


@pytest.mark.parametrize("arguments", [[], ["--output"], ["--unknown"]])
def test_invalid_arguments(arguments):
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("flag,expected_exit", [(True, 0), (False, 1), (1, 1), (None, 1)])
def test_exact_success_flag_and_json_delivery(tmp_path, monkeypatch, capsys, flag, expected_exit):
    evidence = {"all_checks_passed": flag, "measurements": {"ratio": None}}
    monkeypatch.setattr(cli, "run_experiment", lambda: evidence)
    output = tmp_path / "report.json"
    assert cli.main(["--output", str(output)]) == expected_exit
    stdout = capsys.readouterr().out
    assert output.read_text() == stdout
    assert json.loads(stdout) == evidence


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_evidence_leaves_no_output(tmp_path, monkeypatch, value):
    monkeypatch.setattr(cli, "run_experiment", lambda: {"loss": value})
    output = tmp_path / "new-directory" / "report.json"
    with pytest.raises(ValueError):
        cli.main(["--output", str(output)])
    assert not output.parent.exists()


def test_experiment_failure_leaves_no_output(tmp_path, monkeypatch):
    def failed():
        raise RuntimeError("experimental failure")

    monkeypatch.setattr(cli, "run_experiment", failed)
    output = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="experimental failure"):
        cli.main(["--output", str(output)])
    assert not output.exists()


def test_concurrent_output_creation_is_not_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "report.json"

    def racing():
        output.write_text("other writer")
        return {"all_checks_passed": True}

    monkeypatch.setattr(cli, "run_experiment", racing)
    with pytest.raises(FileExistsError):
        cli.main(["--output", str(output)])
    assert output.read_text() == "other writer"


def test_eighteen_installed_commands_and_source_ci_entry():
    root = Path(__file__).parents[1]
    commands = runpy.run_path(str(root / "scripts/check_installed.py"))["COMMANDS"]
    assert len(commands) == 20
    assert commands[17] == ("wine_observations", "pytorch_lab.wine_observations_cli", [])
    workflow = (root / ".github/workflows/tests.yml").read_text()
    assert (
        "python -m pytorch_lab.wine_observations_cli "
        "--output results/local-ci-wine-observations.json"
    ) in workflow


@pytest.mark.parametrize("successful", [True, False])
def test_installed_checker_retains_actual_report_and_hides_stderr(monkeypatch, successful):
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "scripts/check_installed.py"))
    checker = namespace["check_installed"]
    monkeypatch.setitem(
        checker.__globals__,
        "COMMANDS",
        (("wine_observations", "pytorch_lab.wine_observations_cli", []),),
    )
    evidence = {"all_checks_passed": successful, "runs": [{"observed": True}]}

    def execute(command, **kwargs):
        if "-c" in command:
            return SimpleNamespace(stdout='{"noneditable_wheel":true}')
        Path(command[-1]).write_text(json.dumps(evidence))
        if not successful:
            raise subprocess.CalledProcessError(1, command, stderr="private stderr")
        return SimpleNamespace(stdout=json.dumps(evidence), returncode=0)

    monkeypatch.setitem(checker.__globals__, "run_checked", execute)
    result = checker()
    assert result["all_checks_passed"] is successful
    assert result["checks"]["wine_observations"]["report"] == evidence
    assert "private stderr" not in json.dumps(result)
