"""CLI evidence and no-clobber checks, independent of feature-conditioning labels."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from pytorch_lab import feature_conditioning_cli as cli


def test_real_subprocess(tmp_path):
    output = tmp_path / "nested" / "report.json"
    run = subprocess.run(
        [sys.executable, "-m", "pytorch_lab.feature_conditioning_cli", "--output", str(output)],
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    report = json.loads(output.read_text())
    assert report == json.loads(run.stdout)
    assert report["all_checks_passed"] is True


@pytest.mark.parametrize("existing", ["file", "directory", "dangling_symlink"])
def test_existing_target_rejected(tmp_path, monkeypatch, existing):
    output = tmp_path / "existing"
    if existing == "file":
        output.write_text("original")
    elif existing == "directory":
        output.mkdir()
    else:
        output.symlink_to(tmp_path / "absent")

    def should_not_run():
        raise AssertionError("experiment executed despite existing target")

    monkeypatch.setattr(cli, "run_experiment", should_not_run)
    with pytest.raises(SystemExit) as error:
        cli.main(["--output", str(output)])
    assert error.value.code == 2
    if existing == "file":
        assert output.read_text() == "original"
    elif existing == "dangling_symlink":
        assert output.is_symlink() and not (tmp_path / "absent").exists()


@pytest.mark.parametrize("args", [[], ["--unknown"], ["--output"]])
def test_argument_errors(args):
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2


def test_failed_checks_preserved(tmp_path, monkeypatch, capsys):
    result = {"all_checks_passed": False, "checks": {"actual_check": False}}
    monkeypatch.setattr(cli, "run_experiment", lambda: result)
    output = tmp_path / "failed.json"
    assert cli.main(["--output", str(output)]) == 1
    assert json.loads(output.read_text()) == json.loads(capsys.readouterr().out) == result


def test_nonfinite_report_does_not_create_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_experiment", lambda: {"loss": float("nan")})
    output = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        cli.main(["--output", str(output)])
    assert not output.exists()


def test_output_race_does_not_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "race.json"

    def experiment():
        output.write_text("concurrent writer")
        return {"all_checks_passed": True}

    monkeypatch.setattr(cli, "run_experiment", experiment)
    with pytest.raises(FileExistsError):
        cli.main(["--output", str(output)])
    assert output.read_text() == "concurrent writer"


def test_installed_commands_keep_new_entry_last():
    commands = runpy.run_path(str(Path(__file__).parents[1] / "scripts/check_installed.py"))[
        "COMMANDS"
    ]
    assert len(commands) == 25
    assert commands[16] == ("feature_conditioning", "pytorch_lab.feature_conditioning_cli", [])


@pytest.mark.parametrize("success", [True, False])
def test_installed_checker_retains_conditioning_report(monkeypatch, success):
    from types import SimpleNamespace

    functions = runpy.run_path(str(Path(__file__).parents[1] / "scripts/check_installed.py"))
    checker = functions["check_installed"]
    context = checker.__globals__
    monkeypatch.setitem(
        context,
        "COMMANDS",
        (("feature_conditioning", "pytorch_lab.feature_conditioning_cli", []),),
    )
    evidence = {"all_checks_passed": success, "cases": [{"retained": True}]}

    def run(command, **kwargs):
        if "-c" in command:
            return SimpleNamespace(stdout='{"noneditable_wheel":true}')
        Path(command[-1]).write_text(json.dumps(evidence))
        if not success:
            raise subprocess.CalledProcessError(1, command, stderr="private diagnostic")
        return SimpleNamespace(stdout=json.dumps(evidence), returncode=0)

    monkeypatch.setitem(context, "run_checked", run)
    result = checker()
    assert result["all_checks_passed"] is success
    assert result["checks"]["feature_conditioning"]["report"] == evidence
    assert "private diagnostic" not in json.dumps(result)
