"""Retain completed and failed structured evidence at the wheel CLI boundary."""

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/check_installed.py"


def test_failed_child_keeps_previous_and_partial_reports(monkeypatch):
    functions = runpy.run_path(str(SCRIPT))
    checker = functions["check_installed"]
    context = checker.__globals__
    monkeypatch.setitem(
        context, "COMMANDS", (("runtime", "runtime", []), ("numerical", "numeric", []))
    )
    failed = {"all_checks_passed": False, "samples": [{"retained": True}]}

    def run(command, **kwargs):
        if "-c" in command:
            return SimpleNamespace(stdout='{"noneditable_wheel":true}')
        output = Path(command[-1])
        if "numeric" in command:
            output.write_text(json.dumps(failed))
            raise subprocess.CalledProcessError(2, command, stderr="secret/path")
        evidence = {"all_checks_passed": True, "samples": [{"prior": True}]}
        output.write_text(json.dumps(evidence))
        return SimpleNamespace(stdout=json.dumps(evidence), returncode=0)

    monkeypatch.setitem(context, "run_checked", run)
    result = checker()
    assert not result["all_checks_passed"]
    assert result["checks"]["runtime"]["report"]["samples"] == [{"prior": True}]
    assert result["checks"]["numerical"]["report"] == failed
    assert result["checks"]["numerical"]["exit_code"] == 2
    assert "secret" not in json.dumps(result)


def test_checker_cli_writes_failed_report_and_nonzero(monkeypatch, tmp_path, capsys):
    functions = runpy.run_path(str(SCRIPT))
    main = functions["main"]
    result = {"all_checks_passed": False, "checks": {"numerical": {"all_checks_passed": False}}}
    monkeypatch.setitem(main.__globals__, "check_installed", lambda: result)
    output = tmp_path / "failure.json"
    monkeypatch.setattr(sys, "argv", ["check_installed.py", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert json.loads(output.read_text()) == result
    assert json.loads(capsys.readouterr().out) == result
