"""Supervisory test deadlines must cover unchanged per-child CLI deadlines."""

from types import SimpleNamespace

import pytest
import test_numerical_probe
import test_runtime_probe


@pytest.mark.parametrize("module", [test_runtime_probe, test_numerical_probe])
def test_parent_budget_covers_children_without_changing_commands(monkeypatch, module):
    observed = []

    def run(command, **kwargs):
        observed.append((command, kwargs["timeout"]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    arguments = ("--epochs", 1, "--repeats", 2)
    module.cli(*arguments)
    command, timeout = observed[-1]
    assert timeout == 2 * 300 + 60
    assert command[-len(arguments) :] == list(map(str, arguments))
    assert "--timeout" not in command  # The production default remains untouched.
    module.cli("--repeats", 2, "--timeout", 60)
    assert observed[-1][1] == 2 * 60 + 60
    module.cli("--repeats", 2, "--timeout", 0.001)
    assert observed[-1][1] == 2 * 0.001 + 60
    assert observed[-1][0][-2:] == ["--timeout", "0.001"]
    if module is test_numerical_probe:
        module.cli("--dispatch-study", "--timeout", 60)
        assert observed[-1][1] == 4 * 60 + 60
