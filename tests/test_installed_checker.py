"""Exercise process-group ownership at the installed-wheel smoke boundary."""

import errno
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

import pytest

CHECKER = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "check_installed.py"))
run_checked = CHECKER["run_checked"]


def test_capture_success():
    result = run_checked([sys.executable, "-c", "print('checked')"], timeout=5)
    assert result.returncode == 0 and result.stdout == "checked\n"


def test_nonzero_fails():
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_checked([sys.executable, "-c", "raise SystemExit(7)"], timeout=5)
    assert error.value.returncode == 7


def test_missing_executable_fails():
    with pytest.raises(FileNotFoundError):
        run_checked(["/definitely-not-a-pytorch-runtime-executable"], timeout=1)


def test_timeout_terminates_real_descendant(tmp_path):
    pid_path = tmp_path / "worker.pid"
    child = "import time; time.sleep(60)"
    parent = (
        "import subprocess, sys, pathlib, time; "
        f"p=subprocess.Popen([sys.executable, '-c', {child!r}]); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        run_checked([sys.executable, "-c", parent], timeout=1)
    assert pid_path.exists(), "test must actually start a descendant before timeout"
    pid = int(pid_path.read_text())

    assert_descendant_terminated(pid)


def assert_descendant_terminated(pid):
    stat = Path(f"/proc/{pid}/stat")
    # The system's reaper may not have collected a dead child yet; Z is terminated.
    for _ in range(50):
        try:
            state = stat.read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            break  # It may disappear before open (ENOENT) or during read (ESRCH).
        if state == "Z":
            break
        time.sleep(0.01)
    else:
        os.kill(pid, 9)
        pytest.fail("timed-out command left a running descendant")


def test_descendant_observation_accepts_gone_or_zombie(monkeypatch):
    for observation in (
        FileNotFoundError(errno.ENOENT, "gone before open"),
        ProcessLookupError(errno.ESRCH, "gone during read"),
        "123 (worker) Z",
    ):
        reads = []

        def read_stat(path, reads=reads, observation=observation):
            reads.append(path)
            if isinstance(observation, OSError):
                raise observation
            return observation

        monkeypatch.setattr(Path, "read_text", read_stat)
        assert_descendant_terminated(123)
        assert reads == [Path("/proc/123/stat")]


def test_descendant_observation_propagates_unrelated_errors(monkeypatch):
    for error in (
        PermissionError(errno.EACCES, "not readable"),
        OSError(errno.EIO, "read failure"),
    ):

        def read_stat(_path, error=error):
            raise error

        monkeypatch.setattr(Path, "read_text", read_stat)
        with pytest.raises(type(error)) as caught:
            assert_descendant_terminated(123)
        assert caught.value is error


def test_descendant_observation_rejects_still_running(monkeypatch):
    reads, sleeps, kills = [], [], []

    def read_stat(path):
        reads.append(path)
        return "123 (worker) S"

    monkeypatch.setattr(Path, "read_text", read_stat)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(os, "kill", lambda pid, signal: kills.append((pid, signal)))
    with pytest.raises(pytest.fail.Exception, match="left a running descendant"):
        assert_descendant_terminated(123)
    assert reads == [Path("/proc/123/stat")] * 50
    assert sleeps == [0.01] * 50
    assert kills == [(123, 9)]


def test_explicit_linux_scope(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Linux"):
        run_checked([sys.executable, "-c", "print('not run')"], timeout=1)


def test_nested_runtime_budget_fits_outer_limit():
    runtime = next(c for c in CHECKER["COMMANDS"] if c[0] == "runtime")
    args = runtime[2]
    repeats = int(args[args.index("--repeats") + 1])
    per_child = int(args[args.index("--timeout") + 1])
    assert repeats * per_child < 300
