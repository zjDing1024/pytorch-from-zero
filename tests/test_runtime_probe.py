"""Resource protocol tests: real subprocess smoke plus malformed evidence paths."""

import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pytorch_lab import runtime_probe as probe


def measurement():
    return {
        "workload_wall_seconds": 2.0,
        "workload_process_cpu_seconds": 1.8,
        "process_high_water_rss_kib": 2048,
        "process_high_water_rss_mib": 2.0,
        "scientific_result_sha256": "a" * 64,
        "workload": {
            "epochs_per_fit": 1,
            "fits": 6,
            "updates_total": 42,
            "samples_seen_total": 642,
            "all_checks_passed": True,
        },
        "environment": {
            "device": "cpu",
            "torch_cuda_build": None,
            "intraop_threads": 1,
            "interop_threads": 1,
            "python_isolated_mode": True,
        },
        "package_files_sha256": {"runtime_probe.py": "b" * 64},
    }


@pytest.mark.parametrize("epochs", [0, -1, 121, True, 1.0, "1", None])
def test_invalid_epochs(epochs):
    with pytest.raises(ValueError, match="epochs"):
        probe.validate_config(epochs, 1, 3)


@pytest.mark.parametrize("repeats", [0, -1, 11, True, 1.0, "1", None])
def test_invalid_repeats(repeats):
    with pytest.raises(ValueError, match="repeats"):
        probe.validate_config(1, repeats, 3)


@pytest.mark.parametrize("timeout", [0, -1, 3601, True, "1", None, float("nan"), float("inf")])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        probe.validate_config(1, 1, timeout)


def test_valid_boundaries():
    probe.validate_config(1, 1, 0.01)
    probe.validate_config(120, 10, 3600)


def test_digest_only_excludes_explicit_runtime_fields():
    a = {"score": 2, "created_at_utc": "a", "environment": {"python": "3.12"}}
    b = {"environment": {}, "created_at_utc": "b", "score": 2}
    assert probe.result_digest(a) == probe.result_digest(b)
    assert probe.result_digest(a) != probe.result_digest(dict(b, score=3))
    with pytest.raises(ValueError):
        probe.result_digest({"score": float("nan")})


def test_package_provenance_includes_code_data_and_license():
    hashes = probe.package_file_hashes()
    assert {"runtime_probe.py", "wine.py", "data/wine.csv", "data/LICENSE.scikit-learn"} <= set(
        hashes
    )
    assert all(len(v) == 64 for v in hashes.values())
    assert not any("__pycache__" in p or p.startswith("/") for p in hashes)


@pytest.mark.parametrize("bad", [-1, True, "1", float("nan"), float("inf")])
@pytest.mark.parametrize(
    "field", ["workload_wall_seconds", "workload_process_cpu_seconds", "process_high_water_rss_kib"]
)
def test_reject_invalid_resource_number(field, bad):
    value = measurement()
    value[field] = bad
    with pytest.raises(ValueError):
        probe._validate_measurement(value, 1)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("epochs_per_fit", 2),
        ("fits", 5),
        ("updates_total", 41),
        ("samples_seen_total", 641),
        ("all_checks_passed", False),
    ],
)
def test_reject_wrong_workload(field, bad):
    value = measurement()
    value["workload"][field] = bad
    with pytest.raises(ValueError, match="workload"):
        probe._validate_measurement(value, 1)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("device", "cuda"),
        ("torch_cuda_build", "12"),
        ("intraop_threads", 2),
        ("interop_threads", 2),
        ("python_isolated_mode", False),
    ],
)
def test_reject_wrong_controls(field, bad):
    value = measurement()
    value["environment"][field] = bad
    with pytest.raises(ValueError, match="controls"):
        probe._validate_measurement(value, 1)


@pytest.mark.parametrize("digest", [None, "x" * 64, "a" * 63, "A" * 64])
def test_reject_invalid_digest(digest):
    value = measurement()
    value["scientific_result_sha256"] = digest
    with pytest.raises(ValueError, match="digest"):
        probe._validate_measurement(value, 1)


def test_reject_wrong_rss_conversion_and_missing_provenance():
    value = measurement()
    value["process_high_water_rss_mib"] = 2048
    with pytest.raises(ValueError, match="RSS"):
        probe._validate_measurement(value, 1)
    value = measurement()
    value["package_files_sha256"] = {}
    with pytest.raises(ValueError, match="provenance"):
        probe._validate_measurement(value, 1)


def test_probe_uses_no_shell_isolation_timeout_and_thread_controls(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout=json.dumps(measurement()), stderr="optional warning")

    monkeypatch.setattr(probe.subprocess, "run", run)
    result = probe.run_probe(1, 3, 8)
    assert result["all_checks_passed"]
    assert result["summary"]["workload_wall_seconds"] == {"min": 2, "median": 2, "max": 2}
    assert len(calls) == 3
    for cmd, kwargs in calls:
        assert cmd[:4] == [sys.executable, "-I", "-m", "pytorch_lab.runtime_probe"]
        assert kwargs["timeout"] == 8 and kwargs["check"]
        assert "shell" not in kwargs
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["MKL_NUM_THREADS"] == "1"
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert result["samples"][0]["stderr_nonempty"] is True
    assert "optional warning" not in json.dumps(result)


@pytest.mark.parametrize("kind", ["digest", "environment", "package"])
def test_reproducibility_mismatch_fails(monkeypatch, kind):
    values = [measurement(), copy.deepcopy(measurement())]
    if kind == "digest":
        values[1]["scientific_result_sha256"] = "c" * 64
    elif kind == "environment":
        values[1]["environment"]["python"] = "different"
    else:
        values[1]["package_files_sha256"]["runtime_probe.py"] = "c" * 64
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=json.dumps(values.pop(0)), stderr=""),
    )
    assert not probe.run_probe(1, 2, 8)["all_checks_passed"]


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired("child", 1),
        subprocess.CalledProcessError(1, "child"),
        FileNotFoundError("missing executable"),
    ],
)
def test_child_failure_propagates(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(probe.subprocess, "run", fail)
    with pytest.raises(type(error)):
        probe.run_probe(1, 1, 1)


def test_invalid_json_fails(monkeypatch):
    monkeypatch.setattr(
        probe.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="not JSON", stderr="")
    )
    with pytest.raises(ValueError):
        probe.run_probe(1, 1, 1)


def test_unsupported_platform_is_explicit(monkeypatch):
    monkeypatch.setattr(probe.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Linux"):
        probe.run_probe(1, 1, 1)
    with pytest.raises(RuntimeError, match="Linux"):
        probe._worker(1)


def cli(*args):
    return subprocess.run(
        [sys.executable, "-I", "-m", "pytorch_lab.runtime_probe", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_real_two_fresh_processes(tmp_path):
    output = tmp_path / "measurement.json"
    run = cli("--epochs", 1, "--repeats", 2, "--output", output)
    assert run.returncode == 0, run.stderr
    report = json.loads(output.read_text())
    assert json.loads(run.stdout) == report
    assert report["all_checks_passed"]
    for sample in report["samples"]:
        assert sample["environment"]["python_isolated_mode"]
        assert sample["workload"]["updates_total"] == 42
        assert sample["process_high_water_rss_mib"] > 0
        assert sample["child_end_to_end_wall_seconds"] >= sample["workload_wall_seconds"]


@pytest.mark.parametrize("symlink", [False, True])
def test_cli_does_not_clobber(tmp_path, symlink):
    output = tmp_path / "existing.json"
    if symlink:
        output.symlink_to(tmp_path / "missing-target")
    else:
        output.write_text("keep me")
    run = cli("--epochs", 1, "--repeats", 1, "--output", output)
    assert run.returncode == 2
    assert output.is_symlink() if symlink else output.read_text() == "keep me"
    assert not (tmp_path / "missing-target").exists()


def test_real_timeout_fails_without_success_file(tmp_path):
    output = tmp_path / "never.json"
    run = cli("--epochs", 120, "--timeout", 0.001, "--output", output)
    assert run.returncode == 2 and not output.exists()


def test_cli_failed_reproducibility_uses_exit_one(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["runtime_probe", "--output", str(tmp_path / "failed.json")])
    monkeypatch.setattr(probe, "run_probe", lambda *a: {"all_checks_passed": False})
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 1
    assert json.loads((tmp_path / "failed.json").read_text())["all_checks_passed"] is False
