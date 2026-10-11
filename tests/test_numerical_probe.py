"""Evidence integrity and actual fresh-process capture, without tolerance inflation."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from pytorch_lab import numerical_probe as probe
from pytorch_lab.runtime_probe import result_digest

ROOT = Path(__file__).parents[1]


def sample():
    wine = json.loads((ROOT / "results/2026-10-09-wine-cpu.json").read_text())
    return {
        "workload_wall_seconds": 1.0,
        "workload_process_cpu_seconds": 0.9,
        "process_high_water_rss_kib": 2048,
        "process_high_water_rss_mib": 2.0,
        "scientific_result_sha256": result_digest(wine),
        "wine_result": wine,
        "workload": {
            "epochs_per_fit": 120,
            "fits": 6,
            "updates_total": 5040,
            "samples_seen_total": 77040,
            "all_checks_passed": True,
        },
        "environment": {
            "device": "cpu",
            "torch_cuda_build": None,
            "intraop_threads": 1,
            "interop_threads": 1,
            "python_isolated_mode": True,
        },
        "package_files_sha256": {"wine.py": "a" * 64},
        "cpu_runtime": {
            "torch_cpu_capability": "AVX512",
            "cpu_affinity": [0, 1],
            "environment_controls": {"ATEN_CPU_CAPABILITY": None},
        },
    }


def mock_workers(monkeypatch, mutate=None):
    calls = []

    def run(cmd, **kwargs):
        value = sample()
        setting = kwargs["env"].get("ATEN_CPU_CAPABILITY")
        value["cpu_runtime"]["environment_controls"]["ATEN_CPU_CAPABILITY"] = setting
        if setting == "default":
            value["cpu_runtime"]["torch_cpu_capability"] = "DEFAULT"
        if mutate:
            mutate(value, len(calls))
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout=json.dumps(value), stderr="private/path warning")

    monkeypatch.setattr(probe.subprocess, "run", run)
    return calls


def test_cpu_inventory_allowlist_excludes_identifiers():
    block = "vendor_id: Example\nmodel name: Test CPU\nflags: z a\nSerial: SECRET\nprocessor: 0"
    assert probe.cpu_inventory(block + "\n\n" + block) == [
        {"vendor_id": "Example", "model name": "Test CPU", "flags": ["a", "z"]}
    ]


def test_environment_allowlist_redacts_unknown_values(monkeypatch):
    monkeypatch.setenv("ATEN_CPU_CAPABILITY", "private-token")
    monkeypatch.setenv("OMP_NUM_THREADS", "/private/path")
    monkeypatch.setenv("UNRELATED_SECRET", "very-private")
    encoded = json.dumps(probe.environment_controls())
    assert "private" not in encoded and "redacted" in encoded


def test_study_order_and_requested_controls(monkeypatch):
    monkeypatch.setenv("ATEN_CPU_CAPABILITY", "avx2")
    calls = mock_workers(monkeypatch)
    result = probe.run_capture(dispatch_study=True)
    assert result["all_checks_passed"]
    assert [s["condition"] for s in result["samples"]] == ["A", "B", "B", "A"]
    assert [c[1]["env"].get("ATEN_CPU_CAPABILITY") for c in calls] == [
        None,
        "default",
        "default",
        None,
    ]
    assert "private/path" not in json.dumps(result)
    for command, kwargs in calls:
        assert command[:4] == [sys.executable, "-I", "-m", "pytorch_lab.numerical_probe"]
        assert kwargs["check"] and kwargs["timeout"] == 300
        assert all(kwargs["env"][k] == "1" for k in probe.THREAD_ENV)


def test_retained_digest_tampering_rejected(monkeypatch):
    mock_workers(monkeypatch, lambda v, _: v["wine_result"].update(limits=[]))
    result = probe.run_capture()
    assert not result["all_checks_passed"]
    assert result["capture_error"]["error_type"] == "ValueError"


def test_requested_dispatch_metadata_checked(monkeypatch):
    def mutate(value, _):
        value["cpu_runtime"]["environment_controls"]["ATEN_CPU_CAPABILITY"] = "avx2"

    mock_workers(monkeypatch, mutate)
    result = probe.run_capture()
    assert not result["all_checks_passed"]
    assert result["capture_error"]["error_type"] == "ValueError"


def test_changed_affinity_fails_control_gate(monkeypatch):
    def mutate(value, index):
        if index == 1:
            value["cpu_runtime"]["cpu_affinity"] = [0]

    mock_workers(monkeypatch, mutate)
    assert not probe.run_capture(dispatch_study=True)["all_checks_passed"]


def test_ineffective_intervention_fails_control_gate(monkeypatch):
    mock_workers(monkeypatch, lambda v, _: v["cpu_runtime"].update(torch_cpu_capability="DEFAULT"))
    result = probe.run_capture(dispatch_study=True)
    assert not result["all_checks_passed"]
    assert result["capture_checks"]["dispatch_intervention_observed"] is False


def test_within_condition_hash_failure_not_relaxed(monkeypatch):
    def mutate(value, index):
        if index == 1:
            value["wine_result"]["preprocessing"]["mean"][0] += 0.01
            value["scientific_result_sha256"] = result_digest(value["wine_result"])

    mock_workers(monkeypatch, mutate)
    result = probe.run_capture()
    assert not result["all_checks_passed"]
    assert not result["capture_checks"]["strict_scientific_hash_equal_within_each_condition"]


def test_dispatch_outcome_separate_from_control_gate(monkeypatch):
    def mutate(value, index):
        if index in (1, 2):
            value["wine_result"]["preprocessing"]["mean"][0] += 0.01
            value["scientific_result_sha256"] = result_digest(value["wine_result"])

    mock_workers(monkeypatch, mutate)
    result = probe.run_capture(dispatch_study=True)
    assert result["all_checks_passed"]
    assert not result["comparisons_to_first_sample"][0]["approximate"]["passed"]
    assert "separately reported" in result["all_checks_scope"]


def test_without_dispatch_does_not_mutate_input():
    value = sample()["cpu_runtime"]
    original = copy.deepcopy(value)
    reduced = probe._without_dispatch(value)
    assert value == original and "torch_cpu_capability" not in reduced


def cli(*args):
    # This parent supervises sequential workers, each with its own deadline.
    # Keep the child CLI's limits unchanged; reserve the sum plus startup/report time.
    arguments = list(map(str, args))
    repeats = int(arguments[arguments.index("--repeats") + 1]) if "--repeats" in arguments else 2
    child_timeout = (
        float(arguments[arguments.index("--timeout") + 1]) if "--timeout" in arguments else 300
    )
    if "--dispatch-study" in arguments:
        repeats = 4
    outer_timeout = repeats * child_timeout + 60
    return subprocess.run(
        [sys.executable, "-I", "-m", "pytorch_lab.numerical_probe", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=outer_timeout,
    )


def test_real_fresh_processes_retain_all_science(tmp_path):
    output = tmp_path / "evidence.json"
    run = cli("--epochs", 1, "--repeats", 2, "--output", output)
    assert run.returncode == 0, run.stderr
    report = json.loads(output.read_text())
    assert report == json.loads(run.stdout) and report["all_checks_passed"]
    for item in report["samples"]:
        assert result_digest(item["wine_result"]) == item["scientific_result_sha256"]
        assert item["cpu_runtime"]["float_info"]["mant_dig"] == 53
        assert item["cpu_runtime"]["cpu_models_and_features"]
        assert item["cpu_runtime"]["native_build_versions"]["mkl"]
    assert report["comparisons_to_first_sample"][0]["exact"]["passed"]


def test_no_overwrite_or_dangling_symlink(tmp_path):
    for path in [tmp_path / "existing", tmp_path / "symlink"]:
        if path.name == "existing":
            path.write_text("keep")
        else:
            path.symlink_to(tmp_path / "absent")
        run = cli("--output", path)
        assert run.returncode == 2
    assert (tmp_path / "existing").read_text() == "keep"
    assert (tmp_path / "symlink").is_symlink()


def test_cli_timeout_and_invalid_args_leave_no_success(tmp_path):
    output = tmp_path / "no-success.json"
    assert cli("--timeout", 0.001, "--output", output).returncode == 2
    partial = json.loads(output.read_text())
    assert (
        not partial["all_checks_passed"]
        and partial["capture_error"]["error_type"] == "TimeoutExpired"
    )
    assert not partial["samples"]
    absent = tmp_path / "invalid.json"
    assert cli("--epochs", 0, "--output", absent).returncode == 2
    assert not absent.exists()


def test_failure_after_success_keeps_complete_prior_capture(monkeypatch):
    calls = mock_workers(monkeypatch)
    first = probe.subprocess.run

    def fail_second(*args, **kwargs):
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("private command", 1, stderr="private secret")
        return first(*args, **kwargs)

    monkeypatch.setattr(probe.subprocess, "run", fail_second)
    result = probe.run_capture()
    assert not result["all_checks_passed"]
    assert len(result["samples"]) == 1
    assert result["samples"][0]["wine_result"] == sample()["wine_result"]
    assert result["capture_error"]["sample_index"] == 1
    assert "private" not in json.dumps(result)
