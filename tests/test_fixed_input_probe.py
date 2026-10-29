"""Fixed bytes, explicit boundaries, independent gates and bounded failure evidence."""

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pytorch_lab import fixed_input_probe as probe


@pytest.fixture(scope="module")
def scientific():
    torch.set_num_threads(1)
    return probe.evaluate_fixture()


def changed(scientific, stage="exp", index=0, steps=1):
    result = copy.deepcopy(scientific)
    record = next(x for x in result["stages"] if x["name"] == stage)
    record["values"][index] = math.nextafter(record["values"][index], math.inf, steps=steps)
    record["sha256"] = probe._bits(record["values"], record["dtype"])
    return result


def sample(scientific, label="A", capability=None):
    return {
        "scientific": copy.deepcopy(scientific),
        "scientific_sha256": probe.result_digest(scientific),
        "environment": {
            "python": "3.12.test",
            "torch": "2.14.1+cpu",
            "libc": ["glibc", "test"],
            "device": "cpu",
            "dtype": "float64",
            "torch_cuda_build": None,
            "intraop_threads": 1,
            "interop_threads": 1,
            "python_isolated_mode": True,
        },
        "package_files_sha256": probe.package_file_hashes(),
        "cpu_runtime": {
            **probe.runtime_details(),
            "torch_cpu_capability": capability or ("AVX512" if label == "A" else "DEFAULT"),
            "environment_controls": {
                "ATEN_CPU_CAPABILITY": None if label == "A" else "default",
                **dict.fromkeys(probe.THREAD_ENV, "1"),
            },
        },
    }


def fake_run(monkeypatch, outputs):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        value = outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(stdout=json.dumps(value), stderr="")

    monkeypatch.setattr(probe.subprocess, "run", run)
    return calls


def test_fixture_bytes_signed_zero_and_stage_shapes(scientific):
    payload = probe.files("pytorch_lab").joinpath("data/fixed_logits_v1.json").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == probe.FIXTURE_SHA256
    assert math.copysign(1, scientific["inputs"]["logits"]["values"][28]) == -1
    assert scientific["inputs"] == probe.expected_inputs()
    assert [s["name"] for s in scientific["stages"]] == list(probe.STAGES)
    assert all(scientific["controls"].values())
    assert sum(len(s["values"]) for s in scientific["stages"]) == 465
    assert probe.compare_scientific(scientific, scientific)["exact_passed"]


def test_fixture_corruption_rejected(monkeypatch):
    monkeypatch.setattr(
        probe,
        "files",
        lambda _: SimpleNamespace(joinpath=lambda _: SimpleNamespace(read_bytes=lambda: b"{}")),
    )
    with pytest.raises(ValueError, match="input bytes"):
        probe.load_fixture()


@pytest.mark.parametrize("steps,passed", [(8, True), (9, False)])
def test_unchanged_eight_step_boundary(scientific, steps, passed):
    comparison = probe.compare_scientific(scientific, changed(scientific, steps=steps))
    assert not comparison["exact_passed"]
    assert comparison["diagnostic_passed"] is passed
    assert comparison["policy"]["max_ulps"] == 8
    assert comparison["first_unequal_explicit_chain_boundary"] == "exp"
    stage = next(s for s in comparison["stages"] if s["name"] == "exp")
    assert stage["max_ulp_distance"] == steps


def test_independent_branch_is_not_reported_as_chain_boundary(scientific):
    comparison = probe.compare_scientific(scientific, changed(scientific, "log_softmax"))
    assert not comparison["exact_passed"]
    assert comparison["first_unequal_explicit_chain_boundary"] is None


def test_signed_zero_output_is_exact_difference(scientific):
    right = copy.deepcopy(scientific)
    record = right["stages"][2]
    index = record["values"].index(0.0)
    record["values"][index] = -0.0
    record["sha256"] = probe._bits(record["values"], "float64")
    result = probe.compare_scientific(scientific, right)
    assert not result["exact_passed"] and result["diagnostic_passed"]


def test_predictions_must_be_exact(scientific):
    right = copy.deepcopy(scientific)
    record = right["stages"][-1]
    record["values"][0] = (record["values"][0] + 1) % 3
    record["sha256"] = probe._bits(record["values"], "int64")
    result = probe.compare_scientific(scientific, right)
    assert not result["diagnostic_passed"]
    assert result["stages"][-1]["exact_integer_failure_count"] == 1


@pytest.mark.parametrize(
    "case",
    [
        "input",
        "labels",
        "order",
        "shape",
        "dtype",
        "count",
        "nan",
        "bool",
        "digest",
        "schema",
        "extra",
        "controls",
        "identity",
    ],
)
def test_malformed_or_changed_evidence_rejected(scientific, case):
    value = copy.deepcopy(scientific)
    item = value["stages"][3]
    if case in {"input", "labels"}:
        key = "logits" if case == "input" else "labels"
        item = value["inputs"][key]
        item["values"][0] += 1
        item["sha256"] = probe._bits(item["values"], item["dtype"])
    elif case == "order":
        value["stages"][1:3] = reversed(value["stages"][1:3])
    elif case == "shape":
        item["shape"] = [48]
    elif case == "dtype":
        item["dtype"] = "float32"
    elif case == "count":
        item["values"].pop()
    elif case == "nan":
        item["values"][0] = float("nan")
    elif case == "bool":
        item["values"][0] = True
    elif case == "digest":
        item["sha256"] = "0" * 64
    elif case == "schema":
        value["schema_version"] = True
    elif case == "extra":
        value["extra"] = 1
    elif case == "controls":
        value["controls"]["equal_logit_softmax_symmetric"] = False
    elif case == "identity":
        value = changed(scientific, "identity")
    with pytest.raises(ValueError):
        probe.compare_scientific(scientific, value)


def test_equal_logit_control_recomputed_not_trusted(scientific):
    right = changed(scientific, "softmax", index=3)
    assert right["controls"]["equal_logit_softmax_symmetric"]
    with pytest.raises(ValueError, match="controls"):
        probe.validate_scientific(right)


def test_gradient_matches_analytic_formula(scientific):
    values = {s["name"]: s["values"] for s in scientific["stages"]}
    labels = scientific["inputs"]["labels"]["values"]
    expected = [(p - float(i % 3 == labels[i // 3])) / 16 for i, p in enumerate(values["softmax"])]
    assert values["logits_gradient"] == pytest.approx(expected, abs=1e-16, rel=1e-12)
    # This sanity check is not the cross-condition acceptance gate or a ground-truth oracle.
    assert values["cross_entropy_mean"][0] == pytest.approx(sum(values["cross_entropy_none"]) / 16)


def test_dispatch_validity_separate_from_numeric_failure(monkeypatch, scientific):
    a = sample(scientific)
    b = sample(changed(scientific, steps=9), "B")
    outputs = [a, b, b, a]
    calls = fake_run(monkeypatch, outputs)
    result = probe.run_capture(dispatch_study=True)
    assert result["all_checks_passed"]
    assert not result["comparisons_to_first_sample"][0]["diagnostic_passed"]
    assert result["comparisons_to_first_sample"][2]["exact_passed"]
    assert [c[1]["env"].get("ATEN_CPU_CAPABILITY") for c in calls] == [
        None,
        "default",
        "default",
        None,
    ]
    assert all(c[0][1:3] == ["-I", "-m"] for c in calls)


def test_same_condition_drift_fails_even_within_eight(monkeypatch, scientific):
    fake_run(monkeypatch, [sample(scientific), sample(changed(scientific))])
    result = probe.run_capture()
    assert not result["all_checks_passed"]
    assert result["comparisons_to_first_sample"][0]["diagnostic_passed"]
    assert not result["capture_checks"]["within_condition_strict_repeats"]


def test_ineffective_intervention_fails(monkeypatch, scientific):
    a, b = sample(scientific, capability="DEFAULT"), sample(scientific, "B")
    fake_run(monkeypatch, [a, b, b, a])
    result = probe.run_capture(dispatch_study=True)
    assert not result["all_checks_passed"]
    assert not result["capture_checks"]["dispatch_intervention_observed"]


@pytest.mark.parametrize("case", ["runtime", "package", "thread", "digest", "request"])
def test_mutated_worker_controls_fail(monkeypatch, scientific, case):
    a, b = sample(scientific), sample(scientific)
    if case == "runtime":
        b["cpu_runtime"]["changed"] = True
    elif case == "package":
        b["package_files_sha256"]["fake.py"] = "0" * 64
    elif case == "thread":
        b["environment"]["intraop_threads"] = 2
    elif case == "digest":
        b["scientific_sha256"] = "0" * 64
    elif case == "request":
        b["cpu_runtime"]["environment_controls"]["ATEN_CPU_CAPABILITY"] = "default"
    fake_run(monkeypatch, [a, b])
    result = probe.run_capture()
    assert not result["all_checks_passed"]
    assert result["samples"][0]["scientific"] == scientific


def test_partial_worker_failure_retains_evidence_and_redacts(monkeypatch, scientific):
    fake_run(monkeypatch, [sample(scientific), subprocess.CalledProcessError(7, "secret/path")])
    result = probe.run_capture()
    assert not result["all_checks_passed"]
    assert len(result["samples"]) == 1
    assert result["capture_error"]["sample_index"] == 1
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize(
    "repeats,timeout", [(1, 60), (5, 60), (True, 60), (2, 0), (2, 301), (2, float("nan"))]
)
def test_budget_rejected_before_work(monkeypatch, repeats, timeout):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not start")

    monkeypatch.setattr(probe.subprocess, "run", forbidden)
    with pytest.raises(ValueError):
        probe.run_capture(repeats, timeout)


def cli(*args):
    return subprocess.run(
        [sys.executable, "-I", "-m", "pytorch_lab.fixed_input_probe", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_real_cli_repeat_and_no_overwrite(tmp_path):
    output = tmp_path / "fixed.json"
    run = cli("--output", output)
    assert run.returncode == 0, run.stderr
    result = json.loads(output.read_text())
    assert result == json.loads(run.stdout) and result["all_checks_passed"]
    assert result["comparisons_to_first_sample"][0]["exact_passed"]
    before = output.read_bytes()
    assert cli("--output", output).returncode == 2
    assert output.read_bytes() == before
    link = tmp_path / "dangling.json"
    link.symlink_to(tmp_path / "absent.json")
    assert cli("--output", link).returncode == 2
    assert not (tmp_path / "absent.json").exists()


def test_real_cli_timeout_retains_failed_report(tmp_path):
    output = tmp_path / "timeout.json"
    run = cli("--timeout", 0.001, "--output", output)
    assert run.returncode == 1
    result = json.loads(output.read_text())
    assert not result["all_checks_passed"] and result["samples"] == []
    assert result["capture_error"]["error_type"] == "TimeoutExpired"


def test_real_cli_invalid_repeats_no_output(tmp_path):
    output = tmp_path / "invalid.json"
    assert cli("--repeats", 1, "--output", output).returncode == 2
    assert not output.exists()


def test_eighth_installed_cli_and_retention():
    import runpy

    checker = runpy.run_path(str(Path(__file__).parents[1] / "scripts/check_installed.py"))
    # Fixed-input stays eighth; accumulation/recovery are ninth/tenth.
    command = checker["COMMANDS"][7]
    assert len(checker["COMMANDS"]) == 12
    assert command[0:2] == ("fixed_input", "pytorch_lab.fixed_input_probe")
    assert int(command[2][1]) * int(command[2][3]) < 300


@pytest.mark.parametrize(
    "case", ["envelope", "environment", "cpu", "capability", "cpu_controls", "package"]
)
def test_malformed_worker_retains_completed_sample(monkeypatch, scientific, case):
    a, b = sample(scientific), sample(scientific)
    if case == "envelope":
        b = []
    elif case == "environment":
        b["environment"] = []
    elif case == "cpu":
        b["cpu_runtime"] = []
    elif case == "capability":
        b["cpu_runtime"].pop("torch_cpu_capability")
    elif case == "cpu_controls":
        b["cpu_runtime"]["environment_controls"] = []
    elif case == "package":
        b["package_files_sha256"] = []
    fake_run(monkeypatch, [a, b])
    report = probe.run_capture()
    assert len(report["samples"]) == 1 and not report["all_checks_passed"]
    assert report["capture_error"]["sample_index"] == 1


def test_duplicate_json_keys_rejected(monkeypatch):
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout='{"scientific":{},"scientific":{}}', stderr=""),
    )
    report = probe.run_capture()
    assert report["samples"] == [] and not report["all_checks_passed"]
    assert report["capture_error"]["error_type"] == "ValueError"


def test_overflow_number_in_metadata_retains_prior_evidence(monkeypatch, scientific):
    a = sample(scientific)
    b = sample(scientific)
    b["cpu_runtime"]["float_info"]["epsilon"] = "overflow_placeholder"
    outputs = [json.dumps(a), json.dumps(b).replace('"overflow_placeholder"', "1e999")]
    monkeypatch.setattr(
        probe.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=outputs.pop(0), stderr="")
    )
    result = probe.run_capture()
    assert len(result["samples"]) == 1 and not result["all_checks_passed"]
    assert result["capture_error"]["error_type"] == "ValueError"
    json.dumps(result, allow_nan=False)
