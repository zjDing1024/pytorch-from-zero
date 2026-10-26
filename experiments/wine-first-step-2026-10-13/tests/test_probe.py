"""Contract and math checks; cross-dispatch findings are never pass/fail oracles."""

import copy
import json
import math
import os
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import probe


@pytest.fixture(scope="session")
def fixture():
    return probe.load_fixture()[0]


@pytest.fixture(scope="session")
def sample():
    env = dict(os.environ, **{x: "1" for x in probe.THREAD_ENV})
    env.pop("ATEN_CPU_CAPABILITY", None)
    result = subprocess.run(
        [sys.executable, "-I", str(probe.ROOT / "probe.py"), "--worker"],
        capture_output=True,
        check=True,
        timeout=60,
        env=env,
    )
    return probe.parse(result.stdout)


def repair_hash(sample):
    for item in sample["scientific"]["stages"]:
        item["sha256_be"] = probe.sha(struct.pack(f">{len(item['values'])}d", *item["values"]))
    sample["scientific_sha256"] = probe.sha(probe.canonical(sample["scientific"]))


def altered(sample, steps):
    result = copy.deepcopy(sample)
    old = result["scientific"]["stages"][3]["values"][0]
    for _ in range(steps):
        old = math.nextafter(old, math.inf)
    result["scientific"]["stages"][3]["values"][0] = old
    repair_hash(result)
    return result


def test_fixture_train_only_and_bit_round_trip(fixture):
    raw = probe.parse((probe.ROOT / "data/wine_split.json").read_bytes())
    assert fixture["train_row_ids"] == raw["row_ids"]["train"]
    assert not set(fixture["batch_row_ids"]) & set(raw["row_ids"]["test"])
    assert not set(fixture["batch_row_ids"]) & set(raw["row_ids"]["validation"])
    assert len(fixture["batch_row_ids"]) == len(set(fixture["batch_row_ids"])) == 16
    for item in fixture["inputs"].values():
        assert probe.record(probe.tensor_from(item)) == item


def test_capture_contract_and_exact_self(sample):
    probe.validate_sample(sample)
    assert len(sample["scientific"]["stages"]) == 19
    assert sum(len(x["values"]) for x in sample["scientific"]["stages"]) == 1946
    result = probe.compare(sample, sample)
    assert result["strict_equal"] and result["within_8_ulps"]
    assert result["first_unequal_forward_boundary"] is None
    assert result["first_unequal_backward_boundary"] is None


def test_math_chain_and_first_update(fixture, sample):
    x = probe.tensor_from(fixture["inputs"]["x"])
    labels = probe.tensor_from(fixture["inputs"]["y"])
    params = {k: probe.tensor_from(fixture["inputs"][k]) for k in probe.PARAM_SHAPES}
    stages = {
        s["name"]: probe.tensor_from({k: v for k, v in s.items() if k != "name"})
        for s in sample["scientific"]["stages"]
    }
    # Independent analytical relationships, with an explicit sanity tolerance.
    logits_gradient = stages["logits"].softmax(dim=1)
    logits_gradient[torch.arange(16), labels] -= 1
    logits_gradient /= 16
    hidden_gradient = logits_gradient @ params["weight2"]
    pre_gradient = hidden_gradient * (stages["hidden_pre"] > 0)
    expected = {
        "logits_gradient": logits_gradient,
        "hidden_relu_gradient": hidden_gradient,
        "hidden_pre_gradient": pre_gradient,
        "weight1_gradient": pre_gradient.T @ x,
        "bias1_gradient": pre_gradient.sum(dim=0),
        "weight2_gradient": logits_gradient.T @ stages["hidden_relu"],
        "bias2_gradient": logits_gradient.sum(dim=0),
    }
    for name, value in expected.items():
        torch.testing.assert_close(stages[name], value, rtol=1e-13, atol=1e-15)
    # For the first SGD step, momentum is adjusted gradient, not mu*gradient.
    for name, initial in params.items():
        buffer = stages[f"{name}_gradient"] + probe.OPTIMIZER["weight_decay"] * initial
        updated = initial - probe.OPTIMIZER["lr"] * buffer
        torch.testing.assert_close(stages[f"{name}_momentum"], buffer, rtol=1e-13, atol=1e-15)
        torch.testing.assert_close(stages[f"{name}_updated"], updated, rtol=1e-13, atol=1e-15)
    assert torch.equal(stages["hidden_relu"], stages["hidden_pre"].clamp_min(0))
    assert math.isfinite(stages["loss"].item())


@pytest.mark.parametrize("steps,passes", [(8, True), (9, False)])
def test_ulp_budget_boundary(sample, steps, passes):
    result = probe.compare(sample, altered(sample, steps))
    assert not result["strict_equal"]
    assert result["within_8_ulps"] is passes
    assert result["first_unequal_forward_boundary"] == "loss"
    assert result["stages"][3]["max_ulps"] == steps


def test_signed_zero_and_subnormal():
    tiny = math.nextafter(0.0, 1.0)
    assert probe.ulps(-0.0, 0.0) == 0
    assert probe.ulps(-tiny, tiny) == 2
    assert probe.ulps(-1.0, math.nextafter(-1.0, -math.inf)) == 1
    assert probe.ulps(1.0, math.nextafter(1.0, math.inf)) == 1
    assert (
        probe.record(torch.tensor(-0.0, dtype=torch.float64))["sha256_be"]
        != probe.record(torch.tensor(0.0, dtype=torch.float64))["sha256_be"]
    )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 1, True])
def test_ulp_rejects_invalid(value):
    with pytest.raises(ValueError):
        probe.ulps(value, 1.0)


@pytest.mark.parametrize(
    "mutation",
    [
        "stage_order",
        "stage_shape",
        "bool_shape",
        "dtype",
        "int_value",
        "nan_value",
        "bad_tensor_hash",
        "bad_scientific_hash",
        "input_hash",
        "missing_stage",
        "unknown_stage_field",
        "bad_check",
        "bad_source_hash",
        "nonfinite_runtime",
    ],
)
def test_corrupt_sample_rejected(sample, mutation):
    value = copy.deepcopy(sample)
    stages = value["scientific"]["stages"]
    if mutation == "stage_order":
        stages.reverse()
    elif mutation == "stage_shape":
        stages[0]["shape"] = [256]
    elif mutation == "bool_shape":
        stages[3]["shape"] = [True]
    elif mutation == "dtype":
        stages[0]["dtype"] = "float32"
    elif mutation == "int_value":
        stages[0]["values"][0] = 1
    elif mutation == "nan_value":
        stages[0]["values"][0] = math.nan
    elif mutation == "bad_tensor_hash":
        stages[0]["sha256_be"] = "0" * 64
    elif mutation == "bad_scientific_hash":
        value["scientific_sha256"] = "0" * 64
    elif mutation == "input_hash":
        value["scientific"]["input_hashes"]["x"] = "0" * 64
    elif mutation == "missing_stage":
        stages.pop()
    elif mutation == "unknown_stage_field":
        stages[0]["extra"] = 1
    elif mutation == "bad_check":
        value["checks"]["inputs_unchanged"] = 1
    elif mutation == "bad_source_hash":
        value["runtime"]["source_sha256"] = "0" * 64
    elif mutation == "nonfinite_runtime":
        value["runtime"]["extra"] = math.inf
    with pytest.raises(ValueError):
        probe.validate_sample(value)


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}', b'{"x":1e999}', b"NaN"])
def test_invalid_json(payload):
    with pytest.raises(ValueError):
        probe.parse(payload)


def test_size_and_depth_rejected():
    with pytest.raises(ValueError):
        probe.parse(b" " * (probe.MAX_BYTES + 1))
    with pytest.raises(ValueError):
        probe.parse(("[" * 25 + "0" + "]" * 25).encode())


def mock_sequence(sample, b_sample=None):
    a = copy.deepcopy(sample)
    a["runtime"]["capability"] = "AVX512"
    a["runtime"]["environment"]["ATEN_CPU_CAPABILITY"] = None
    b = copy.deepcopy(sample if b_sample is None else b_sample)
    b["runtime"]["capability"] = "DEFAULT"
    b["runtime"]["environment"]["ATEN_CPU_CAPABILITY"] = "default"
    return [a, b, copy.deepcopy(b), copy.deepcopy(a)]


def test_negative_result_is_valid_capture(sample):
    result = probe.summarize(mock_sequence(sample), ["A", "B", "B", "A"])
    assert result["capture_valid"]
    assert all(c["strict_equal"] for c in result["comparisons_to_A1"])


def test_numeric_failure_retained_as_valid_capture(sample):
    sequence = mock_sequence(sample, altered(sample, 9))
    result = probe.summarize(sequence, ["A", "B", "B", "A"])
    assert result["capture_valid"]
    assert not result["comparisons_to_A1"][0]["within_8_ulps"]


@pytest.mark.parametrize(
    "problem", ["same_condition_drift", "ineffective", "changed_host", "wrong_request"]
)
def test_capture_controls_reject(sample, problem):
    sequence = mock_sequence(sample)
    if problem == "same_condition_drift":
        sequence[3] = altered(sequence[3], 1)
    elif problem == "ineffective":
        for item in sequence:
            item["runtime"]["capability"] = "DEFAULT"
    elif problem == "changed_host":
        sequence[1]["runtime"]["cpu_models_and_features"][0]["model name"] = "other"
    else:
        sequence[1]["runtime"]["environment"]["ATEN_CPU_CAPABILITY"] = None
    assert not probe.summarize(sequence, ["A", "B", "B", "A"])["capture_valid"]


def test_partial_timeout_retains_completed_sample(sample):
    count = 0

    def runner(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise subprocess.TimeoutExpired("worker", 0.01, stderr="private-string")
        return SimpleNamespace(stdout=probe.canonical(sample))

    result = probe.capture(0.01, runner)
    assert len(result["samples"]) == 1
    assert not result["capture_valid"]
    assert result["failure"]["category"] == "TimeoutExpired"
    assert "private-string" not in json.dumps(result)


def test_malformed_worker_retains_previous(sample):
    queue = [probe.canonical(sample), b"{}"]
    result = probe.capture(1, lambda *args, **kwargs: SimpleNamespace(stdout=queue.pop(0)))
    assert len(result["samples"]) == 1
    assert result["failure"]["category"] == "ValueError"


@pytest.mark.parametrize("timeout", [0, -1, 61, math.nan, math.inf, True])
def test_bad_timeout(timeout):
    with pytest.raises(ValueError):
        probe.capture(timeout)


def test_cli_no_overwrite_and_invalid_timeout(tmp_path):
    target = tmp_path / "result.json"
    target.write_text("keep")
    command = [sys.executable, "-I", str(probe.ROOT / "probe.py"), "--output", str(target)]
    assert subprocess.run(command, capture_output=True, timeout=10).returncode == 2
    assert target.read_text() == "keep"
    target.unlink()
    target.symlink_to(tmp_path / "absent")
    assert subprocess.run(command, capture_output=True, timeout=10).returncode == 2
    assert target.is_symlink()
    target.unlink()
    assert (
        subprocess.run(command + ["--timeout", "nan"], capture_output=True, timeout=10).returncode
        == 2
    )
    assert not target.exists()


def test_fixture_file_corruption_rejected(tmp_path, monkeypatch):
    for rel in ("docs/plan.json", "docs/freeze.json", "data/frozen_fixture.json"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((probe.ROOT / rel).read_bytes())
    target = tmp_path / "data/frozen_fixture.json"
    target.write_bytes(target.read_bytes() + b" ")
    # Even a rewritten manifest cannot bless changed fixture bytes.
    manifest = json.loads((tmp_path / "docs/freeze.json").read_text())
    manifest["fixture_sha256"] = probe.sha(target.read_bytes())
    (tmp_path / "docs/freeze.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        probe.load_fixture()


@pytest.mark.parametrize(
    "problem",
    [
        "bool_threads",
        "extra",
        "missing",
        "affinity_bool",
        "affinity_duplicate",
        "bad_hash",
        "missing_package_hash",
        "bad_cpu",
        "bad_flags",
        "bad_environment",
        "unrecognized_environment",
        "wrong_version_type",
        "wrong_capability_type",
    ],
)
def test_runtime_schema_rejects(sample, problem):
    value = copy.deepcopy(sample)
    details = value["runtime"]
    if problem == "bool_threads":
        details["threads"] = True
    elif problem == "extra":
        details["extra"] = "unknown"
    elif problem == "missing":
        del details["kernel"]
    elif problem == "affinity_bool":
        details["affinity"] = [True]
    elif problem == "affinity_duplicate":
        details["affinity"] = [1, 1]
    elif problem == "bad_hash":
        details["torch_build_sha256"] = "z" * 64
    elif problem == "missing_package_hash":
        del details["torch_files_sha256"]["lib/libtorch_cpu.so"]
    elif problem == "bad_cpu":
        details["cpu_models_and_features"] = [{}]
    elif problem == "bad_flags":
        details["cpu_models_and_features"][0]["flags"] = "avx"
    elif problem == "bad_environment":
        details["environment"] = []
    elif problem == "unrecognized_environment":
        details["environment"]["MKL_CBWR"] = "redacted-unrecognized"
    elif problem == "wrong_version_type":
        details["python"] = 3
    else:
        details["capability"] = []
    with pytest.raises(ValueError):
        probe.validate_sample(value)


def test_extreme_nesting_retains_previous(sample):
    queue = [probe.canonical(sample), ("[" * 2000 + "0" + "]" * 2000).encode()]
    result = probe.capture(1, lambda *args, **kwargs: SimpleNamespace(stdout=queue.pop(0)))
    assert len(result["samples"]) == 1
    assert result["failure"]["category"] == "ValueError"
    assert not result["capture_valid"]
