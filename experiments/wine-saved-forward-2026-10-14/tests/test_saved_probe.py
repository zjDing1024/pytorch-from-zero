"""Scientific outcomes are observations; tests enforce method and failure boundaries."""

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

import saved_probe as probe


@pytest.fixture(scope="session")
def sample():
    env = dict(os.environ, **{name: "1" for name in probe.reference.THREAD_ENV})
    env.pop("ATEN_CPU_CAPABILITY", None)
    done = subprocess.run(
        [sys.executable, "-I", str(probe.ROOT / "saved_probe.py"), "--worker"],
        env=env,
        capture_output=True,
        check=True,
        timeout=60,
    )
    return probe.parse(done.stdout)


def rehash(sample):
    sample["scientific_sha256"] = probe.reference.sha(
        probe.reference.canonical(sample["scientific"])
    )


def repair_record(record):
    code = "d" if record["dtype"] == "float64" else "q"
    record["sha256_be"] = probe.reference.sha(
        struct.pack(f">{len(record['values'])}{code}", *record["values"])
    )


def test_reference_and_fixture_immutable():
    assert probe.protocol()["inputs"]["fixture_sha256"] == probe.reference.FROZEN_SHA


def test_real_worker_valid(sample):
    probe.validate_sample(sample)
    assert all(sample["checks"].values())
    assert sample["source_sha256"] == probe.reference.sha(
        (probe.ROOT / "saved_probe.py").read_bytes()
    )


def test_observer_preserves_original_before_and_after(sample):
    s = sample["scientific"]
    assert s["raw_before"] == s["observed"] == s["raw_after"]


def test_packed_objects_detached_not_original():
    original = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    observer = probe.SavedObserver()
    i, detached = observer.pack(original)
    assert detached is not original and detached.grad_fn is None and not detached.requires_grad
    assert i == 0
    assert observer.unpack((i, detached)) is detached
    assert observer.packs[0] == observer.unpacks[0]["tensor"]


def test_observer_does_not_hide_value_mutation_in_capture():
    value = torch.tensor([1.0], dtype=torch.float64)
    observer = probe.SavedObserver()
    packed = observer.pack(value)
    value.add_(1.0)
    observer.unpack(packed)
    assert observer.packs[0] != observer.unpacks[0]["tensor"]


def test_observer_rejects_unbounded_saves():
    observer = probe.SavedObserver()
    for _ in range(8):
        observer.pack(torch.tensor(1.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="count"):
        observer.pack(torch.tensor(1.0, dtype=torch.float64))


def test_upstream_is_exact_mean_nll():
    fixture, _ = probe.reference.load_fixture()
    labels = probe.reference.tensor_from(fixture["inputs"]["y"])
    gradient = probe.fixed_upstream(labels)
    assert gradient.shape == (16, 3) and gradient.dtype == torch.float64
    assert sum(x == -0.0625 for x in gradient.flatten().tolist()) == 16
    assert sum(x == 0.0 for x in gradient.flatten().tolist()) == 32
    assert gradient.sum().item() == -1.0
    for row, label in enumerate(labels.tolist()):
        assert gradient[row, label].item() == -0.0625


def test_manual_closed_form_is_sanity_not_strict_oracle(sample):
    s = sample["scientific"]
    logits = probe.reference.tensor_from(s["matched"]["logits"])
    upstream = probe.reference.tensor_from(s["matched"]["upstream"])
    expected = upstream - logits.softmax(dim=1) * upstream.sum(dim=1, keepdim=True)
    actual = probe.reference.tensor_from(s["matched"]["logits_gradient"])
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-17)


def test_scientific_hash_covers_observer_events(sample):
    changed = copy.deepcopy(sample)
    changed["scientific"]["saved_observer"]["unpacks"].reverse()
    with pytest.raises(ValueError, match="scientific hash"):
        probe.validate_sample(changed)


@pytest.mark.parametrize("target", ["source_sha256", "scientific_sha256"])
def test_rejects_changed_source_or_hash(sample, target):
    altered = copy.deepcopy(sample)
    altered[target] = "0" * 64
    with pytest.raises(ValueError):
        probe.validate_sample(altered)


@pytest.mark.parametrize(
    "kind",
    [
        "index_bool",
        "index_oob",
        "extra_event_key",
        "bad_dtype",
        "bad_shape",
        "nan",
        "changed_byte",
        "empty_packs",
    ],
)
def test_observer_schema_failures(sample, kind):
    altered = copy.deepcopy(sample)
    obs = altered["scientific"]["saved_observer"]
    if kind == "index_bool":
        obs["unpacks"][0]["index"] = True
    elif kind == "index_oob":
        obs["unpacks"][0]["index"] = 10
    elif kind == "extra_event_key":
        obs["unpacks"][0]["extra"] = 1
    elif kind == "bad_dtype":
        obs["packs"][0]["dtype"] = "float32"
    elif kind == "bad_shape":
        obs["packs"][0]["shape"] = [48]
    elif kind == "nan":
        obs["packs"][0]["values"][0] = float("nan")
    elif kind == "changed_byte":
        obs["packs"][0]["values"][0] += 0.1
    elif kind == "empty_packs":
        obs["packs"] = []
    with pytest.raises((ValueError, TypeError, IndexError)):
        probe.validate_sample(altered)


def test_signed_zero_upstream_tamper_rejected_even_after_rehash(sample):
    altered = copy.deepcopy(sample)
    upstream = altered["scientific"]["fixed_upstream"]
    upstream["values"][upstream["values"].index(0.0)] = -0.0
    repair_record(upstream)
    rehash(altered)
    with pytest.raises(ValueError, match="fixed upstream"):
        probe.validate_sample(altered)


@pytest.mark.parametrize("which", ["checks", "relations"])
def test_cached_assertions_cannot_lie(sample, which):
    altered = copy.deepcopy(sample)
    key = next(k for k, v in altered[which].items() if type(v) is bool)
    altered[which][key] = not altered[which][key]
    with pytest.raises(ValueError, match="inconsistent"):
        probe.validate_sample(altered)


def test_noninterference_failure_retained_as_invalid_control(sample):
    changed = copy.deepcopy(sample)
    record = changed["scientific"]["observed"]["stages"][3]
    record["values"][0] = math.nextafter(record["values"][0], math.inf)
    repair_record(record)
    rehash(changed)
    changed["checks"] = probe.local_checks(changed["scientific"])
    probe.validate_sample(changed)
    assert not changed["checks"]["observer_noninterference"]
    assert not probe.summarize([changed])["capture_valid"]


def test_upstream_relation_mismatch_is_retained_outcome(sample):
    changed = copy.deepcopy(sample)
    record = changed["scientific"]["decomposed"]["upstream"]
    record["values"][0] = math.nextafter(record["values"][0], -math.inf)
    repair_record(record)
    rehash(changed)
    changed["relations"] = probe.observed_relations(changed["scientific"])
    probe.validate_sample(changed)
    assert not changed["relations"]["decomposed_upstream_matches_fixed"]
    assert all(changed["checks"].values())


def float_record(value):
    return {
        "dtype": "float64",
        "shape": [],
        "values": [value],
        "sha256_be": probe.reference.sha(struct.pack(">d", value)),
    }


@pytest.mark.parametrize("steps,over", [(8, 0), (9, 1)])
def test_eight_nine_ulp_budget(steps, over):
    a, b = 1.0, 1.0
    for _ in range(steps):
        b = math.nextafter(b, math.inf)
    result = probe.compare_record(float_record(a), float_record(b))
    assert result["max_ulps"] == steps and result["over_budget"] == over
    assert not result["strict_equal"]


def test_signed_zero_strict_and_numeric_distinct():
    result = probe.compare_record(float_record(0.0), float_record(-0.0))
    assert result["max_ulps"] == 0 and result["changed_values"] == 1


def test_duplicate_and_nonfinite_and_deep_json_rejected():
    for payload in [b'{"a":1,"a":2}', b"[NaN]", b"[" * 30 + b"0" + b"]" * 30]:
        with pytest.raises(ValueError):
            probe.parse(payload)


@pytest.mark.parametrize("timeout", [0, -1, 61, float("nan"), True])
def test_invalid_timeout_rejected(timeout):
    with pytest.raises(ValueError, match="timeout"):
        probe.capture(timeout)


def test_partial_failure_retains_completed_worker(sample):
    calls = []

    def runner(command, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(stdout=json.dumps(sample).encode())
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    result = probe.capture(runner=runner)
    assert len(result["samples"]) == 1 and not result["capture_valid"]
    assert result["failure"]["condition"] == "B"
    assert result["failure"]["completed_workers"] == 1
    assert result["failure"]["category"] == "TimeoutExpired"
    assert "ATEN_CPU_CAPABILITY" not in calls[0]["env"]
    assert calls[1]["env"]["ATEN_CPU_CAPABILITY"] == "default"


def test_same_condition_drift_and_ineffective_intervention_fail(sample):
    result = probe.summarize([sample] * 4)
    assert not result["checks"]["effective_intervention"]
    drift = copy.deepcopy(sample)
    drift["scientific_sha256"] = "0" * 64
    assert not probe.summarize([sample, sample, sample, drift])["checks"]["A_strict_repeat"]


def test_real_cli_no_overwrite(tmp_path):
    output = tmp_path / "keep.json"
    output.write_bytes(b"original")
    done = subprocess.run(
        [sys.executable, "-I", str(probe.ROOT / "saved_probe.py"), "--output", str(output)],
        capture_output=True,
        timeout=10,
    )
    assert done.returncode == 2 and output.read_bytes() == b"original"


def test_real_cli_tiny_timeout_retains_failure(tmp_path):
    output = tmp_path / "timeout.json"
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            str(probe.ROOT / "saved_probe.py"),
            "--output",
            str(output),
            "--timeout",
            "0.001",
        ],
        capture_output=True,
        timeout=15,
    )
    assert done.returncode == 1
    result = json.loads(output.read_text())
    assert result["samples"] == [] and result["failure"]["category"] == "TimeoutExpired"


def test_comparison_includes_integer_saved_labels(sample):
    equal = probe.compare(sample, sample)
    assert equal["strict_equal"] and equal["within_8_ulps_and_exact_integers"]
    integers = [r for r in equal["records"].values() if "exact_integer_equal" in r]
    assert integers and all(r["exact_integer_equal"] for r in integers)


@pytest.mark.parametrize("replacement", [1, 1.0])
def test_boolean_relations_require_exact_type(sample, replacement):
    changed = copy.deepcopy(sample)
    changed["relations"]["decomposed_loss_matches_original"] = replacement
    with pytest.raises(ValueError, match="inconsistent relations"):
        probe.validate_sample(changed)


def test_saved_slot_indices_require_exact_type(sample):
    changed = copy.deepcopy(sample)
    changed["relations"]["saved_pack_indices_equal_decomposed_log_probs"] = [False, True]
    with pytest.raises(ValueError, match="inconsistent relations"):
        probe.validate_sample(changed)


@pytest.mark.parametrize("failure", ["nonzero", "timeout", "malformed"])
def test_failed_worker_exact_bytes_retained(failure):
    import base64

    stdout, stderr = b"partial-out\x00\xff", b"partial-error\xfe"

    def runner(command, **kwargs):
        if failure == "nonzero":
            raise subprocess.CalledProcessError(7, command, output=stdout, stderr=stderr)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                command, kwargs["timeout"], output=stdout, stderr=stderr
            )
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=0)

    result = probe.capture(runner=runner)
    assert not result["capture_valid"]
    for name, expected in (("stdout", stdout), ("stderr", stderr)):
        evidence = result["failure"][name]
        assert base64.b64decode(evidence["base64"]) == expected
        assert evidence["bytes"] == evidence["retained_bytes"] == len(expected)
        assert evidence["sha256"] == probe.reference.sha(expected) and not evidence["truncated"]
    assert result["failure"]["returncode"] == (
        7 if failure == "nonzero" else 0 if failure == "malformed" else None
    )


def test_failure_byte_limit_is_explicit():
    import base64

    evidence = probe.byte_evidence(b"123456", limit=4)
    assert evidence["bytes"] == 6 and evidence["retained_bytes"] == 4 and evidence["truncated"]
    assert evidence["sha256"] == probe.reference.sha(b"123456")
    assert base64.b64decode(evidence["base64"]) == b"1234"
