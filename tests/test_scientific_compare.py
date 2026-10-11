"""Original boundary tests using full published Wine evidence, without training."""

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pytorch_lab import scientific_compare as compare

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results/2026-10-09-wine-cpu.json"
PUBLISHED_DIGEST = "b7ab96864bf8265cc5ec54c7318cb65bc78cbb0404174ef868a803ce3eb75fc1"


@pytest.fixture
def result():
    return compare.load_result(PUBLISHED)


def change(result, path, value):
    changed = copy.deepcopy(result)
    parent = changed
    for component in path[:-1]:
        parent = parent[component]
    parent[path[-1]] = value
    return changed


def forward(value, steps):
    for _ in range(steps):
        value = math.nextafter(value, math.inf)
    return value


def run_cli(left, right, output, *options):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytorch_lab.scientific_compare",
            str(left),
            str(right),
            "--output",
            str(output),
            *options,
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        check=False,
        timeout=20,
    )


def save(tmp_path, result, name="right.json"):
    path = tmp_path / name
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def test_published_fixture_exact_digest_and_numeric_counts(result):
    before = copy.deepcopy(result)
    report = compare.compare_results(result, result)
    assert report["exact"]["passed"] and report["approximate"]["passed"]
    assert report["exact"]["left_result_sha256"] == PUBLISHED_DIGEST
    assert report["exact"]["mismatch_count"] == 0
    assert report["statistics"]["compared_numeric_leaves"] == 6609
    assert report["statistics"]["compared_float_leaves"] == 3769
    assert report["statistics"]["continuous_float_leaves"] == 3514
    assert result == before


def test_excluded_root_metadata_is_validated_but_not_a_gate(result):
    right = copy.deepcopy(result)
    right["created_at_utc"] = "2026-10-10T16:00:00+00:00"
    right["environment"]["python"] = "3.12.15"
    right["environment"]["torch"] = "2.14.1+cpu.other"
    report = compare.compare_results(result, right)
    assert report["exact"]["passed"] and report["approximate"]["passed"]
    assert not report["excluded_metadata"]["identical"]
    assert report["excluded_metadata"]["right"]["environment"] == right["environment"]


def test_invalid_schema_identity_and_runtime_are_rejected(result):
    for path, value in (
        (("schema_version",), 2),
        (("dataset", "sha256"), "0" * 64),
        (("environment", "device"), "cuda"),
        (("environment", "dtype"), "float32"),
        (("environment", "threads"), 2),
        (("environment", "python"), ""),
        (("created_at_utc",), "not a timestamp"),
    ):
        with pytest.raises(ValueError):
            compare.compare_results(result, change(result, path, value))


def test_empty_missing_and_failed_check_results_cannot_pass(result):
    missing = copy.deepcopy(result)
    del missing["models"]["linear"]["runs"][0]["test"]["probabilities"]
    for invalid in (
        {},
        {"schema_version": 1},
        missing,
        change(result, ("checks", "equal_update_and_sample_budgets"), False),
    ):
        with pytest.raises(ValueError):
            compare.compare_results(invalid, invalid)


def test_nonfinite_numbers_rejected_even_in_excluded_metadata(result):
    for value in (math.nan, math.inf, -math.inf):
        invalid = change(result, ("preprocessing", "mean", 0), value)
        with pytest.raises(ValueError, match="nonfinite"):
            compare.compare_results(invalid, invalid)
    result["environment"]["extension"] = math.nan
    with pytest.raises(ValueError, match="nonfinite"):
        compare.compare_results(result, result)


def test_boolean_never_substitutes_for_integer_or_float(result):
    for path in (
        ("schema_version",),
        ("protocol", "config", "epochs"),
        ("environment", "threads"),
        ("preprocessing", "mean", 0),
    ):
        invalid = change(result, path, True)
        with pytest.raises(ValueError, match="type"):
            compare.compare_results(invalid, invalid)
    result["extra"] = {"flag": True, "count": 1}
    report = compare.compare_results(result, result)
    assert report["statistics"]["compared_numeric_leaves"] == 6610


def test_float_optimizer_configuration_is_exact(result):
    value = result["protocol"]["config"]["learning_rate"]
    report = compare.compare_results(
        result, change(result, ("protocol", "config", "learning_rate"), forward(value, 1))
    )
    assert not report["approximate"]["passed"]
    assert report["approximate"]["exact_required_mismatch_count"] == 1
    assert report["approximate"]["continuous_floats_passed"]


def test_count_derived_floats_and_their_summaries_are_exact(result):
    paths = (
        ("training_prior_baseline", "priors", 0),
        ("training_prior_baseline", "test", "accuracy"),
        ("training_prior_baseline", "test", "macro_f1"),
        ("training_prior_baseline", "test", "per_class", 0, "precision"),
        ("training_prior_baseline", "test", "per_class", 0, "recall"),
        ("training_prior_baseline", "test", "per_class", 0, "f1"),
        ("models", "linear", "test", "accuracy", "mean"),
        ("models", "linear", "test", "macro_f1", "sample_std"),
        ("models", "linear", "test", "macro_f1", "values", 0),
    )
    for path in paths:
        value = result
        for key in path:
            value = value[key]
        report = compare.compare_results(result, change(result, path, forward(value, 1)))
        assert not report["approximate"]["passed"], path
        assert report["exact"]["mismatch_categories"] == {"exact_float": 1}


def test_predictions_confusion_and_selection_changes_are_exact_failures(result):
    for path, value in (
        (("models", "linear", "runs", 0, "test", "predictions", 0), 2),
        (("models", "linear", "runs", 0, "test", "confusion_matrix", 0, 0), 11),
        (("protocol", "selected_architecture"), "linear"),
    ):
        report = compare.compare_results(result, change(result, path, value))
        assert not report["exact"]["passed"] and not report["approximate"]["passed"]
        assert report["exact"]["mismatch_categories"] == {"exact_scalar": 1}


def test_unknown_paths_and_nested_environment_names_default_exact(result):
    result["extension"] = {"cross_entropy": 1.0, "environment": {"score": 1.0}}
    for path in (("extension", "cross_entropy"), ("extension", "environment", "score")):
        report = compare.compare_results(result, change(result, path, forward(1.0, 1)))
        assert report["approximate"]["exact_required_mismatch_count"] == 1


def test_key_presence_list_length_order_and_types_are_exact(result):
    result["extension"] = {"values": [1, 2], "flag": True}
    variants = (
        (change(result, ("extension",), {"values": [1, 2]}), "key_presence"),
        (change(result, ("extension", "values"), [1]), "list_length"),
        (change(result, ("extension", "values"), [2, 1]), "exact_scalar"),
        (change(result, ("extension", "values", 0), 1.0), "type"),
        (change(result, ("extension", "flag"), 1), "type"),
    )
    for right, category in variants:
        report = compare.compare_results(result, right)
        assert not report["approximate"]["passed"]
        assert category in report["exact"]["mismatch_categories"]


def test_every_continuous_path_family_has_a_tight_allowance(result):
    paths = (
        ("preprocessing", "mean", 0),
        ("preprocessing", "scale", 0),
        ("training_prior_baseline", "test", "cross_entropy"),
        ("training_prior_baseline", "test", "probabilities", 0, 0),
        ("training_prior_baseline", "test", "errors", 0, "true_probability"),
        ("training_prior_baseline", "test", "errors", 0, "predicted_probability"),
        ("models", "linear", "runs", 0, "history", 0, "online_cross_entropy"),
        ("models", "linear", "runs", 0, "history", 0, "final_train_cross_entropy"),
        ("models", "linear", "runs", 0, "train", "cross_entropy"),
        ("models", "linear", "runs", 0, "validation", "probabilities", 0, 0),
        ("models", "linear", "runs", 0, "test", "errors", 0, "true_probability"),
        ("models", "mlp16", "validation", "cross_entropy", "mean"),
        ("models", "mlp16", "test", "cross_entropy", "sample_std"),
        ("models", "mlp16", "test", "cross_entropy", "values", 0),
    )
    for path in paths:
        value = result
        for key in path:
            value = value[key]
        report = compare.compare_results(result, change(result, path, forward(value, 8)))
        assert not report["exact"]["passed"], path
        assert report["approximate"]["passed"], path
        assert report["statistics"]["max_ulp_distance"]["value"] == 8


def test_eight_steps_pass_but_nine_steps_fail_at_same_value(result):
    value = result["preprocessing"]["mean"][0]
    for steps in (8, 9):
        report = compare.compare_results(
            result, change(result, ("preprocessing", "mean", 0), forward(value, steps))
        )
        assert report["approximate"]["passed"] is (steps == 8)
        assert report["approximate"]["continuous_ulp_failure_count"] == int(steps == 9)
        assert report["statistics"]["max_ulp_distance"]["value"] == steps
        assert report["statistics"]["max_ulp_distance"]["path"] == '$["preprocessing"]["mean"][0]'


def test_no_absolute_floor_even_for_subnormal_differences(result):
    result["preprocessing"]["mean"][0] = 0.0
    report = compare.compare_results(
        result, change(result, ("preprocessing", "mean", 0), forward(0.0, 9))
    )
    assert report["statistics"]["max_absolute_difference"]["value"] < 1e-300
    assert not report["approximate"]["passed"]
    assert report["policy"]["absolute_tolerance"] == 0.0


def test_signed_zero_numeric_equality_only_applies_to_continuous_paths(result):
    for path, continuous in (
        (("preprocessing", "mean", 0), True),
        (("training_prior_baseline", "test", "per_class", 0, "precision"), False),
        (("protocol", "config", "momentum"), False),
    ):
        left = change(result, path, 0.0)
        right = change(result, path, -0.0)
        report = compare.compare_results(left, right)
        assert not report["exact"]["passed"]
        assert report["approximate"]["passed"] is continuous
        assert report["approximate"]["exact_required_mismatch_count"] == int(not continuous)
        assert report["exact"]["mismatch_categories"] == {"signed_zero": 1}
        assert report["statistics"]["float_value_difference_count"] == 0
        assert report["statistics"]["float_bit_difference_count"] == 1
        assert report["statistics"]["signed_zero_difference_count"] == 1
        assert report["statistics"]["max_ulp_distance"]["value"] == 0


def test_ulp_distance_handles_negatives_zero_subnormals_and_binade_boundary():
    tiny = math.ulp(0.0)
    for left, right, expected in (
        (-0.0, 0.0, 0),
        (-tiny, 0.0, 1),
        (-tiny, tiny, 2),
        (-1.0, math.nextafter(-1.0, 0.0), 1),
        (math.nextafter(1.0, 0.0), math.nextafter(1.0, 2.0), 2),
        (-2.0, forward(-2.0, 8), 8),
    ):
        assert compare.ulp_distance(left, right) == expected
        assert compare.ulp_distance(right, left) == expected
    for invalid in (1, True, math.inf, math.nan):
        with pytest.raises(ValueError):
            compare.ulp_distance(invalid, 0.0)


def test_finite_extremes_never_emit_infinite_absolute_difference(result):
    result["preprocessing"]["mean"][0] = -sys.float_info.max
    report = compare.compare_results(
        result, change(result, ("preprocessing", "mean", 0), sys.float_info.max)
    )
    maximum = report["statistics"]["max_absolute_difference"]
    assert maximum["overflow"] and isinstance(maximum["value"], str)
    assert maximum["path"] == '$["preprocessing"]["mean"][0]'
    assert report["statistics"]["max_ulp_distance"]["value"] == 2 * 0x7FEFFFFFFFFFFFFF
    assert not report["approximate"]["passed"]
    json.dumps(report, allow_nan=False)


def test_mismatch_examples_are_bounded_but_totals_complete(result):
    right = copy.deepcopy(result)
    for index, value in enumerate(right["preprocessing"]["mean"]):
        right["preprocessing"]["mean"][index] = forward(value, 9)
    for index, value in enumerate(right["preprocessing"]["scale"]):
        right["preprocessing"]["scale"][index] = forward(value, 9)
    report = compare.compare_results(result, right)
    assert report["exact"]["mismatch_count"] == 26
    assert report["approximate"]["continuous_ulp_failure_count"] == 26
    assert len(report["exact"]["examples"]) == len(report["approximate"]["examples"]) == 20
    assert all(item["ulp_distance"] == 9 for item in report["approximate"]["examples"])


def test_loader_rejects_duplicate_keys_bad_json_and_overflow_numbers(tmp_path):
    for payload in (
        '{"schema_version":1,"schema_version":1}',
        '{"a":{"b":1,"b":2}}',
        '{"a":NaN}',
        '{"a":Infinity}',
        '{"a":1e999}',
        "{oops",
    ):
        path = tmp_path / "bad.json"
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError):
            compare.load_result(path)


def test_loader_and_python_api_have_bounded_input_limits(tmp_path, result, monkeypatch):
    monkeypatch.setattr(compare, "MAX_JSON_BYTES", 10)
    with pytest.raises(ValueError, match="byte limit"):
        compare.load_result(save(tmp_path, result))
    value = 1
    for _ in range(34):
        value = [value]
    result["extension"] = value
    with pytest.raises(ValueError, match="structural limits"):
        compare.compare_results(result, result)


def test_cli_success_persists_same_report_and_has_no_torch_import(tmp_path):
    output = tmp_path / "report.json"
    completed = run_cli(PUBLISHED, PUBLISHED, output)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == json.loads(output.read_text())
    assert json.loads(completed.stdout)["gate"] == {"mode": "approximate", "passed": True}
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pytorch_lab.scientific_compare; assert 'torch' not in sys.modules",
        ],
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0


def test_cli_require_exact_fails_after_retaining_approximate_pass(tmp_path, result):
    value = result["preprocessing"]["mean"][0]
    right = save(tmp_path, change(result, ("preprocessing", "mean", 0), forward(value, 1)))
    output = tmp_path / "report.json"
    completed = run_cli(PUBLISHED, right, output, "--require-exact")
    assert completed.returncode == 1
    report = json.loads(output.read_text())
    assert report["gate"] == {"mode": "exact", "passed": False}
    assert report["approximate"]["passed"] and not report["exact"]["passed"]


def test_cli_numerical_failure_report_is_retained(tmp_path, result):
    value = result["preprocessing"]["mean"][0]
    right = save(tmp_path, change(result, ("preprocessing", "mean", 0), forward(value, 9)))
    output = tmp_path / "report.json"
    completed = run_cli(PUBLISHED, right, output)
    assert completed.returncode == 1
    assert json.loads(output.read_text())["approximate"]["continuous_ulp_failure_count"] == 1


def test_cli_malformed_is_exit_two_and_does_not_disclose_path_or_input(tmp_path):
    malformed = tmp_path / "private-identity.json"
    malformed.write_text('{"SECRET_UNTRUSTED_TEXT": NaN}', encoding="utf-8")
    output = tmp_path / "report.json"
    completed = run_cli(malformed, PUBLISHED, output)
    assert completed.returncode == 2
    assert not output.exists() and completed.stdout == ""
    assert (
        "private-identity" not in completed.stderr
        and "SECRET_UNTRUSTED_TEXT" not in completed.stderr
    )


def test_cli_never_overwrites_existing_report_or_input(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("original evidence", encoding="utf-8")
    completed = run_cli(PUBLISHED, PUBLISHED, output)
    assert completed.returncode == 2
    assert output.read_text() == "original evidence"
    source = tmp_path / "source.json"
    source.write_bytes(PUBLISHED.read_bytes())
    before = source.read_bytes()
    assert run_cli(source, source, source).returncode == 2
    assert source.read_bytes() == before


def test_cli_rejects_live_and_dangling_output_symlinks(tmp_path):
    for name, exists in (("live", True), ("dangling", False)):
        target = tmp_path / f"{name}-target.json"
        if exists:
            target.write_text("original evidence", encoding="utf-8")
        output = tmp_path / f"{name}.json"
        output.symlink_to(target)
        assert run_cli(PUBLISHED, PUBLISHED, output).returncode == 2
        assert output.is_symlink()
        assert target.read_text() == "original evidence" if exists else not target.exists()
