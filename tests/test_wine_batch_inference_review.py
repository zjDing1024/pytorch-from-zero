"""Independent partial-batch review against equations and fresh CLI processes."""

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import pytest
import torch

from pytorch_lab import wine_batch_inference as batch
from pytorch_lab import wine_model_bundle as bundle_api


@pytest.fixture(scope="module")
def fixture():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        bundle, request, _, _ = bundle_api.train_bundle()
        yield bundle, request
    finally:
        torch.set_num_threads(old)


def oracle(bundle, rows):
    payload = bundle["payload"]
    state = {k: torch.tensor(v, dtype=torch.float64) for k, v in payload["state"].items()}
    x = torch.tensor(rows, dtype=torch.float64)
    x = (x - torch.tensor(payload["mean"], dtype=torch.float64)) / torch.tensor(
        payload["scale"], dtype=torch.float64
    )
    hidden = (x @ state["weights.0"].T + state["biases.0"]).relu()
    return hidden @ state["weights.1"].T + state["biases.1"]


def check_timing(response):
    timings = response["timings_seconds"]
    assert set(timings) == {
        "validation",
        "preprocessing",
        "model_forward",
        "postprocessing",
        "in_memory_total",
    }
    assert all(math.isfinite(v) and v >= 0 for v in timings.values())
    stages = sum(v for k, v in timings.items() if k != "in_memory_total")
    assert timings["in_memory_total"] >= stages - 1e-12


def test_mixed_isolation_single_load_vectorization_and_preservation(fixture):
    bundle, source = fixture
    request = {
        "features": source["features"],
        "rows": [
            source["rows"][0],
            [],
            [True] * 13,
            source["rows"][7],
            [10**400] * 13,
            [None] * 13,
            [float("inf")] * 13,
            source["rows"][0],
        ],
    }
    before_request = copy.deepcopy(request)
    before_bundle = copy.deepcopy(bundle)
    rng = torch.get_rng_state().clone()
    with patch.object(batch, "load_payload", wraps=batch.load_payload) as loader:
        engine = batch.WineBatchPredictor(bundle)
        with patch.object(engine.model, "forward", wraps=engine.model.forward) as forward:
            response = engine.predict(request)
            assert forward.call_count == 1
            assert tuple(forward.call_args.args[0].shape) == (3, 13)
            assert not forward.call_args.args[0].requires_grad
            engine.predict({"features": source["features"], "rows": [[], [False] * 13]})
            assert forward.call_count == 1
        assert loader.call_count == 1
    assert request == before_request
    assert bundle == before_bundle
    assert torch.equal(rng, torch.get_rng_state())
    assert response["summary"] == {"requested": 8, "succeeded": 3, "failed": 5}
    assert [item["index"] for item in response["items"]] == list(range(8))
    valid_indices = [0, 3, 7]
    expected = oracle(bundle, [request["rows"][i] for i in valid_indices])
    for j, i in enumerate(valid_indices):
        item = response["items"][i]
        assert item["ok"] is True and "error" not in item
        prediction = item["prediction"]
        torch.testing.assert_close(
            torch.tensor(prediction["logits"], dtype=torch.float64),
            expected[j],
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            torch.tensor(prediction["probabilities"], dtype=torch.float64),
            expected[j].softmax(0),
            rtol=1e-12,
            atol=1e-12,
        )
        assert prediction["class_index"] == expected[j].argmax().item()
        assert prediction["class"] == f"class_{prediction['class_index']}"
    for i in [1, 2, 4, 5, 6]:
        assert response["items"][i] == {
            "index": i,
            "ok": False,
            "error": {"code": "invalid_shape" if i == 1 else "invalid_number"},
        }
    check_timing(response)


@pytest.mark.parametrize(
    "row,code",
    [
        (None, "invalid_shape"),
        ({}, "invalid_shape"),
        ((0,) * 13, "invalid_shape"),
        ([0] * 12, "invalid_shape"),
        ([0] * 14, "invalid_shape"),
        ([float("nan")] * 13, "invalid_number"),
        (["0"] * 13, "invalid_number"),
        ([1_000_001] * 13, "invalid_number"),
        ([-1_000_001] * 13, "invalid_number"),
    ],
)
def test_all_invalid_does_not_preprocess_or_forward(fixture, row, code):
    bundle, source = fixture
    engine = batch.WineBatchPredictor(bundle)
    with (
        patch.object(engine.model, "forward", side_effect=AssertionError("forward called")),
        patch.object(
            type(engine.scaler), "transform", side_effect=AssertionError("preprocess called")
        ),
    ):
        response = engine.predict({"features": source["features"], "rows": [row]})
    assert response["items"] == [{"index": 0, "ok": False, "error": {"code": code}}]
    for stage in ["preprocessing", "model_forward", "postprocessing"]:
        assert response["timings_seconds"][stage] == 0
    check_timing(response)


@pytest.mark.parametrize("fault", ["missing", "extra", "features", "empty", "oversize", "tuple"])
def test_bad_envelope_aborts_before_forward(fixture, fault):
    bundle, source = fixture
    request = copy.deepcopy(source)
    if fault == "missing":
        del request["features"]
    elif fault == "extra":
        request["extra"] = None
    elif fault == "features":
        request["features"].reverse()
    elif fault == "empty":
        request["rows"] = []
    elif fault == "oversize":
        request["rows"] = [source["rows"][0]] * 1025
    else:
        request["rows"] = tuple(request["rows"])
    engine = batch.WineBatchPredictor(bundle)
    with patch.object(engine.model, "forward", side_effect=AssertionError("forward called")):
        with pytest.raises(ValueError):
            engine.predict(request)


def test_bad_bundle_aborts_and_runtime_fault_is_global(fixture):
    bundle, source = fixture
    corrupt = copy.deepcopy(bundle)
    corrupt["payload"]["mean"][0] += 1
    with pytest.raises(ValueError, match="integrity"):
        batch.WineBatchPredictor(corrupt)
    engine = batch.WineBatchPredictor(bundle)
    with patch.object(engine.model, "forward", return_value=torch.full((107, 3), float("nan"))):
        with pytest.raises(ValueError, match="nonfinite"):
            engine.predict(source)


@pytest.mark.parametrize("size", [1, 16, 107, 1024])
def test_batch_boundaries_numeric_oracle(fixture, size):
    bundle, source = fixture
    rows = [source["rows"][i % 107] for i in range(size)]
    response = batch.WineBatchPredictor(bundle).predict(
        {"features": source["features"], "rows": rows}
    )
    logits = torch.tensor(
        [i["prediction"]["logits"] for i in response["items"]], dtype=torch.float64
    )
    torch.testing.assert_close(logits, oracle(bundle, rows), rtol=1e-12, atol=1e-12)
    assert response["summary"] == {"requested": size, "succeeded": size, "failed": 0}
    check_timing(response)


def run_cli(tmp_path, bundle, request, output):
    env = dict(os.environ, PYTHONPATH=str(Path(batch.__file__).parents[1]))
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytorch_lab.wine_batch_inference_cli",
            "--bundle",
            str(bundle),
            "--request",
            str(request),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_real_cli_partial_response_timing_and_no_clobber(fixture, tmp_path):
    bundle, source = fixture
    bp, rp, op = [tmp_path / name for name in ["bundle.json", "request.json", "output.json"]]
    bundle_api.write_new(bp, bundle)
    bundle_api.write_new(rp, {"features": source["features"], "rows": [source["rows"][0], []]})
    start = perf_counter()
    result = run_cli(tmp_path, bp, rp, op)
    wall = perf_counter() - start
    assert result.returncode == 0, result.stderr
    original = op.read_bytes()
    response = json.loads(original)
    assert response["summary"] == {"requested": 2, "succeeded": 1, "failed": 1}
    check_timing(response)
    times = response["file_timings_seconds"]
    assert all(math.isfinite(v) and 0 <= v <= wall for v in times.values())
    assert times["prediction"] >= response["timings_seconds"]["in_memory_total"]
    assert times["before_output_encoding"] == pytest.approx(
        sum(
            times[k]
            for k in [
                "bundle_read_parse",
                "bundle_validate_load",
                "request_read_parse",
                "prediction",
            ]
        ),
        abs=1e-12,
    )
    assert run_cli(tmp_path, bp, rp, op).returncode != 0
    assert op.read_bytes() == original
    link = tmp_path / "dangling.json"
    link.symlink_to(tmp_path / "absent.json")
    assert run_cli(tmp_path, bp, rp, link).returncode != 0
    assert link.is_symlink() and not link.exists()


@pytest.mark.parametrize(
    "raw",
    [
        b'{"features":[],"features":[],"rows":[]}',
        b'{"rows":[NaN]}',
        b'{"rows":[Infinity]}',
        b" " * 1_000_001,
        b"[" * 2000,
        b"\xff",
    ],
    ids=["duplicate", "nan", "infinity", "oversize", "deep", "invalid_utf8"],
)
def test_real_cli_strict_json_fault_has_no_output(fixture, tmp_path, raw):
    bp, rp, op = [tmp_path / name for name in ["bundle.json", "request.json", "output.json"]]
    bundle_api.write_new(bp, fixture[0])
    rp.write_bytes(raw)
    result = run_cli(tmp_path, bp, rp, op)
    assert result.returncode == 2
    assert not op.exists()


def test_numeric_limit_inclusive_and_inference_mode(fixture):
    bundle, source = fixture
    engine = batch.WineBatchPredictor(bundle)
    seen = []

    def check_mode(module, args):
        seen.append(args[0].shape)
        assert not module.training
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()

    handle = engine.model.register_forward_pre_hook(check_mode)
    try:
        response = engine.predict(
            {"features": source["features"], "rows": [[1_000_000] * 13, [-1_000_000] * 13]}
        )
    finally:
        handle.remove()
    assert seen == [torch.Size([2, 13])]
    assert response["summary"] == {"requested": 2, "succeeded": 2, "failed": 0}
    assert all(math.isfinite(v) for item in response["items"] for v in item["prediction"]["logits"])
