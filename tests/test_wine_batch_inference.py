"""Primary partial-batch and installed-delivery contract tests."""

import copy
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from pytorch_lab import wine_batch_inference as api
from pytorch_lab import wine_batch_inference_cli as cli
from pytorch_lab import wine_model_bundle as bundle


@pytest.fixture(scope="module")
def prepared():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        value, request, _, _ = bundle.train_bundle()
        yield value, request
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize("size", [1, 16, 107, 1024])
def test_vectorized_reference_and_no_mutation(prepared, size):
    value, original = prepared
    request = {
        "features": original["features"],
        "rows": [original["rows"][i % 107] for i in range(size)],
    }
    saved = copy.deepcopy(request)
    engine = api.WineBatchPredictor(value)
    with patch.object(engine.model, "forward", wraps=engine.model.forward) as forward:
        response = engine.predict(request)
    assert forward.call_count == 1
    reference = bundle.predict(value, request)
    actual = [item["prediction"]["logits"] for item in response["items"]]
    torch.testing.assert_close(
        torch.tensor(actual, dtype=torch.float64),
        torch.tensor(reference["logits"], dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    assert request == saved
    assert response["summary"] == {"requested": size, "succeeded": size, "failed": 0}
    assert [item["index"] for item in response["items"]] == list(range(size))
    timings = response["timings_seconds"]
    assert all(value >= 0 for value in timings.values())
    assert timings["in_memory_total"] >= sum(
        value for key, value in timings.items() if key != "in_memory_total"
    )


@pytest.mark.parametrize(
    "bad", [True, None, "1", float("nan"), float("inf"), -float("inf"), 1e7, 10**400]
)
def test_per_row_number_errors(prepared, bad):
    value, original = prepared
    row = list(original["rows"][0])
    row[3] = bad
    response = api.WineBatchPredictor(value).predict(
        {"features": original["features"], "rows": [row, original["rows"][1]]}
    )
    assert response["items"][0] == {"index": 0, "ok": False, "error": {"code": "invalid_number"}}
    assert response["items"][1]["ok"] is True


@pytest.mark.parametrize("row", [None, {}, "bad", [], [1] * 12, [1] * 14])
def test_all_invalid_skips_model(prepared, row):
    engine = api.WineBatchPredictor(prepared[0])
    with patch.object(engine.model, "forward", side_effect=AssertionError("must not run")):
        response = engine.predict({"features": prepared[1]["features"], "rows": [row]})
    assert response["summary"] == {"requested": 1, "succeeded": 0, "failed": 1}
    assert response["items"][0]["error"]["code"] == "invalid_shape"
    assert response["timings_seconds"]["model_forward"] == 0


@pytest.mark.parametrize("mutation", ["extra", "missing", "order", "empty", "too_many", "not_list"])
def test_global_request_errors(prepared, mutation):
    request = copy.deepcopy(prepared[1])
    if mutation == "extra":
        request["secret"] = 1
    elif mutation == "missing":
        del request["features"]
    elif mutation == "order":
        request["features"].reverse()
    elif mutation == "empty":
        request["rows"] = []
    elif mutation == "too_many":
        request["rows"] = [request["rows"][0]] * 1025
    else:
        request["rows"] = {}
    with pytest.raises(ValueError):
        api.WineBatchPredictor(prepared[0]).predict(request)


def test_model_loaded_once_and_bad_bundle_rejected(prepared):
    with patch.object(api, "load_payload", wraps=api.load_payload) as load:
        engine = api.WineBatchPredictor(prepared[0])
        engine.predict(prepared[1])
        engine.predict(prepared[1])
    assert load.call_count == 1
    invalid = copy.deepcopy(prepared[0])
    invalid["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        api.WineBatchPredictor(invalid)


def test_cli_partial_success_no_overwrite_and_bad_json(prepared, tmp_path):
    model, request = prepared
    request = copy.deepcopy(request)
    request["rows"][2] = [False] * 13
    bp, rp, output = [tmp_path / name for name in ["model.json", "request.json", "out.json"]]
    bundle.write_new(bp, model)
    bundle.write_new(rp, request)
    env = dict(os.environ, PYTHONPATH=str(Path(api.__file__).parents[1]))
    command = [
        sys.executable,
        "-m",
        "pytorch_lab.wine_batch_inference_cli",
        "--bundle",
        str(bp),
        "--request",
        str(rp),
        "--output",
        str(output),
    ]
    first = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, timeout=60)
    assert first.returncode == 0, first.stderr
    raw = output.read_bytes()
    response = json.loads(raw)
    assert response["summary"] == {"requested": 107, "succeeded": 106, "failed": 1}
    assert response["items"][2]["error"]["code"] == "invalid_number"
    assert subprocess.run(command, env=env, capture_output=True, timeout=60).returncode == 2
    assert output.read_bytes() == raw
    rp.write_text('{"features":[], "features":[]}')
    command[-1] = str(tmp_path / "must-not-exist.json")
    assert subprocess.run(command, env=env, capture_output=True, timeout=60).returncode == 2
    assert not Path(command[-1]).exists()


def test_thread_setting_restored_after_failure(tmp_path):
    before = torch.get_num_threads()
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--bundle",
                str(tmp_path / "missing"),
                "--request",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "out"),
            ]
        )
    assert torch.get_num_threads() == before


def test_installed_and_source_ci_integration():
    root = Path(__file__).parents[1]
    commands = runpy.run_path(str(root / "scripts/check_installed.py"))["COMMANDS"]
    assert len(commands) == 26
    assert commands[-1] == ("wine_batch_inference", "pytorch_lab.wine_batch_inference_cli", [])
    assert (
        "python -m pytorch_lab.wine_batch_inference_cli "
        "--output results/local-ci-wine-batch-inference.json"
        in (root / ".github/workflows/tests.yml").read_text()
    )
