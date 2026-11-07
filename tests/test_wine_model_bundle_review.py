"""Independent inference-contract review: no pickle or held-out scoring."""

import copy
import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from pytorch_lab import wine
from pytorch_lab import wine_model_bundle as b
from pytorch_lab.wine_learning_curves import prepare_data, train_run


@pytest.fixture(scope="module")
def trained():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield b.train_bundle()
    finally:
        torch.set_num_threads(old)


def test_prior_selected_endpoint_and_training_statistics(trained):
    bundle, request, _, work = trained
    x, y = wine.load_wine()
    ids = bundle["payload"]["fit_rows"]
    assert work == {"updates": 140, "training_visits": 2140}
    assert request["rows"] == x[ids].tolist()
    assert bundle["payload"]["mean"] == x[ids].mean(0).tolist()
    assert bundle["payload"]["scale"] == x[ids].std(0, correction=0).tolist()
    tx, ty, vx, vy, _ = prepare_data()
    run = train_run(tx, ty, vx, vy, "mlp16", 42, "constant")
    from pytorch_lab.wine_learning_curves import digest_model

    model, _ = b.load_payload(bundle)
    assert digest_model(model) == run["curve"][-1]["parameters_sha256"]


def test_nontraining_features_do_not_affect_export(trained):
    bundle, request, expected, work = trained
    x, y = wine.load_wine()
    heldout = sorted(set(range(178)) - set(bundle["payload"]["fit_rows"]))
    x[heldout] = float("nan")
    with patch.object(wine, "load_wine", return_value=(x, y)):
        assert b.train_bundle() == (bundle, request, expected, work)


def test_direct_equation_predictions_and_rng(trained):
    bundle, request, expected, _ = trained
    state = torch.get_rng_state().clone()
    result = b.predict(bundle, request)
    p = bundle["payload"]
    x = torch.tensor(request["rows"], dtype=torch.float64)
    x = (x - torch.tensor(p["mean"], dtype=torch.float64)) / torch.tensor(
        p["scale"], dtype=torch.float64
    )
    s = {k: torch.tensor(v, dtype=torch.float64) for k, v in p["state"].items()}
    h = torch.addmm(s["biases.0"], x, s["weights.0"].T).relu()
    logits = torch.addmm(s["biases.1"], h, s["weights.1"].T)
    assert torch.equal(torch.get_rng_state(), state)
    assert result["logits"] == expected
    torch.testing.assert_close(
        torch.tensor(result["logits"], dtype=torch.float64), logits, rtol=0, atol=0
    )
    assert result["class_indices"] == logits.argmax(1).tolist()
    assert result["probabilities"] == logits.softmax(1).tolist()


@pytest.mark.parametrize(
    "mutation",
    ["seed_float", "feature_order", "class_order", "state_shape", "zero_scale", "nan_weight"],
)
def test_resealed_invalid_artifact_rejected(trained, mutation):
    p = copy.deepcopy(trained[0]["payload"])
    if mutation == "seed_float":
        p["config"]["seed"] = 42.0
    elif mutation == "feature_order":
        p["features"].reverse()
    elif mutation == "class_order":
        p["classes"].reverse()
    elif mutation == "state_shape":
        p["state"]["weights.0"].pop()
    elif mutation == "zero_scale":
        p["scale"][0] = 0
    else:
        p["state"]["weights.0"][0][0] = float("nan")
    with pytest.raises(ValueError):
        b.load_payload(b.seal(p))


@pytest.mark.parametrize(
    "mutation", ["bool", "nan", "empty", "wide", "reversed", "too_many", "range", "huge_int"]
)
def test_invalid_request_rejected(trained, mutation):
    bundle, original, _, _ = trained
    request = copy.deepcopy(original)
    if mutation == "bool":
        request["rows"][0][0] = True
    elif mutation == "nan":
        request["rows"][0][0] = float("nan")
    elif mutation == "empty":
        request["rows"] = []
    elif mutation == "wide":
        request["rows"][0].append(1)
    elif mutation == "reversed":
        request["features"].reverse()
    elif mutation == "too_many":
        request["rows"] = [request["rows"][0]] * 1025
    elif mutation == "huge_int":
        request["rows"][0][0] = 10**400
    else:
        request["rows"][0][0] = 1e7
    with pytest.raises(ValueError):
        b.predict(bundle, request)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b"[" * 2000, b" " * 1000001])
def test_bounded_strict_json(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        b.strict_json(path)


def test_fresh_process_no_clobber(trained, tmp_path):
    bundle, request, _, _ = trained
    bp, rp, out = (tmp_path / p for p in ("bundle.json", "request.json", "out.json"))
    b.write_new(bp, bundle)
    b.write_new(rp, request)
    cmd = [
        sys.executable,
        "-m",
        "pytorch_lab.wine_model_bundle_cli",
        "--bundle",
        str(bp),
        "--request",
        str(rp),
        "--output",
        str(out),
    ]
    env = dict(os.environ)
    from pathlib import Path

    env["PYTHONPATH"] = str(Path(b.__file__).parents[1])
    first = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert first.returncode == 0, first.stderr
    result = out.read_bytes()
    assert json.loads(result) == b.predict(bundle, request)
    second = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert second.returncode != 0
    assert out.read_bytes() == result


def test_real_optimizer_and_training_visits(trained):
    actual = {"updates": 0, "training_visits": 0}
    original_step = torch.optim.SGD.step
    original_ce = torch.nn.functional.cross_entropy

    def step(self, *args, **kwargs):
        actual["updates"] += 1
        return original_step(self, *args, **kwargs)

    def ce(logits, targets, *args, **kwargs):
        assert torch.is_grad_enabled()
        actual["training_visits"] += len(targets)
        return original_ce(logits, targets, *args, **kwargs)

    with (
        patch.object(torch.optim.SGD, "step", step),
        patch.object(torch.nn.functional, "cross_entropy", ce),
    ):
        _, _, _, reported = b.train_bundle()
    assert actual == reported == {"updates": 140, "training_visits": 2140}
