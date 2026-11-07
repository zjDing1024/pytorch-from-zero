"""Bounded JSON inference bundle for one predeclared Wine model, not a service."""

import hashlib
import json
import math
from importlib.resources import files
from pathlib import Path

import torch
from torch.nn import functional as F

from pytorch_lab import wine

MAX_BYTES = 1_000_000
MAX_ROWS = 1024
LIMIT = 1_000_000
CONFIG = {"architecture": "mlp16", "seed": 42, "epochs": 20, "arm": "constant"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def seal(payload):
    return {"payload": payload, "sha256": hashlib.sha256(canonical(payload)).hexdigest()}


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("unexpected object fields")


def _matrix(values, rows, cols):
    if type(values) is not list or len(values) != rows:
        raise ValueError("invalid rows")
    for row in values:
        if type(row) is not list or len(row) != cols:
            raise ValueError("invalid columns")
        if any(type(v) not in (int, float) or abs(v) > LIMIT or not math.isfinite(v) for v in row):
            raise ValueError("invalid numeric value")
    return torch.tensor(values, dtype=torch.float64)


def _vector(values, length):
    return _matrix([values], 1, length)[0]


def strict_json(path):
    # Bound the read itself, not only a preceding stat (which can race).
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("file too large")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON constant")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid JSON") from exc


def write_new(path, value):
    encoded = canonical(value) + b"\n"
    if len(encoded) > MAX_BYTES:
        raise ValueError("file too large")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)


def train_bundle():
    """Train seed42 without reading validation scores or selecting another seed."""
    x, y = wine.load_wine()
    split = json.loads(files("pytorch_lab").joinpath("data/wine_split.json").read_text())
    if split["data_sha256"] != wine.DATA_SHA256:
        raise ValueError("data digest mismatch")
    expected = wine.build_split(y)
    if split["row_ids"] != expected:
        raise ValueError("split identity mismatch")
    ids = expected["train"]
    scaler = wine.Standardizer.fit(x[ids])
    train = scaler.transform(x[ids])
    model = wine.WineClassifier("mlp16", 42)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(42)
    updates = visits = 0
    for _ in range(20):
        order = torch.randperm(107, generator=generator)
        for offset in range(0, 107, 16):
            batch = order[offset : offset + 16]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train[batch]), y[ids][batch])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            optimizer.step()
            updates += 1
            visits += len(batch)
    model.eval()
    payload = {
        "schema": "wine-model-bundle-v1",
        "config": CONFIG.copy(),
        "features": list(wine.FEATURE_NAMES),
        "classes": ["class_0", "class_1", "class_2"],
        "dtype": "float64",
        "raw_units": "unchanged vendored UCI Wine measurements",
        "data_sha256": wine.DATA_SHA256,
        "fit_rows": ids,
        "mean": scaler.mean.tolist(),
        "scale": scaler.scale.tolist(),
        "state": {k: v.detach().tolist() for k, v in model.state_dict().items()},
    }
    bundle = seal(payload)
    load_payload(bundle)  # Validate producer output through the consumer contract.
    request = {"features": list(wine.FEATURE_NAMES), "rows": x[ids].tolist()}
    with torch.no_grad():
        expected_logits = model(train).tolist()
    return bundle, request, expected_logits, {"updates": updates, "training_visits": visits}


def load_payload(bundle):
    _keys(bundle, ("payload", "sha256"))
    payload = bundle["payload"]
    if type(bundle["sha256"]) is not str or seal(payload)["sha256"] != bundle["sha256"]:
        raise ValueError("integrity mismatch")
    _keys(
        payload,
        (
            "schema",
            "config",
            "features",
            "classes",
            "dtype",
            "raw_units",
            "data_sha256",
            "fit_rows",
            "mean",
            "scale",
            "state",
        ),
    )
    fixed = {
        "schema": "wine-model-bundle-v1",
        "config": CONFIG,
        "features": list(wine.FEATURE_NAMES),
        "classes": ["class_0", "class_1", "class_2"],
        "dtype": "float64",
        "raw_units": "unchanged vendored UCI Wine measurements",
        "data_sha256": wine.DATA_SHA256,
    }
    # Canonical comparison rejects bool/int aliasing and 42.0 instead of 42.
    if any(canonical(payload[k]) != canonical(v) for k, v in fixed.items()):
        raise ValueError("unsupported metadata")
    manifest = json.loads(files("pytorch_lab").joinpath("data/wine_split.json").read_text())
    if canonical(payload["fit_rows"]) != canonical(manifest["row_ids"]["train"]):
        raise ValueError("training row identity mismatch")
    mean, scale = _vector(payload["mean"], 13), _vector(payload["scale"], 13)
    if not (scale >= 1e-12).all():
        raise ValueError("scale too small")
    model = wine.WineClassifier("mlp16", 42)
    state = payload["state"]
    template = model.state_dict()
    _keys(state, template)
    tensors = {}
    for key, tensor in template.items():
        tensors[key] = (
            _vector(state[key], tensor.numel())
            if tensor.ndim == 1
            else _matrix(state[key], *tensor.shape)
        )
    model.load_state_dict(tensors, strict=True)
    model.eval()
    return model, wine.Standardizer(mean, scale)


def predict(bundle, request):
    model, scaler = load_payload(bundle)
    _keys(request, ("features", "rows"))
    if request["features"] != list(wine.FEATURE_NAMES):
        raise ValueError("feature order mismatch")
    rows = request["rows"]
    if type(rows) is not list or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("batch size must be in [1, 1024]")
    raw = _matrix(rows, len(rows), 13)
    with torch.no_grad():
        logits = model(scaler.transform(raw))
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite output")
        probabilities = torch.softmax(logits, dim=1)
        labels = logits.argmax(dim=1)
    return {
        "logits": logits.tolist(),
        "probabilities": probabilities.tolist(),
        "class_indices": labels.tolist(),
        "classes": [f"class_{i}" for i in labels.tolist()],
    }
