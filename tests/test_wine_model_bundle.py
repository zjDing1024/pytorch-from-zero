import copy
import json
import subprocess
import sys

import pytest
import torch

from pytorch_lab import wine_model_bundle as b


@pytest.fixture(scope="module")
def prepared():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    value = b.train_bundle()
    yield value
    torch.set_num_threads(threads)


def test_training_and_prediction(prepared):
    bundle, request, expected, work = prepared
    assert work == {"updates": 140, "training_visits": 2140}
    assert b.predict(bundle, request)["logits"] == expected
    assert len(expected) == 107


@pytest.mark.parametrize("value", [True, None, "1", float("nan"), float("inf"), 1e7])
def test_bad_numbers(prepared, value):
    bundle, request, _, _ = prepared
    request = copy.deepcopy(request)
    request["rows"][0][0] = value
    with pytest.raises(ValueError):
        b.predict(bundle, request)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "float32"),
        ("classes", [0, 1, 2]),
        ("features", []),
        ("schema", "v2"),
        ("fit_rows", []),
        ("scale", [0.0] * 13),
    ],
)
def test_signed_metadata_invalid(prepared, field, value):
    payload = copy.deepcopy(prepared[0]["payload"])
    payload[field] = value
    with pytest.raises(ValueError):
        b.load_payload(b.seal(payload))


def test_hash_and_shape(prepared):
    bad = copy.deepcopy(prepared[0])
    bad["payload"]["mean"][0] += 1
    with pytest.raises(ValueError):
        b.load_payload(bad)
    bad["payload"]["state"]["weights.0"] = [[0.0]]
    with pytest.raises(ValueError):
        b.load_payload(b.seal(bad["payload"]))


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', "invalid", "[" * 1500])
def test_parser(tmp_path, text):
    file = tmp_path / "bad.json"
    file.write_text(text)
    with pytest.raises(ValueError):
        b.strict_json(file)


def test_size_no_overwrite(tmp_path):
    file = tmp_path / "a.json"
    b.write_new(file, {"a": 1})
    with pytest.raises(FileExistsError):
        b.write_new(file, {})
    file.write_bytes(b" " * (b.MAX_BYTES + 1))
    with pytest.raises(ValueError):
        b.strict_json(file)


def test_fresh_process(tmp_path, prepared):
    bundle, request, expected, _ = prepared
    b.write_new(tmp_path / "model.json", bundle)
    b.write_new(tmp_path / "input.json", request)
    command = [
        sys.executable,
        "-m",
        "pytorch_lab.wine_model_bundle_cli",
        "--bundle",
        str(tmp_path / "model.json"),
        "--request",
        str(tmp_path / "input.json"),
        "--output",
        str(tmp_path / "output.json"),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    result = json.loads((tmp_path / "output.json").read_text())
    assert result["logits"] == expected
    assert subprocess.run(command, capture_output=True, timeout=60).returncode != 0
