"""Contract checks for the deliberately narrow stochastic replay probe."""

from copy import deepcopy

import pytest
import torch

from pytorch_lab.dropout_rng import DropoutConfig, DropoutReplay, mode_probe


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("p", [0.0, 0.35, 1.0])
@pytest.mark.parametrize("training", [False, True])
def test_exact_replay_and_global_isolation(seed, p, training):
    outer = torch.get_rng_state().clone()
    probe = DropoutReplay(DropoutConfig(seed=seed, p=p, training=training))
    initial = deepcopy(probe.snapshot()["model"])
    probe.step()
    restored = DropoutReplay.from_snapshot(probe.snapshot())
    for _ in range(3):
        assert probe.step() == restored.step()
    assert probe.snapshot() == restored.snapshot()
    assert initial == probe.snapshot()["model"]
    assert torch.equal(outer, torch.get_rng_state())


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", True),
        ("seed", -1),
        ("seed", 2**63),
        ("model_seed", 1.5),
        ("p", True),
        ("p", -0.1),
        ("p", 1.1),
        ("p", float("nan")),
        ("training", 1),
    ],
)
def test_bad_config(field, value):
    with pytest.raises(ValueError):
        DropoutConfig(**{field: value})


@pytest.mark.parametrize(
    "kind",
    [
        "keys",
        "format",
        "version",
        "fixture",
        "steps",
        "rng_length",
        "rng_byte",
        "model_key",
        "model_shape",
        "model_nan",
        "config",
    ],
)
def test_bad_snapshot(kind):
    state = DropoutReplay().snapshot()
    if kind == "keys":
        state["extra"] = 1
    elif kind == "format":
        state["format"] = "future"
    elif kind == "version":
        state["torch_version"] = "other"
    elif kind == "fixture":
        state["fixture"] = "other"
    elif kind == "steps":
        state["steps"] = True
    elif kind == "rng_length":
        state["rng"].pop()
    elif kind == "rng_byte":
        state["rng"][0] = True
    elif kind == "model_key":
        state["model"].pop("0.bias")
    elif kind == "model_shape":
        state["model"]["0.bias"] = [1.0]
    elif kind == "model_nan":
        state["model"]["0.bias"][0] = float("nan")
    elif kind == "config":
        state["config"]["p"] = 2
    with pytest.raises(ValueError):
        DropoutReplay.from_snapshot(state)


def test_expected_config():
    with pytest.raises(ValueError, match="expected config"):
        DropoutReplay.from_snapshot(
            DropoutReplay().snapshot(), expected_config=DropoutConfig(p=0.5)
        )


def test_file_roundtrip_integrity_and_no_overwrite(tmp_path):
    p = tmp_path / "snapshot.json"
    probe = DropoutReplay()
    probe.step()
    probe.save(p)
    assert probe.step() == DropoutReplay.load(p).step()
    before = p.read_bytes()
    with pytest.raises(FileExistsError):
        probe.save(p)
    assert p.read_bytes() == before
    p.write_text(p.read_text().replace('"steps": 1', '"steps": 2'))
    with pytest.raises(ValueError, match="checksum"):
        DropoutReplay.load(p)


def test_mode_is_not_gradient_recording():
    r = mode_probe()
    assert r["train"]["output"] == r["no_grad"]["output"]
    assert r["train"]["rng_changed"] and r["no_grad"]["rng_changed"]
    assert r["train"]["requires_grad"] and not r["no_grad"]["requires_grad"]
    assert not r["eval"]["rng_changed"] and r["eval"]["requires_grad"]


def test_snapshot_no_alias():
    probe = DropoutReplay()
    snapshot = probe.snapshot()
    snapshot["rng"][0] ^= 1
    snapshot["model"]["0.bias"][0] += 1
    assert snapshot != probe.snapshot()
