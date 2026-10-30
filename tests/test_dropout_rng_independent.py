"""Independent equations and failure review for the fixed CPU replay probe.

The derivative oracle does not call the implementation's fixture, model forward,
loss, or autograd. Its Bernoulli mask does use the same upstream PyTorch Dropout
primitive, so this is not an independent RNG algorithm or cross-version proof.
"""

import copy
import hashlib
import json

import pytest
import torch
from torch.nn import functional as F

from pytorch_lab.dropout_rng import DropoutConfig, DropoutReplay


@pytest.fixture(autouse=True)
def preserve_callers_cpu_state():
    with torch.random.fork_rng(devices=[]):
        yield


def _state_digest(state):
    return hashlib.sha256(bytes(state.tolist())).hexdigest()


def _manual_equations(snapshot):
    """Explicit chain rule for SUM squared error through two affine layers."""
    step = snapshot["steps"]
    x = torch.arange(20, dtype=torch.float64).reshape(5, 4) / 8 - 1 + step / 32
    target = torch.arange(10, dtype=torch.float64).reshape(5, 2) / 16 - 0.25
    parameters = {
        name: torch.tensor(value, dtype=torch.float64) for name, value in snapshot["model"].items()
    }
    first_w, first_b = parameters["0.weight"], parameters["0.bias"]
    last_w, last_b = parameters["2.weight"], parameters["2.bias"]
    hidden = x @ first_w.T + first_b
    rng_before = torch.tensor(snapshot["rng"], dtype=torch.uint8)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng_before)
        mask = F.dropout(
            torch.ones_like(hidden),
            p=snapshot["config"]["p"],
            training=snapshot["config"]["training"],
        )
        rng_after = torch.get_rng_state().clone()
    masked = hidden * mask
    output = masked @ last_w.T + last_b
    residual = output - target
    output_derivative = 2 * residual
    hidden_derivative = (output_derivative @ last_w) * mask
    return {
        "output": output,
        "loss": residual.square().sum(),
        "input_grad": hidden_derivative @ first_w,
        "parameter_grads": {
            "0.weight": hidden_derivative.T @ x,
            "0.bias": hidden_derivative.sum(0),
            "2.weight": output_derivative.T @ masked,
            "2.bias": output_derivative.sum(0),
        },
        "rng_before_digest": _state_digest(rng_before),
        "rng_after_digest": _state_digest(rng_after),
    }


@pytest.mark.parametrize("seed,model_seed", [(42, 17), (7, 3), (123, 99)])
@pytest.mark.parametrize("probability", [0.0, 0.35, 1.0])
@pytest.mark.parametrize("training", [True, False])
def test_four_steps_and_recovery_against_explicit_chain_rule(
    seed, model_seed, probability, training
):
    config = DropoutConfig(seed=seed, model_seed=model_seed, p=probability, training=training)
    caller_state = torch.get_rng_state().clone()
    replay = DropoutReplay(config)
    original_weights = copy.deepcopy(replay.snapshot()["model"])
    for _ in range(4):
        before = replay.snapshot()
        expected = _manual_equations(before)
        restored = DropoutReplay.from_snapshot(before, expected_config=config)
        actual = replay.step()
        assert restored.step() == actual  # Exact within-runtime restoration, not a tolerance claim.
        for field in ("output", "loss", "input_grad"):
            torch.testing.assert_close(
                torch.tensor(actual[field], dtype=torch.float64),
                expected[field],
                atol=1e-12,
                rtol=1e-12,
            )
        for name, gradient in expected["parameter_grads"].items():
            torch.testing.assert_close(
                torch.tensor(actual["parameter_grads"][name], dtype=torch.float64),
                gradient,
                atol=1e-12,
                rtol=1e-12,
            )
        assert actual["rng_before_digest"] == expected["rng_before_digest"]
        assert actual["rng_after_digest"] == expected["rng_after_digest"]
        assert replay.snapshot()["model"] == original_weights  # No optimizer updates.
        assert torch.equal(torch.get_rng_state(), caller_state)


@pytest.mark.parametrize("stage", ["forward", "backward"])
@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_rng_and_retry_after_real_random_or_gradient_mutation(monkeypatch, stage, exception):
    replay = DropoutReplay()
    replay.step()
    before = replay.snapshot()
    caller_state = torch.get_rng_state().clone()
    expected = DropoutReplay.from_snapshot(before).step()
    with monkeypatch.context() as patch:
        if stage == "forward":
            original = replay.model[1].forward

            def mutate_then_raise(value):
                original(value)
                assert not torch.equal(torch.get_rng_state(), torch.tensor(before["rng"]))
                raise exception("review fault after Dropout consumed RNG")

            patch.setattr(replay.model[1], "forward", mutate_then_raise)
        else:
            original = torch.Tensor.backward

            def mutate_then_raise(value, *args, **kwargs):
                original(value, *args, **kwargs)
                assert all(parameter.grad is not None for parameter in replay.model.parameters())
                raise exception("review fault after backward wrote gradients")

            patch.setattr(torch.Tensor, "backward", mutate_then_raise)
        with pytest.raises(exception):
            replay.step()
    assert replay.snapshot() == before
    assert torch.equal(torch.get_rng_state(), caller_state)
    assert replay.step() == expected  # Stale partial gradients cannot contaminate the retry.
    assert torch.equal(torch.get_rng_state(), caller_state)


def test_unrelated_random_consumption_between_calls_does_not_change_probe():
    first, second = DropoutReplay(), DropoutReplay()
    for count in (1, 73, 2, 104):
        expected = first.step()
        torch.randn(count)
        caller_state = torch.get_rng_state().clone()
        assert second.step() == expected
        assert torch.equal(torch.get_rng_state(), caller_state)


def _bad_snapshots(good):
    changes = [
        ("steps", -1),
        ("steps", True),
        ("steps", 1_000_001),
        ("rng", good["rng"][:-1]),
        ("rng", [0] * len(good["rng"])),
        ("rng", [True] + good["rng"][1:]),
        ("rng", [256] + good["rng"][1:]),
        ("torch_version", "different-runtime"),
        ("fixture", "different-data"),
    ]
    for field, value in changes:
        candidate = copy.deepcopy(good)
        candidate[field] = value
        yield candidate
    wrong_shape = copy.deepcopy(good)
    wrong_shape["model"]["0.weight"] = [[1.0]]
    yield wrong_shape
    nonfinite = copy.deepcopy(good)
    nonfinite["model"]["0.weight"][0][0] = float("nan")
    yield nonfinite
    extra = copy.deepcopy(good)
    extra["unrecognized"] = True
    yield extra


def test_twelve_malformed_snapshots_leave_callers_rng_unchanged():
    replay = DropoutReplay()
    replay.step()
    good = replay.snapshot()
    caller_state = torch.get_rng_state().clone()
    variants = list(_bad_snapshots(good))
    assert len(variants) == 12
    for bad in variants:
        with pytest.raises(ValueError):
            DropoutReplay.from_snapshot(bad)
        assert torch.equal(torch.get_rng_state(), caller_state)
    assert DropoutReplay.from_snapshot(good).step() == replay.step()


def test_trusted_file_roundtrip_checksum_and_config_binding(tmp_path):
    replay = DropoutReplay()
    replay.step()
    file = tmp_path / "review-snapshot.json"
    caller_state = torch.get_rng_state().clone()
    replay.save(file)
    assert DropoutReplay.load(file).step() == replay.step()
    with pytest.raises(FileExistsError):
        replay.save(file)
    with pytest.raises(ValueError):
        DropoutReplay.load(file, expected_config=DropoutConfig(training=False))
    document = json.loads(file.read_text())
    document["payload"]["steps"] += 1
    file.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="checksum"):
        DropoutReplay.load(file)
    assert torch.equal(torch.get_rng_state(), caller_state)


@pytest.mark.parametrize("target", ["root", "linear", "dropout", "probability"])
def test_mode_or_probability_mutation_cannot_emit_a_misleading_snapshot(target):
    replay = DropoutReplay()
    if target == "root":
        replay.model.eval()
    elif target == "linear":
        replay.model[0].eval()
    elif target == "dropout":
        replay.model[1].eval()
    else:
        replay.model[1].p = 0.5
    caller_state = torch.get_rng_state().clone()
    for operation in (replay.snapshot, replay.step):
        with pytest.raises(ValueError):
            operation()
        assert torch.equal(torch.get_rng_state(), caller_state)


def test_terminal_counter_roundtrips_but_cannot_advance():
    from pytorch_lab.dropout_rng import MAX_STEPS

    snapshot = DropoutReplay().snapshot()
    snapshot["steps"] = MAX_STEPS
    # A syntactically valid trusted snapshot is not proof of prior execution.
    replay = DropoutReplay.from_snapshot(snapshot)
    assert replay.snapshot() == snapshot
    caller_state = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        replay.step()
    assert replay.snapshot() == snapshot
    assert torch.equal(torch.get_rng_state(), caller_state)


def test_nonfinite_input_gradient_is_rejected_without_advancing_rng(monkeypatch):
    import pytorch_lab.dropout_rng as probe

    replay = DropoutReplay()
    before = replay.snapshot()
    expected = DropoutReplay.from_snapshot(before).step()
    caller_state = torch.get_rng_state().clone()
    captured = {}
    original_fixture, original_backward = probe.fixture, torch.Tensor.backward

    def observe_fixture(step):
        x, y = original_fixture(step)
        captured["input"] = x
        return x, y

    def corrupt_input_gradient(value, *args, **kwargs):
        original_backward(value, *args, **kwargs)
        captured["input"].grad.fill_(float("inf"))

    with monkeypatch.context() as patch:
        patch.setattr(probe, "fixture", observe_fixture)
        patch.setattr(torch.Tensor, "backward", corrupt_input_gradient)
        with pytest.raises(ValueError, match="non-finite"):
            replay.step()
    assert replay.snapshot() == before
    assert torch.equal(torch.get_rng_state(), caller_state)
    assert replay.step() == expected


@pytest.mark.parametrize("value", [True, "0.25", None, [0.25]])
def test_snapshot_model_scalar_schema_is_not_implicit_coercion(value):
    snapshot = DropoutReplay().snapshot()
    snapshot["model"]["0.weight"][0][0] = value
    caller_state = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        DropoutReplay.from_snapshot(snapshot)
    assert torch.equal(torch.get_rng_state(), caller_state)


@pytest.mark.parametrize("mutation", ["nonfinite", "dtype", "architecture", "buffer"])
def test_mutated_owner_rejected_before_snapshot_or_step(mutation):
    replay = DropoutReplay()
    if mutation == "nonfinite":
        with torch.no_grad():
            replay.model[0].weight[0, 0] = float("inf")
    elif mutation == "dtype":
        replay.model.float()
    elif mutation == "architecture":
        replay.model[1] = torch.nn.Identity()
    else:
        replay.model.register_buffer("untracked", torch.zeros(1, dtype=torch.float64))
    caller_state = torch.get_rng_state().clone()
    for operation in (replay.snapshot, replay.step):
        with pytest.raises(ValueError):
            operation()
        assert torch.equal(torch.get_rng_state(), caller_state)
