"""Independent measured-state audits of the fixed Wine momentum comparison."""

import json
from importlib.resources import files

import pytest
import torch

from pytorch_lab import wine
from pytorch_lab import wine_momentum as experiment
from pytorch_lab.wine_observations import prepare_train


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def flat(values):
    return torch.cat([value.reshape(-1) for value in values])


@pytest.mark.parametrize("architecture", ["linear", "mlp16"])
@pytest.mark.parametrize("seed", [42, 7, 123])
def test_actual_states_equations_pairing_and_legacy(monkeypatch, architecture, seed):
    x, y, _ = prepare_train()
    snapshots = {0.0: [], 0.9: []}
    real_step = torch.optim.SGD.step
    expected_buffers = {}

    def audited_step(optimizer, closure=None):
        assert closure is None
        assert len(optimizer.param_groups) == 1
        group = optimizer.param_groups[0]
        mu = group["momentum"]
        assert mu in (0.0, 0.9)
        assert (group["lr"], group["weight_decay"], group["dampening"]) == (0.05, 1e-4, 0)
        assert not group["nesterov"] and not group["maximize"]
        parameters = group["params"]
        before = [p.detach().clone() for p in parameters]
        raw = [p.grad.detach().clone() for p in parameters]
        effective = [g + 1e-4 * p for g, p in zip(raw, before, strict=True)]
        direction = []
        for parameter, gradient in zip(parameters, effective, strict=True):
            old = expected_buffers.get(parameter)
            current = gradient if old is None or mu == 0 else 0.9 * old + gradient
            direction.append(current)
            if mu:
                expected_buffers[parameter] = current.clone()
        result = real_step(optimizer, closure)
        after = [p.detach().clone() for p in parameters]
        delta = [q - p for p, q in zip(before, after, strict=True)]
        for parameter, change, velocity in zip(parameters, delta, direction, strict=True):
            torch.testing.assert_close(change, -0.05 * velocity, atol=1e-14, rtol=1e-14)
            state = optimizer.state.get(parameter, {})
            if mu:
                torch.testing.assert_close(
                    state["momentum_buffer"], velocity, atol=1e-14, rtol=1e-14
                )
            else:
                assert "momentum_buffer" not in state
        snapshots[mu].append(
            {
                "before": flat(before),
                "raw": flat(raw),
                "effective": flat(effective),
                "after": flat(after),
                "delta": flat(delta),
                "direction": flat(direction),
            }
        )
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.SGD, "step", audited_step)
        models, report = experiment.train_pair(x, y, architecture, seed)
    assert len(snapshots[0.0]) == len(snapshots[0.9]) == 14
    plain, momentum = snapshots[0.0], snapshots[0.9]
    for key in ("before", "raw", "delta", "after"):
        assert torch.equal(plain[0][key], momentum[0][key])
    assert torch.equal(plain[1]["before"], momentum[1]["before"])
    assert torch.equal(plain[1]["raw"], momentum[1]["raw"])
    assert not torch.equal(plain[1]["after"], momentum[1]["after"])
    torch.testing.assert_close(
        momentum[1]["delta"] - plain[1]["delta"],
        -0.05 * 0.9 * plain[0]["effective"],
        atol=1e-14,
        rtol=1e-14,
    )
    assert not torch.equal(plain[2]["raw"], momentum[2]["raw"])
    assert [t["batch_size"] for t in report["trace"]] == [16] * 6 + [11] + [16] * 6 + [11]
    for index, trace in enumerate(report["trace"]):
        expected_distance = float(
            torch.linalg.vector_norm(momentum[index]["after"] - plain[index]["after"])
        )
        assert trace["paired_parameter_distance_l2"] == expected_distance
        for arm, mu in enumerate((0.0, 0.9)):
            observed = trace["arms"][arm]
            actual = snapshots[mu][index]
            assert observed["actual_delta_l2"] == float(torch.linalg.vector_norm(actual["delta"]))
            assert observed["raw_gradient_l2"] == float(torch.linalg.vector_norm(actual["raw"]))
            if mu == 0:
                assert observed["buffer_l2"] is None
            else:
                assert observed["buffer_l2"] == pytest.approx(
                    float(torch.linalg.vector_norm(actual["direction"])), abs=1e-14, rel=1e-14
                )
    # The old uninstrumented trainer independently verifies order, budget, and history.
    for arm, mu in enumerate((0.0, 0.9)):
        reference, history = wine.fit_candidate(
            x, y, architecture, seed, wine.WineConfig(epochs=2, momentum=mu)
        )
        for measured, expected in zip(
            models[arm].parameters(), reference.parameters(), strict=True
        ):
            assert torch.equal(measured, expected)
        assert report["arms"][arm]["history"] == history["history"]
        assert report["arms"][arm]["updates"] == history["updates"] == 14
        assert report["arms"][arm]["sample_visits"] == history["samples_seen"] == 214


def test_heldout_poison_forward_guard_real_counts_and_rng(monkeypatch):
    train_x, train_y, preprocessing = prepare_train()
    raw_x, raw_y = wine.load_wine()
    manifest = json.loads(files("pytorch_lab").joinpath("data/wine_split.json").read_text())
    heldout = manifest["row_ids"]["validation"] + manifest["row_ids"]["test"]
    raw_x[heldout] = float("nan")
    raw_y[heldout] = -100
    counts = {"train_visits": 0, "updates": 0, "eval_forwards": 0}
    real_forward = wine.WineClassifier.forward
    real_step = torch.optim.SGD.step

    def reject(*args, **kwargs):
        raise AssertionError("heldout evaluation or split regeneration is forbidden")

    def guarded_forward(model, value):
        assert torch.isfinite(value).all()
        assert all(bool((train_x == row).all(dim=1).any()) for row in value)
        if torch.is_grad_enabled():
            counts["train_visits"] += len(value)
        else:
            assert torch.equal(value, train_x)
            counts["eval_forwards"] += 1
        return real_forward(model, value)

    def counted_step(optimizer, closure=None):
        counts["updates"] += 1
        return real_step(optimizer, closure)

    monkeypatch.setattr(wine, "load_wine", lambda: (raw_x.clone(), raw_y.clone()))
    monkeypatch.setattr(wine, "build_split", reject)
    monkeypatch.setattr(wine, "metrics", reject)
    monkeypatch.setattr(wine.WineClassifier, "forward", guarded_forward)
    monkeypatch.setattr(torch.optim.SGD, "step", counted_step)
    before_rng = torch.get_rng_state().clone()
    before_threads = torch.get_num_threads()
    result = experiment.run_experiment()
    assert torch.equal(before_rng, torch.get_rng_state())
    assert before_threads == torch.get_num_threads()
    assert result["preprocessing"] == preprocessing
    assert counts == {"train_visits": 2568, "updates": 168, "eval_forwards": 24}
    assert result["protocol"]["total_training_sample_visits"] == counts["train_visits"]
    assert result["protocol"]["total_optimizer_updates"] == counts["updates"]
    assert result["protocol"]["full_train_evaluation_forwards"] == counts["eval_forwards"]
    assert train_y.shape == (107,)
