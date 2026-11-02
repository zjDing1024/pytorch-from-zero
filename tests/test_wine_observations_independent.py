"""Independent review: optimizer algebra, real reference and partition isolation."""

import json
from importlib.resources import files

import pytest
import torch

from pytorch_lab import wine
from pytorch_lab import wine_observations as observations


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("architecture", ["linear", "mlp16"])
@pytest.mark.parametrize("seed", [42, 7, 123])
def test_independent_step_algebra_and_true_legacy(monkeypatch, architecture, seed):
    x, y, _ = observations.prepare_train()
    original_step = torch.optim.SGD.step
    snapshots = []
    previous_buffers = {}

    def checked_step(optimizer, closure=None):
        assert closure is None
        captured = []
        for group in optimizer.param_groups:
            assert (group["lr"], group["momentum"], group["weight_decay"]) == (0.05, 0.9, 0.0001)
            for parameter in group["params"]:
                before = parameter.detach().clone()
                raw = parameter.grad.detach().clone()
                # Independent formula: initial buffer has no 0.9 multiplier.
                effective = raw + 0.0001 * before
                buffer = (
                    effective
                    if id(parameter) not in previous_buffers
                    else (0.9 * previous_buffers[id(parameter)] + effective)
                )
                expected = before - 0.05 * buffer
                captured.append((parameter, before, raw, buffer, expected))
        result = original_step(optimizer)
        checked = []
        for parameter, before, raw, buffer, expected in captured:
            torch.testing.assert_close(parameter, expected, atol=1e-14, rtol=1e-14)
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"], buffer, atol=1e-14, rtol=1e-14
            )
            previous_buffers[id(parameter)] = buffer.clone()
            checked.append(
                (
                    before,
                    raw,
                    parameter.detach().clone(),
                    optimizer.state[parameter]["momentum_buffer"].detach().clone(),
                )
            )
        snapshots.append(checked)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.SGD, "step", checked_step)
        model, info, trace = observations.train_prefix(x, y, architecture, seed, observe=True)
    assert len(snapshots) == info["updates"] == 14
    assert info["samples_seen"] == 214
    assert [row["batch_size"] for row in trace] == [16] * 6 + [11] + [16] * 6 + [11]
    for row, captured in zip(trace, snapshots, strict=True):
        assert len(row["parameters"]) == len(captured)
        for (name, metric), (before, raw, after, buffer) in zip(
            row["parameters"].items(), captured, strict=True
        ):
            assert metric["gradient_l2"] == float(raw.norm())
            assert metric["parameter_before_l2"] == float(before.norm())
            assert metric["actual_delta_l2"] == float((after - before).norm())
            assert metric["momentum_after_l2"] == float(buffer.norm())
            if row["step"] == 1 and name.startswith("biases."):
                assert torch.count_nonzero(before) == 0
                assert metric["update_to_parameter_ratio"] is None
                assert metric["ratio_status"] == "undefined_zero_parameter_norm"
            else:
                assert metric["update_to_parameter_ratio"] == (
                    float((after - before).norm()) / float(before.norm())
                )
    for epoch, history in enumerate(info["history"]):
        rows = trace[epoch * 7 : (epoch + 1) * 7]
        weighted = 0.0
        for row in rows:
            weighted += row["pre_update_batch_cross_entropy"] * row["batch_size"]
        assert history["online_cross_entropy"] == weighted / 107
    legacy, legacy_info = wine.fit_candidate(x, y, architecture, seed, wine.WineConfig(epochs=2))
    audit, audit_info, audit_trace = observations.train_prefix(
        x, y, architecture, seed, observe=False
    )
    assert info == legacy_info == audit_info
    assert [{k: v for k, v in r.items() if k != "parameters"} for r in trace] == audit_trace
    for name, value in model.state_dict().items():
        assert torch.equal(value, legacy.state_dict()[name])
        assert torch.equal(value, audit.state_dict()[name])


def test_heldout_mutation_and_actual_execution_budget(monkeypatch):
    original_x, original_y = wine.load_wine()
    x, y, preprocessing = observations.prepare_train()
    split = json.loads(files("pytorch_lab").joinpath("data/wine_split.json").read_text())["row_ids"]
    expected_train = original_x[split["train"]]
    assert preprocessing["mean"] == expected_train.mean(0).tolist()
    assert preprocessing["scale"] == expected_train.std(0, correction=0).tolist()
    heldout = split["validation"] + split["test"]
    poisoned_x, poisoned_y = original_x.clone(), original_y.clone()
    poisoned_x[heldout] = float("nan")
    poisoned_y[heldout] = -100
    monkeypatch.setattr(wine, "load_wine", lambda: (poisoned_x, poisoned_y))
    changed_x, changed_y, changed_preprocessing = observations.prepare_train()
    assert torch.equal(x, changed_x) and torch.equal(y, changed_y)
    assert preprocessing == changed_preprocessing

    def forbidden(*args, **kwargs):
        pytest.fail("held-out scoring or new partition derivation called")

    monkeypatch.setattr(wine, "build_split", forbidden)
    monkeypatch.setattr(wine, "evaluate_classifier", forbidden)
    monkeypatch.setattr(wine, "metrics", forbidden)
    original_forward = wine.WineClassifier.forward
    original_step = torch.optim.SGD.step
    counts = {"updates": 0, "training_visits": 0, "epoch_evaluations": 0}

    def guarded_forward(model, batch):
        assert batch.shape[0] in (11, 16, 107)
        # Every forwarded row must originate from standardized training rows.
        assert ((batch[:, None, :] == x[None, :, :]).all(2).any(1)).all()
        if torch.is_grad_enabled():
            counts["training_visits"] += len(batch)
        else:
            assert torch.equal(batch, x)
            counts["epoch_evaluations"] += 1
        return original_forward(model, batch)

    def counted_step(optimizer, closure=None):
        counts["updates"] += 1
        return original_step(optimizer, closure)

    monkeypatch.setattr(wine.WineClassifier, "forward", guarded_forward)
    monkeypatch.setattr(torch.optim.SGD, "step", counted_step)
    rng = torch.get_rng_state().clone()
    result = observations.run_experiment()
    assert result["all_checks_passed"]
    assert counts == {"updates": 252, "training_visits": 3852, "epoch_evaluations": 36}
    assert result["protocol"]["total_optimizer_updates"] == counts["updates"]
    assert result["protocol"]["total_training_sample_visits"] == counts["training_visits"]
    assert torch.equal(rng, torch.get_rng_state())
