"""Independent actual-tensor and sample-budget audit of the batch experiment."""

import json
from importlib.resources import files

import pytest
import torch
from torch.nn import functional as F

from pytorch_lab import wine
from pytorch_lab import wine_batch_budget as experiment
from pytorch_lab.wine_observations import prepare_train, tensor_digest


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def batches(orders, size, limit):
    result = []
    for order in orders:
        for start in range(0, len(order), size):
            result.append(order[start : start + size])
            if len(result) == limit:
                return result
    raise AssertionError("not enough explicit sample orders")


@pytest.mark.parametrize("architecture", ["linear", "mlp16"])
@pytest.mark.parametrize("seed", [42, 7, 123])
def test_real_rows_gradients_updates_and_manual_reference(monkeypatch, architecture, seed):
    x, y, _ = prepare_train()
    orders = experiment.make_orders(seed)
    assert len(orders) == 4
    assert all(sorted(row) == list(range(107)) for row in orders)
    observed = {}
    for arm, size, updates, visits in (
        ("batch16_14updates", 16, 14, 214),
        ("batch32_2epochs", 32, 8, 214),
        ("batch32_14updates", 32, 14, 385),
    ):
        expected_batches = batches(orders, size, updates)
        assert sum(map(len, expected_batches)) == visits
        actual_batches, actual_states = [], []
        real_forward, real_step = wine.WineClassifier.forward, torch.optim.SGD.step

        def forward(model, value, actual_batches=actual_batches, real_forward=real_forward):
            if torch.is_grad_enabled():
                actual_batches.append(
                    [int(torch.nonzero((x == row).all(dim=1)).item()) for row in value]
                )
            return real_forward(model, value)

        def step(optimizer, closure=None, real_step=real_step, actual_states=actual_states):
            result = real_step(optimizer, closure)
            actual_states.append([p.detach().clone() for p in optimizer.param_groups[0]["params"]])
            return result

        with monkeypatch.context() as patch:
            patch.setattr(wine.WineClassifier, "forward", forward)
            patch.setattr(torch.optim.SGD, "step", step)
            model, report = experiment.train_run(x, y, architecture, seed, arm, orders)
        assert actual_batches == expected_batches
        assert len(actual_states) == updates
        assert report["updates"] == updates and report["sample_visits"] == visits
        assert [row["row_positions"] for row in report["trace"]] == actual_batches
        assert [row["batch_size"] for row in report["trace"]] == list(map(len, expected_batches))
        reference = wine.WineClassifier(architecture, seed)
        parameters = list(reference.parameters())
        buffers = [None] * len(parameters)
        for index, ids in enumerate(expected_batches):
            # Sum divided by the actual count independently exercises the 11-row tail.
            loss = F.cross_entropy(reference(x[ids]), y[ids], reduction="sum") / len(ids)
            assert report["trace"][index]["loss_before_update"] == pytest.approx(
                loss.item(), rel=1e-14, abs=1e-14
            )
            gradients = torch.autograd.grad(loss, parameters)
            before = torch.cat([p.detach().reshape(-1) for p in parameters]).clone()
            with torch.no_grad():
                for j, (p, gradient) in enumerate(zip(parameters, gradients, strict=True)):
                    q = gradient + 1e-4 * p
                    buffers[j] = q.clone() if buffers[j] is None else 0.9 * buffers[j] + q
                    p.add_(buffers[j], alpha=-0.05)
                    torch.testing.assert_close(p, actual_states[index][j], atol=1e-14, rtol=1e-14)
                    observed_flat = torch.cat([p.reshape(-1) for p in actual_states[index]])
            assert report["trace"][index]["parameters_after_sha256"] == tensor_digest(observed_flat)
            expected_norm = float(torch.linalg.vector_norm(observed_flat - before))
            assert report["trace"][index]["actual_update_l2"] == pytest.approx(
                expected_norm, rel=1e-14, abs=1e-14
            )
        with torch.no_grad():
            final_ce = F.cross_entropy(reference(x), y).item()
        assert report["final_train_cross_entropy"] == pytest.approx(final_ce, rel=1e-14, abs=1e-14)
        for a, b in zip(model.parameters(), parameters, strict=True):
            torch.testing.assert_close(a, b, atol=1e-14, rtol=1e-14)
        observed[arm] = actual_batches
    assert [i for b in observed["batch16_14updates"] for i in b] == [
        i for b in observed["batch32_2epochs"] for i in b
    ]
    assert observed["batch32_14updates"][:8] == observed["batch32_2epochs"]


def test_poisoned_heldout_actual_totals_no_duplicate_baseline(monkeypatch):
    train_x, _, preprocessing = prepare_train()
    raw_x, raw_y = wine.load_wine()
    split = json.loads(files("pytorch_lab").joinpath("data/wine_split.json").read_text())
    heldout = split["row_ids"]["validation"] + split["row_ids"]["test"]
    raw_x[heldout], raw_y[heldout] = float("nan"), -100
    counts = {"updates": 0, "train_visits": 0, "eval_forwards": 0}
    real_forward, real_step = wine.WineClassifier.forward, torch.optim.SGD.step

    def reject(*args, **kwargs):
        raise AssertionError("heldout evaluation or split regeneration forbidden")

    def forward(model, value):
        assert torch.isfinite(value).all()
        assert all(bool((train_x == row).all(dim=1).any()) for row in value)
        if torch.is_grad_enabled():
            counts["train_visits"] += len(value)
        else:
            assert torch.equal(value, train_x)
            counts["eval_forwards"] += 1
        return real_forward(model, value)

    def step(optimizer, closure=None):
        counts["updates"] += 1
        return real_step(optimizer, closure)

    monkeypatch.setattr(wine, "load_wine", lambda: (raw_x.clone(), raw_y.clone()))
    monkeypatch.setattr(wine, "build_split", reject)
    monkeypatch.setattr(wine, "metrics", reject)
    monkeypatch.setattr(wine.WineClassifier, "forward", forward)
    monkeypatch.setattr(torch.optim.SGD, "step", step)
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    result = experiment.run_experiment()
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.get_num_threads() == threads
    assert counts == {"updates": 216, "train_visits": 4878, "eval_forwards": 24}
    assert result["preprocessing"] == preprocessing
