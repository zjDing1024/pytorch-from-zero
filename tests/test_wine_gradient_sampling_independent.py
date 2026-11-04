"""Independent reconstruction review: actual tensors, finite populations and isolation."""

import itertools

import pytest
import torch

from pytorch_lab import wine
from pytorch_lab import wine_gradient_sampling as study


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_exhaustive_finite_population(size):
    population = torch.tensor(
        [[1.0, 2.0], [-3.0, 4.0], [2.0, -1.0], [5.0, 0.0]], dtype=torch.float64
    )
    means = torch.stack(
        [population[list(ids)].mean(0) for ids in itertools.combinations(range(4), size)]
    )
    mean = population.mean(0)
    assert torch.allclose(means.mean(0), mean, atol=1e-14, rtol=1e-14)
    actual = float((means - mean).square().sum(1).mean())
    assert study.expected_squared_error(population, size) == pytest.approx(actual, abs=1e-14)


def test_linear_gradient_analytic_and_existing_buffers():
    x, y, _ = study.prepare_train()
    model = wine.WineClassifier("linear", 42)
    before = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.full_like(p, 3.25)
    with torch.no_grad():
        residual = model(x).softmax(1)
        residual[torch.arange(len(y)), y] -= 1
        residual /= len(y)
        expected = torch.cat([(residual.T @ x).flatten(), residual.sum(0)])
    actual = study.gradient(model, x, y)
    assert torch.allclose(actual, expected, atol=1e-14, rtol=1e-14)
    assert not actual.requires_grad
    for p, old in zip(model.parameters(), before, strict=True):
        assert torch.equal(p, old)
        assert torch.equal(p.grad, torch.full_like(p, 3.25))


@pytest.mark.parametrize("architecture", wine.ARCHITECTURES)
def test_capture_actual_subset_gradients_and_tail(monkeypatch, architecture):
    x, y, _ = study.prepare_train()
    original = study.gradient
    captured = []

    def capture(model, xb, yb):
        value = original(model, xb, yb)
        captured.append(value.clone())
        return value

    monkeypatch.setattr(study, "gradient", capture)
    result = study.analyze_model(x, y, architecture, 7)
    full = captured[0]
    individuals = torch.stack(captured[1:108])
    partitions = captured[108:115]
    weighted = (
        torch.stack([g * n for g, n in zip(partitions, [16] * 6 + [11], strict=True)]).sum(0) / 107
    )
    assert torch.allclose(weighted, full, atol=1e-14, rtol=1e-14)
    naive_error = float(torch.linalg.vector_norm(torch.stack(partitions).mean(0) - full))
    assert result["naive_partition_error_l2"] == pytest.approx(naive_error, abs=1e-14)
    assert naive_error > 1e-3
    offset = 115
    for group in result["sampling"]:
        for draw in group["draws"]:
            ids = draw["indices"]
            assert len(ids) == len(set(ids)) == group["batch_size"]
            expected = individuals[ids].mean(0)
            actual = captured[offset]
            assert torch.allclose(actual, expected, atol=1e-14, rtol=1e-14)
            assert draw["gradient_sha256"] == study.tensor_digest(actual)
            assert draw["squared_error"] == pytest.approx(
                float((actual - full).square().sum()), abs=1e-14
            )
            offset += 1
    assert offset == len(captured) == 211


def test_full_probe_train_isolation_and_actual_cost(monkeypatch):
    x, y, prep = study.prepare_train()
    raw_x, raw_y = wine.load_wine()
    excluded = sorted(set(range(178)) - set(prep["row_ids"]))
    raw_x[excluded] = float("nan")
    raw_y[excluded] = -100
    monkeypatch.setattr(wine, "load_wine", lambda: (raw_x, raw_y))

    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden optimizer update or resplit")

    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    monkeypatch.setattr(wine, "build_split", forbidden)
    original = wine.WineClassifier.forward
    calls = visits = 0

    def guarded(model, xb):
        nonlocal calls, visits
        assert torch.isfinite(xb).all()
        assert (xb[:, None, :] == x[None, :, :]).all(2).any(1).all()
        calls += 1
        visits += len(xb)
        return original(model, xb)

    monkeypatch.setattr(wine.WineClassifier, "forward", guarded)
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    result = study.run_experiment()
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.get_num_threads() == threads
    assert calls == result["protocol"]["gradient_calls"] == 1266
    assert visits == result["protocol"]["sample_visits"] == 18054
    assert result["protocol"]["optimizer_updates"] == 0
    assert result["all_checks_passed"]
