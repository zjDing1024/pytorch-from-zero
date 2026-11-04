"""Core fixed-state gradient study invariants, including negative contracts."""

import pytest
import torch

from pytorch_lab import wine
from pytorch_lab import wine_gradient_sampling as gs


@pytest.fixture(scope="module")
def report():
    return gs.run_experiment()


def test_full_report_budget_and_controls(report):
    assert report["all_checks_passed"]
    assert len(report["runs"]) == 6
    assert report["protocol"]["gradient_calls"] == 1266
    assert report["protocol"]["sample_visits"] == 18054
    assert report["protocol"]["optimizer_updates"] == 0
    for run in report["runs"]:
        assert run["all_checks_passed"]
        assert run["weighted_max_abs_error"] < 1e-12
        assert run["naive_partition_error_l2"] > 1e-5
        assert len(run["per_example_gradient_hashes"]) == 107
        for group in run["sampling"]:
            assert len(group["draws"]) == 32
            for draw in group["draws"]:
                assert len(set(draw["indices"])) == group["batch_size"]
                assert all(0 <= i < 107 for i in draw["indices"])
                assert draw["squared_error"] >= 0


def test_replay_and_rng(report):
    state = torch.get_rng_state().clone()
    assert gs.run_experiment() == report
    assert torch.equal(state, torch.get_rng_state())


def test_gradient_buffers_and_parameters_preserved():
    x, y, _ = gs.prepare_train()
    model = wine.WineClassifier("linear", 42)
    before = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    gs.gradient(model, x, y)
    for p, old in zip(model.parameters(), before, strict=True):
        assert torch.equal(p, old)
        assert torch.equal(p.grad, torch.ones_like(p))


@pytest.mark.parametrize("size", [0, -1, 108, 2.0, True])
def test_invalid_population_size(size):
    with pytest.raises(ValueError):
        gs.expected_squared_error(torch.ones(107, 3), size)


def test_zero_population_variance():
    assert gs.expected_squared_error(torch.ones(107, 3), 16) == 0


def test_full_population_zero_error():
    assert gs.expected_squared_error(torch.arange(321.0).reshape(107, 3), 107) == 0


def test_thread_restore_on_failure(monkeypatch):
    old = torch.get_num_threads()

    def fail():
        raise RuntimeError("intentional")

    monkeypatch.setattr(gs, "prepare_train", fail)
    with pytest.raises(RuntimeError):
        gs.run_experiment()
    assert torch.get_num_threads() == old
