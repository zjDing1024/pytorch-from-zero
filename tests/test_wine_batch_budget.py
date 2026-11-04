import pytest
import torch

from pytorch_lab import wine_batch_budget as w


@pytest.fixture(scope="module")
def report():
    return w.run_experiment()


def test_report_budget_and_controls(report):
    assert report["all_checks_passed"]
    assert report["protocol"]["optimizer_updates"] == 216
    assert report["protocol"]["training_sample_visits"] == 4878
    assert report["protocol"]["evaluation_forwards"] == 24
    assert len(report["configurations"]) == 6
    for c in report["configurations"]:
        assert len(c["runs"]) == 3
        for r in c["runs"]:
            assert r["final_train_cross_entropy"] >= 0
            assert r["sample_visits"] == sum(t["batch_size"] for t in r["trace"])
            assert r["updates"] == len(r["trace"])
            total = 0.0
            for t in r["trace"]:
                total += t["loss_before_update"] * t["batch_size"]
            assert r["sample_weighted_online_cross_entropy"] == total / r["sample_visits"]


@pytest.mark.parametrize("seed", [42, 7, 123])
def test_orders_local_and_reproducible(seed):
    state = torch.get_rng_state().clone()
    assert w.make_orders(seed) == w.make_orders(seed)
    assert all(sorted(o) == list(range(107)) for o in w.make_orders(seed))
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("seed", [-1, True, 1.5, None])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        w.make_orders(seed)


@pytest.mark.parametrize("bad", [[], [[0] * 107] * 4, [list(range(107))] * 3])
def test_invalid_orders(bad):
    x, y, _ = w.prepare_train()
    with pytest.raises(ValueError, match="permutations"):
        w.train_run(x, y, "linear", 42, "batch16_14updates", bad)


def test_invalid_arm():
    x, y, _ = w.prepare_train()
    with pytest.raises(ValueError):
        w.train_run(x, y, "linear", 42, "unknown", w.make_orders(42))


def test_failure_restores_threads(monkeypatch):
    before = torch.get_num_threads()

    def fail():
        raise RuntimeError("failed")

    monkeypatch.setattr(w, "prepare_train", fail)
    with pytest.raises(RuntimeError):
        w.run_experiment()
    assert torch.get_num_threads() == before
