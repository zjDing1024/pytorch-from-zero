"""Contract tests for a paired optimizer experiment, not model-performance gates."""

import json
from pathlib import Path

import pytest
import torch

from pytorch_lab import wine
from pytorch_lab import wine_momentum as wm


@pytest.fixture(scope="module")
def report():
    return wm.run_experiment()


def test_frozen_protocol_and_actual_cost(report):
    plan = json.loads((Path(__file__).parents[1] / "docs/wine-momentum-plan.json").read_text())
    assert plan["epochs"] == report["protocol"]["epochs"] == 2
    assert plan["total_updates"] == report["protocol"]["total_optimizer_updates"] == 168
    assert report["protocol"]["total_training_sample_visits"] == 2568
    assert report["protocol"]["full_train_evaluation_forwards"] == 24
    assert report["protocol"]["scored_partitions"] == ["train"]
    assert report["all_checks_passed"] is True
    assert len(report["runs"]) == 6
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("index", range(6))
def test_pair_controls_and_observed_budgets(report, index):
    run = report["runs"][index]
    assert all(run["checks"].values())
    trace = run["trace"]
    assert [r["batch_size"] for r in trace] == [16] * 6 + [11] + [16] * 6 + [11]
    assert trace[0]["paired_parameter_distance_l2"] == 0
    assert trace[1]["paired_parameter_distance_l2"] > 0
    assert trace[0]["paired_raw_gradients_equal"]
    assert trace[1]["paired_raw_gradients_equal"]
    # Later gradients need not be equal; this is a trajectory intervention.
    for arm, mu in enumerate(wm.MOMENTA):
        data = run["arms"][arm]
        assert data["momentum"] == mu
        assert data["updates"] == 14 and data["sample_visits"] == 214
        length = 0.0
        for row in trace:
            length += row["arms"][arm]["actual_delta_l2"]
        assert data["cumulative_path_length"] == length
        for epoch in (1, 2):
            rows = [r for r in trace if r["epoch"] == epoch]
            online = 0.0
            for row in rows:
                online += row["arms"][arm]["pre_update_batch_cross_entropy"] * row["batch_size"]
            online /= 107
            assert data["history"][epoch - 1]["online_cross_entropy"] == online
        assert all((r["arms"][arm]["buffer_l2"] is None) == (mu == 0) for r in trace)
    assert run["train_ce_difference_momentum_minus_plain"] == (
        run["arms"][1]["history"][-1]["final_train_cross_entropy"]
        - run["arms"][0]["history"][-1]["final_train_cross_entropy"]
    )


@pytest.mark.parametrize("architecture", wine.ARCHITECTURES)
def test_mu_point_nine_preserves_existing_prefix(architecture):
    from pytorch_lab.wine_observations import train_prefix

    x, y, _ = wm.prepare_train()
    models, data = wm.train_pair(x, y, architecture, 42)
    old, info, _ = train_prefix(x, y, architecture, 42, observe=False)
    assert data["arms"][1]["history"] == info["history"]
    for key, tensor in old.state_dict().items():
        assert torch.equal(tensor, models[1].state_dict()[key])


@pytest.mark.parametrize("bad", ["rows", "dtype", "nan", "labels"])
def test_invalid_training_inputs(bad):
    x, y, _ = wm.prepare_train()
    if bad == "rows":
        x, y = x[:106], y[:106]
    elif bad == "dtype":
        x = x.float()
    elif bad == "nan":
        x[0, 0] = float("nan")
    else:
        y[0] = 3
    with pytest.raises((ValueError, TypeError)):
        wm.train_pair(x, y, "linear", 42)


def test_threads_and_cpu_rng_restored():
    count = torch.get_num_threads()
    state = torch.get_rng_state().clone()
    wm.run_experiment()
    assert torch.get_num_threads() == count
    assert torch.equal(torch.get_rng_state(), state)


def test_threads_restored_on_failure(monkeypatch):
    count = torch.get_num_threads()

    def fail():
        raise RuntimeError("fixture failed")

    monkeypatch.setattr(wm, "prepare_train", fail)
    with pytest.raises(RuntimeError, match="fixture failed"):
        wm.run_experiment()
    assert torch.get_num_threads() == count


def test_false_optimizer_equation_is_not_hidden(monkeypatch):
    original = torch.optim.SGD.step

    def corrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        with torch.no_grad():
            self.param_groups[0]["params"][0].add_(0.001)
        return result

    monkeypatch.setattr(torch.optim.SGD, "step", corrupt)
    result = wm.run_experiment()
    assert result["all_checks_passed"] is False
    assert all(not run["checks"]["all_sgd_equations"] for run in result["runs"])
