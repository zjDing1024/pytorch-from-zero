"""Local integration and measurement contract tests for the frozen Wine prefix."""

import copy
import json

import pytest
import torch

from pytorch_lab import wine
from pytorch_lab import wine_observations as obs


@pytest.fixture(scope="module")
def report():
    return obs.run_experiment()


@pytest.mark.parametrize("index", range(6))
def test_prefix_contract(report, index):
    run = report["runs"][index]
    assert run["all_checks_passed"]
    assert run["updates"] == 14
    assert run["samples_seen"] == 214
    assert [row["batch_size"] for row in run["trace"]] == ([16] * 6 + [11]) * 2
    assert len(run["history"]) == 2
    assert all(row["pre_update_batch_cross_entropy"] > 0 for row in run["trace"])
    assert run["architecture"] == wine.ARCHITECTURES[index // 3]
    assert run["seed"] == wine.SEEDS[index % 3]
    json.dumps(run, allow_nan=False)


@pytest.mark.parametrize("index", range(6))
def test_initial_zero_bias_ratios_are_undefined_not_zero(report, index):
    params = report["runs"][index]["trace"][0]["parameters"]
    for name, values in params.items():
        assert values["gradient_present"]
        assert values["actual_delta_l2"] > 0
        if name.startswith("biases"):
            assert values["parameter_before_l2"] == 0
            assert values["update_to_parameter_ratio"] is None
            assert values["ratio_status"] == "undefined_zero_parameter_norm"
        else:
            assert values["update_to_parameter_ratio"] > 0


@pytest.mark.parametrize("index", range(6))
def test_epoch_weighted_loss_is_derived_from_actual_batches(report, index):
    run = report["runs"][index]
    for epoch in (1, 2):
        rows = [row for row in run["trace"] if row["epoch"] == epoch]
        expected = 0.0
        for r in rows:
            expected += r["pre_update_batch_cross_entropy"] * r["batch_size"]
        expected /= 107
        assert expected == run["history"][epoch - 1]["online_cross_entropy"]


def test_report_train_only_fixed_budget_and_checks(report):
    assert report["protocol"]["scored_partitions"] == ["train"]
    assert report["protocol"]["total_optimizer_updates"] == 252
    assert report["protocol"]["total_training_sample_visits"] == 3852
    assert report["protocol"]["prefix_only"]
    assert report["all_checks_passed"]
    assert len(report["preprocessing"]["row_ids"]) == 107


def test_measurement_uses_actual_delta_and_none_gradient():
    before = torch.tensor([3.0, 4.0], dtype=torch.float64)
    after = torch.tensor([3.0, 6.0], dtype=torch.float64)
    result = obs.observe_parameter(before, after, None, torch.zeros(2, dtype=torch.float64), 0.05)
    assert result["gradient_present"] is False
    assert result["gradient_l2"] is None
    assert result["parameter_before_l2"] == 5
    assert result["actual_delta_l2"] == 2
    assert result["update_to_parameter_ratio"] == 0.4
    assert not result["sgd_delta_matches"]
    assert torch.equal(before, torch.tensor([3.0, 4.0], dtype=torch.float64))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_observations_rejected(bad):
    with pytest.raises(ValueError, match="non-finite"):
        obs.observe_parameter(torch.ones(2), torch.tensor([bad, 1.0]), None, torch.zeros(2), 0.05)


def test_digest_distinguishes_dtype_shape_and_signed_zero():
    x = torch.tensor([0.0, 1.0], dtype=torch.float64)
    assert obs.tensor_digest(x) != obs.tensor_digest(x.reshape(1, 2))
    assert obs.tensor_digest(x) != obs.tensor_digest(x.float())
    y = x.clone()
    y[0] = -0.0
    assert obs.tensor_digest(x) != obs.tensor_digest(y)
    assert obs.tensor_digest(x) == obs.tensor_digest(x.clone())


def test_caller_rng_and_threads_unchanged():
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    obs.run_experiment()
    assert torch.equal(rng, torch.get_rng_state())
    assert threads == torch.get_num_threads()


def test_threads_restored_on_failure(monkeypatch):
    threads = torch.get_num_threads()

    def fail():
        raise RuntimeError("injected")

    monkeypatch.setattr(obs, "_run_experiment", fail)
    with pytest.raises(RuntimeError, match="injected"):
        obs.run_experiment()
    assert torch.get_num_threads() == threads


def test_no_heldout_scoring_or_selection(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("held-out scoring/selection forbidden")

    for name in ("metrics", "evaluate_classifier", "select_architecture", "run_wine_experiment"):
        monkeypatch.setattr(wine, name, forbidden)
    assert obs.run_experiment()["all_checks_passed"]


def test_heldout_mutation_cannot_change_training(report, monkeypatch):
    x, y = wine.load_wine()
    train = set(report["preprocessing"]["row_ids"])
    heldout = [i for i in range(178) if i not in train]
    changed_x, changed_y = x.clone(), y.clone()
    changed_x[heldout] = float("nan")
    changed_y[heldout] = -999
    monkeypatch.setattr(wine, "load_wine", lambda: (changed_x, changed_y))
    assert obs.run_experiment() == report


def test_optimizer_noop_is_observed_and_fails_delta_check(monkeypatch):
    monkeypatch.setattr(
        torch.optim.SGD,
        "step",
        lambda self: [
            self.state[p].update(momentum_buffer=torch.ones_like(p))
            for group in self.param_groups
            for p in group["params"]
        ],
    )
    result = obs.run_experiment()
    assert not result["all_checks_passed"]
    for run in result["runs"]:
        assert not run["checks"]["actual_updates_match_momentum"]
        assert all(
            p["actual_delta_l2"] == 0 for row in run["trace"] for p in row["parameters"].values()
        )


@pytest.mark.parametrize("value", [0, 1, None, "yes"])
def test_observe_requires_bool(value):
    with pytest.raises(ValueError, match="observe"):
        obs.train_prefix(torch.zeros(107, 13), torch.zeros(107), "linear", 42, observe=value)


def test_diagnostic_side_effect_detected(monkeypatch):
    original = obs.observe_parameter

    def corrupt(before, parameter, gradient, momentum, learning_rate):
        result = original(before, parameter, gradient, momentum, learning_rate)
        with torch.no_grad():
            parameter.add_(0.01)
        return result

    monkeypatch.setattr(obs, "observe_parameter", corrupt)
    result = obs.run_experiment()
    assert not result["all_checks_passed"]
    assert all(not run["checks"]["legacy_uninstrumented_final_equal"] for run in result["runs"])
    assert all(not run["checks"]["audit_trajectory_equal"] for run in result["runs"])


def test_repeated_report_is_identical(report):
    assert obs.run_experiment() == copy.deepcopy(report)
