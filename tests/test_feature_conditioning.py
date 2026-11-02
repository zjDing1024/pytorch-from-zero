import math

import pytest
import torch

from pytorch_lab.feature_conditioning import (
    ConditioningConfig,
    analytic_state,
    geometry,
    run_case,
    run_experiment,
)


@pytest.mark.parametrize("scale", [0, 1, True, None, float("nan"), float("inf"), 0.24, 4.01])
def test_reject_scale(scale):
    with pytest.raises(ValueError):
        run_case(ConditioningConfig(scale=scale))


@pytest.mark.parametrize("steps", [0, 17, True, 1.0, None])
def test_reject_steps(steps):
    with pytest.raises(ValueError):
        run_case(ConditioningConfig(steps=steps))


@pytest.mark.parametrize("policy", ["", "adam", None, 1, ["shared"]])
def test_reject_policy(policy):
    with pytest.raises(ValueError):
        run_case(ConditioningConfig(policy=policy))


@pytest.mark.parametrize("config", [{}, False, "shared"])
def test_reject_config(config):
    with pytest.raises(ValueError):
        run_case(config)


@pytest.mark.parametrize("step", [-1, 17, True, 0.0])
def test_reject_oracle_step(step):
    with pytest.raises(ValueError):
        analytic_state(ConditioningConfig(), step)


@pytest.mark.parametrize("scale", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("policy", ["shared", "curvature", "mapped"])
def test_trace_oracle_and_geometry(scale, policy):
    cfg = ConditioningConfig(scale, policy)
    result = run_case(cfg)
    assert len(result["trace"]) == 12
    assert result["all_oracle_matches"]
    info = result["geometry"]
    assert info["hessian_eigenvalues"] == [scale**2, 2.0]
    assert info["condition_number"] == max(scale**2, 2) / min(scale**2, 2)
    assert analytic_state(cfg, 0) == {"canonical_parameters": [0.0, 0.0], "loss": 2.25}
    previous = [0.0, 0.0]
    for row in result["trace"]:
        assert row["canonical_parameters"][0] == pytest.approx(scale * row["theta_bias"][0])
        assert row["canonical_gradients"] == pytest.approx(
            [previous[0] - 2, 2 * (previous[1] - 0.5)]
        )
        assert row["coordinate_gradients"][0] == pytest.approx(scale * (previous[0] - 2))
        ew, eb = row["canonical_error"]
        assert row["loss_after"] == pytest.approx(0.5 * ew**2 + eb**2)
        dw, db = row["canonical_delta"]
        assert row["prediction_update_rms"] == pytest.approx(math.sqrt(0.5 * dw**2 + db**2))
        assert row["prediction_rmse"] ** 2 == pytest.approx(row["loss_after"])
        previous = row["canonical_parameters"]


def test_predeclared_experiment():
    result = run_experiment()
    assert result["all_checks_passed"]
    assert len(result["cases"]) == 9
    assert len(result["checks"]) == 9
    assert all(result["checks"].values())
    assert result["steps_per_case"] == 12


def test_mapped_metric_not_scalar_stability():
    info = geometry(ConditioningConfig(0.25, "mapped"))
    assert info["parameter_learning_rates"][0] > info["scalar_stability_upper_exclusive"]
    assert info["canonical_error_factors"] == pytest.approx([0.85, 0.7])


def test_rng_and_result_independence():
    before = torch.random.get_rng_state().clone()
    first = run_case()
    second = run_case()
    assert torch.equal(torch.random.get_rng_state(), before)
    assert first == second
    first["trace"][0]["predictions"][0] = 99
    assert second["trace"][0]["predictions"][0] != 99


def test_wrong_optimizer_is_detected(monkeypatch):
    monkeypatch.setattr(torch.optim.SGD, "step", lambda self: None)
    assert not run_case()["all_oracle_matches"]


def test_nonfinite_optimizer_rejected(monkeypatch):
    def broken(self):
        with torch.no_grad():
            self.param_groups[0]["params"][0].fill_(float("nan"))

    monkeypatch.setattr(torch.optim.SGD, "step", broken)
    with pytest.raises(ValueError, match="non-finite"):
        run_case()
