"""Independent equations and observation checks for feature reparameterization."""

import math

import pytest
import torch

from pytorch_lab.feature_conditioning import ConditioningConfig, geometry, run_case


@pytest.mark.parametrize("scale", [0.25, 0.5, 1.0, 1.3, 2.0, 4.0])
@pytest.mark.parametrize("policy", ["shared", "curvature", "mapped"])
def test_independent_closed_form_and_gradients(scale, policy):
    if policy == "shared":
        lt, lb = 0.15, 0.15
    elif policy == "curvature":
        lt = lb = 0.5 / max(scale * scale, 2)
    else:
        lt, lb = 0.15 / (scale * scale), 0.15
    result = run_case(ConditioningConfig(scale, policy, 12))
    previous = [0.0, 0.0]
    for k, row in enumerate(result["trace"], 1):
        a = 2 - 2 * (1 - lt * scale * scale) ** k
        b = 0.5 - 0.5 * (1 - 2 * lb) ** k
        loss = 0.5 * (a - 2) ** 2 + (b - 0.5) ** 2
        delta = [a - previous[0], b - previous[1]]
        expected_grad = [scale * (previous[0] - 2), 2 * (previous[1] - 0.5)]
        assert row["canonical_parameters"] == pytest.approx([a, b], abs=1e-12, rel=1e-12)
        assert row["theta_bias"] == pytest.approx([a / scale, b], abs=1e-12, rel=1e-12)
        assert row["coordinate_gradients"] == pytest.approx(expected_grad, abs=1e-12, rel=1e-12)
        assert row["canonical_gradients"] == pytest.approx(
            [previous[0] - 2, 2 * (previous[1] - 0.5)], abs=1e-12, rel=1e-12
        )
        assert row["loss_after"] == pytest.approx(loss, abs=1e-12, rel=1e-12)
        assert row["canonical_delta"] == pytest.approx(delta, abs=1e-12, rel=1e-12)
        assert row["prediction_update_rms"] == pytest.approx(
            math.sqrt(0.5 * delta[0] ** 2 + delta[1] ** 2), abs=1e-12, rel=1e-12
        )
        assert row["predictions"] == pytest.approx(
            [a * x + b for x in [-1, -0.5, 0, 0.5, 1]], abs=1e-12, rel=1e-12
        )
        previous = [a, b]
    assert result["all_oracle_matches"]


@pytest.mark.parametrize("scale", [0.25, 1.0, 4.0])
def test_autograd_hessian_from_original_samples(scale):
    x = torch.tensor([-1, -0.5, 0, 0.5, 1], dtype=torch.float64)
    y = 2 * x + 0.5
    point = torch.tensor([0.7, -0.2], dtype=torch.float64, requires_grad=True)
    h = torch.autograd.functional.hessian(
        lambda p: ((p[0] * (scale * x) + p[1] - y) ** 2).mean(), point
    )
    expected = torch.diag(torch.tensor([scale**2, 2.0], dtype=torch.float64))
    torch.testing.assert_close(h, expected, atol=1e-12, rtol=1e-12)
    info = geometry(ConditioningConfig(scale))
    assert info["condition_number"] == max(scale**2, 2) / min(scale**2, 2)
    assert info["scalar_stability_upper_exclusive"] == 2 / max(scale**2, 2)


def test_mapped_policy_preserves_functions_not_coordinate_norms():
    baseline = run_case(ConditioningConfig(1.0, "shared"))["trace"]
    for scale in (0.25, 1.3, 4.0):
        mapped = run_case(ConditioningConfig(scale, "mapped"))["trace"]
        for got, ref in zip(mapped, baseline, strict=True):
            assert got["predictions"] == pytest.approx(ref["predictions"], abs=1e-12, rel=1e-12)
            assert got["canonical_parameters"] == pytest.approx(
                ref["canonical_parameters"], abs=1e-12, rel=1e-12
            )
            assert got["prediction_update_rms"] == pytest.approx(
                ref["prediction_update_rms"], abs=1e-12, rel=1e-12
            )
        assert mapped[-1]["theta_bias"][0] != baseline[-1]["theta_bias"][0]


def test_equal_initial_function_does_not_make_shared_sgd_invariant():
    first = [run_case(ConditioningConfig(s, "shared", 1))["trace"][0] for s in (0.25, 1.0, 4.0)]
    assert all(r["loss_before"] == 2.25 for r in first)
    assert len({tuple(r["canonical_parameters"]) for r in first}) == 3
    # Small coordinates can worsen conditioning; larger is not invariably better.
    assert (
        geometry(ConditioningConfig(0.25))["condition_number"]
        > geometry(ConditioningConfig(1.0))["condition_number"]
    )


def test_real_optimizer_noop_is_measured_not_replaced_with_oracle(monkeypatch):
    monkeypatch.setattr(torch.optim.SGD, "step", lambda self, closure=None: None)
    result = run_case(ConditioningConfig(1.0, "mapped", 3))
    assert not result["all_oracle_matches"]
    assert all(r["canonical_parameters"] == [0, 0] for r in result["trace"])
    assert all(r["prediction_update_rms"] == 0 for r in result["trace"])
    assert all(r["loss_after"] == 2.25 for r in result["trace"])


def test_nonfinite_real_optimizer_mutation_rejected(monkeypatch):
    def bad_step(self, closure=None):
        with torch.no_grad():
            self.param_groups[0]["params"][0].fill_(float("inf"))

    monkeypatch.setattr(torch.optim.SGD, "step", bad_step)
    with pytest.raises(ValueError, match="non-finite trajectory"):
        run_case()


def test_repeatability_and_caller_rng_unchanged():
    before = torch.get_rng_state().clone()
    a = run_case(ConditioningConfig(1.3, "mapped", 5))
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
    assert a == run_case(ConditioningConfig(1.3, "mapped", 5))
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), 0.0, -1.0, 4.01, True, 1])
def test_invalid_scale_rejected_without_rng_change(scale):
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        run_case(ConditioningConfig(scale))
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
