import math

import pytest
import torch

from pytorch_lab.lr_stability import (
    StabilityConfig,
    analytic_state,
    run_experiment,
    run_rate,
    spectral_regime,
)


@pytest.mark.parametrize(
    "rate,regime",
    [
        (0.0, "stationary"),
        (0.1, "contracting"),
        (0.75, "contracting"),
        (1.0, "critical"),
        (1.25, "unstable"),
        (1.5, "unstable"),
    ],
)
def test_regime(rate, regime):
    assert spectral_regime(rate) == regime


@pytest.mark.parametrize("rate", [0.0, 0.1, 0.5, 0.75, 1.0, 1.25, 1.5])
@pytest.mark.parametrize("steps", [1, 12, 32])
def test_trajectory(rate, steps):
    torch.manual_seed(912)
    before = torch.get_rng_state().clone()
    result = run_rate(StabilityConfig(rate, steps))
    assert torch.equal(before, torch.get_rng_state())
    assert len(result["trace"]) == steps
    if (rate, steps) == (1.5, 32):
        # Finite stress trajectory is valid, but exact-arithmetic oracle is not promised.
        assert result["all_oracle_matches"] == all(r["oracle_matches"] for r in result["trace"])
        assert all(math.isfinite(r["loss_after"]) for r in result["trace"])
        return
    assert result["all_oracle_matches"]
    for row in result["trace"]:
        oracle = analytic_state(rate, row["step"])
        assert row["parameters_after"] == pytest.approx(oracle["parameters"], abs=1e-12, rel=1e-12)
        assert row["loss_after"] == pytest.approx(oracle["loss"], abs=1e-12, rel=1e-12)
        assert row["gradient"][0] == pytest.approx(
            row["parameters_before"][0] - 2, abs=1e-12, rel=1e-12
        )
        assert row["gradient"][1] == pytest.approx(
            2 * (row["parameters_before"][1] - 0.5), abs=1e-12, rel=1e-12
        )
        assert row["parameter_delta"] == pytest.approx(
            [-rate * v for v in row["gradient"]], abs=1e-12, rel=1e-12
        )


@pytest.mark.parametrize("rate", [-0.1, 1.5001, float("nan"), float("inf"), True, 1, "0.1", None])
def test_bad_rate(rate):
    with pytest.raises(ValueError):
        run_rate(StabilityConfig(rate))


@pytest.mark.parametrize("steps", [0, 33, 1.0, True, "12", None])
def test_bad_steps(steps):
    with pytest.raises(ValueError):
        run_rate(StabilityConfig(0.1, steps))


@pytest.mark.parametrize("step", [-1, 33, 1.0, True])
def test_bad_oracle_step(step):
    with pytest.raises(ValueError):
        analytic_state(0.1, step)


def test_none_default_and_bad_config():
    assert run_rate() == run_rate(StabilityConfig())
    with pytest.raises(ValueError):
        run_rate({})


def test_full_experiment_and_zero_rate():
    result = run_experiment()
    assert result["all_checks_passed"]
    assert len(result["cases"]) == 4
    assert sum(len(c["trace"]) for c in result["cases"]) == 48
    stationary = run_rate(StabilityConfig(0.0))
    assert all(r["update_norm"] == 0 and r["loss_after"] == 2.25 for r in stationary["trace"])
    assert all(r["gradient_norm"] > 0 for r in stationary["trace"])
    unstable = result["cases"][-1]["trace"]
    assert unstable[0]["loss_after"] < 2.25
    assert unstable[-1]["loss_after"] > 2.25
    assert math.isfinite(unstable[-1]["loss_after"])


def test_output_owns_measurements():
    result = run_rate()
    first = result["trace"][0]["parameters_after"].copy()
    result["trace"][-1]["parameters_after"][0] = 999
    assert result["trace"][0]["parameters_after"] == first
