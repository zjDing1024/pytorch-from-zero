"""Independent scalar recurrence and observed-dynamics checks."""

import math

import pytest
import torch

from pytorch_lab.lr_stability import StabilityConfig, analytic_state, run_experiment, run_rate


@pytest.mark.parametrize("rate", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
def test_scalar_closed_form_and_gradient(rate):
    result = run_rate(StabilityConfig(rate, 12))
    for k, row in enumerate(result["trace"]):
        ew, eb = -2 * (1 - rate) ** k, -0.5 * (1 - 2 * rate) ** k
        expected = {
            "parameters_before": [2 + ew, 0.5 + eb],
            "gradient": [ew, 2 * eb],
            "parameter_delta": [-rate * ew, -2 * rate * eb],
            "parameters_after": [2 + (1 - rate) * ew, 0.5 + (1 - 2 * rate) * eb],
            "parameter_error": [(1 - rate) * ew, (1 - 2 * rate) * eb],
        }
        for key, values in expected.items():
            assert row[key] == pytest.approx(values, rel=1e-12, abs=1e-12)
        assert row["loss_before"] == pytest.approx(0.5 * ew**2 + eb**2, rel=1e-12, abs=1e-12)
        assert row["loss_after"] == pytest.approx(
            0.5 * ((1 - rate) * ew) ** 2 + ((1 - 2 * rate) * eb) ** 2, rel=1e-12, abs=1e-12
        )
        assert row["gradient_norm"] == pytest.approx(math.hypot(ew, 2 * eb), rel=1e-12, abs=1e-12)
        assert row["update_norm"] == pytest.approx(
            math.hypot(rate * ew, 2 * rate * eb), rel=1e-12, abs=1e-12
        )


def test_critical_loss_falls_first_then_plateaus():
    rows = run_rate(StabilityConfig(1.0))["trace"]
    assert rows[0]["loss_before"] == 2.25
    assert rows[0]["loss_after"] == 0.25
    assert all(row["parameters_after"][0] == 2 for row in rows)
    assert [row["parameters_after"][1] for row in rows] == [1.0, 0.0] * 6
    assert all(row["loss_after"] == 0.25 and row["update_norm"] == 1 for row in rows[1:])


def test_unstable_is_not_monotone_loss_from_initial():
    rows = run_rate(StabilityConfig(1.25))["trace"]
    assert rows[0]["loss_after"] < rows[0]["loss_before"]
    assert rows[-1]["loss_after"] > rows[0]["loss_before"]
    assert all(
        abs(b["parameter_error"][1]) > abs(a["parameter_error"][1])
        for a, b in zip(rows, rows[1:], strict=False)
    )


def test_stationary_has_nonzero_gradient_but_zero_update():
    rows = run_rate(StabilityConfig(0.0))["trace"]
    assert all(
        row["gradient_norm"] > 0 and row["update_norm"] == 0 and row["loss_after"] == 2.25
        for row in rows
    )


def test_maximum_steps_and_rng_unchanged():
    before = torch.get_rng_state().clone()
    first = run_rate(StabilityConfig(0.75, 32))
    assert first == run_rate(StabilityConfig(0.75, 32))
    assert len(first["trace"]) == 32
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("rate", [True, 1, -0.1, 1.6, float("nan"), float("inf"), "0.1"])
def test_bad_rate_rejected(rate):
    with pytest.raises(ValueError):
        run_rate(StabilityConfig(rate))


@pytest.mark.parametrize("step", [True, -1, 33, 1.5])
def test_bad_oracle_step_rejected(step):
    with pytest.raises(ValueError):
        analytic_state(0.1, step)


def test_raw_trace_is_not_silently_replaced_with_oracle(monkeypatch):
    original = torch.optim.SGD.step

    def no_update(self, *args, **kwargs):
        return None

    monkeypatch.setattr(torch.optim.SGD, "step", no_update)
    report = run_experiment()
    assert not report["all_checks_passed"]
    assert not report["cases"][0]["all_oracle_matches"]
    assert report["cases"][0]["trace"][0]["parameters_after"] == [0, 0]
    assert report["cases"][0]["trace"][0]["oracle"]["parameters"] != [0, 0]
    monkeypatch.setattr(torch.optim.SGD, "step", original)


def test_nonfinite_parameter_mutation_is_rejected(monkeypatch):
    def corrupt(self, *args, **kwargs):
        with torch.no_grad():
            self.param_groups[0]["params"][0].fill_(float("inf"))

    monkeypatch.setattr(torch.optim.SGD, "step", corrupt)
    with pytest.raises(ValueError, match="non-finite"):
        run_rate()


def test_finite_stress_configuration_reports_measured_oracle_status():
    # Exploratory stress, separate from the frozen four-rate 12-step gate.
    # Cancellation in the large bias trajectory can break small weight tolerance.
    report = run_rate(StabilityConfig(1.5, 32))
    assert len(report["trace"]) == 32
    assert all(math.isfinite(v) for row in report["trace"] for v in row["parameters_after"])
    expected = []
    for k, row in enumerate(report["trace"], 1):
        ew, eb = -2 * (-0.5) ** k, -0.5 * (-2.0) ** k
        reference = [2 + ew, 0.5 + eb, 0.5 * ew**2 + eb**2]
        matches = all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
            for a, b in zip(row["parameters_after"] + [row["loss_after"]], reference, strict=True)
        )
        assert row["oracle_matches"] == matches
        expected.append(matches)
    assert report["all_oracle_matches"] == all(expected)
    assert run_experiment()["all_checks_passed"]
