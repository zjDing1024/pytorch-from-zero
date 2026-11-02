"""Independent scalar equations and observation-only diagnosis checks."""

import inspect
import math

import pytest
import torch

from pytorch_lab.training_diagnostics import DiagnosticConfig, diagnose, run_experiment, run_path


@pytest.mark.parametrize("lr", [0.01, 0.1, 0.25])
@pytest.mark.parametrize("path", ["correct", "omit_backward", "omit_step"])
def test_scalar_closed_form_oracle(lr, path):
    rows = run_path(path, DiagnosticConfig(steps=12, learning_rate=lr))["traces"]
    for k, row in enumerate(rows):
        w = 2 * (1 - (1 - lr) ** k) if path == "correct" else 0.0
        b = 0.5 * (1 - (1 - 2 * lr) ** k) if path == "correct" else 0.0
        gw, gb = w - 2, 2 * b - 1
        dw, db = (-lr * gw, -lr * gb) if path == "correct" else (0.0, 0.0)
        assert row["parameters_before"] == pytest.approx([w, b], abs=1e-12, rel=1e-12)
        assert row["parameters_after"] == pytest.approx([w + dw, b + db], abs=1e-12, rel=1e-12)
        assert row["parameter_delta"] == pytest.approx([dw, db], abs=1e-12, rel=1e-12)
        assert row["loss_before"] == pytest.approx(
            0.5 * (w - 2) ** 2 + (b - 0.5) ** 2, abs=1e-12, rel=1e-12
        )
        assert row["loss_after"] == pytest.approx(
            0.5 * (w + dw - 2) ** 2 + (b + db - 0.5) ** 2, abs=1e-12, rel=1e-12
        )
        expected_grad = [None, None] if path == "omit_backward" else [gw, gb]
        if path == "omit_backward":
            assert row["gradients"] == expected_grad
        else:
            assert row["gradients"] == pytest.approx(expected_grad, abs=1e-12, rel=1e-12)
        assert row["gradient_norm"] == pytest.approx(
            0 if path == "omit_backward" else math.hypot(gw, gb), abs=1e-12, rel=1e-12
        )
        assert row["update_norm"] == pytest.approx(math.hypot(dw, db), abs=1e-12, rel=1e-12)


def test_diagnosis_signature_has_no_fault_label():
    assert set(inspect.signature(diagnose).parameters) == {
        "present",
        "total",
        "gradient_norm",
        "update_norm",
    }


def test_actual_parameter_change_not_optimizer_call_or_path_label(monkeypatch):
    monkeypatch.setattr(torch.optim.SGD, "step", lambda self: None)
    rows = run_path("correct")["traces"]
    assert all(r["diagnosis"] == "gradient_without_update" for r in rows)
    assert all(r["update_norm"] == 0 for r in rows)
    assert all(r["gradients"] == [-2.0, -1.0] for r in rows)


def test_mutation_without_gradient_is_observed(monkeypatch):
    def mutate(self):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    p.add_(0.125)

    monkeypatch.setattr(torch.optim.SGD, "step", mutate)
    rows = run_path("omit_backward")["traces"]
    assert all(r["diagnosis"] == "missing_gradients_with_update" for r in rows)
    assert all(r["parameter_delta"] == [0.125, 0.125] for r in rows)


def test_default_rng_unchanged_and_exact_repeat():
    before = torch.get_rng_state().clone()
    first = run_experiment()
    assert torch.equal(before, torch.get_rng_state())
    assert first == run_experiment()
    assert first["all_checks_passed"]


def test_saved_measurements_are_not_later_parameter_aliases():
    rows = run_path("correct")["traces"]
    assert rows[0]["parameters_before"] == [0.0, 0.0]
    assert rows[0]["parameters_after"] == pytest.approx([0.2, 0.1])
    assert rows[0]["parameters_after"] != rows[-1]["parameters_after"]
    assert rows[0]["gradients"] == [-2.0, -1.0]


@pytest.mark.parametrize(
    "present,grad,update,label",
    [
        (0, 0.0, 0.0, "missing_gradients_no_update"),
        (0, 0.0, 1.0, "missing_gradients_with_update"),
        (1, 0.0, 0.0, "partial_gradients_no_update"),
        (1, 2.0, 1.0, "partial_gradients_with_update"),
        (2, 0.0, 0.0, "zero_gradients_no_update"),
        (2, 0.0, 1.0, "zero_gradients_with_update"),
        (2, 2.0, 0.0, "gradient_without_update"),
        (2, 2.0, 1.0, "gradient_and_update_observed"),
    ],
)
def test_observational_classes(present, grad, update, label):
    assert diagnose(present=present, total=2, gradient_norm=grad, update_norm=update) == label


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True, "1"])
def test_invalid_norms_rejected(bad):
    with pytest.raises(ValueError):
        diagnose(present=2, total=2, gradient_norm=bad, update_norm=0.0)
    with pytest.raises(ValueError):
        diagnose(present=2, total=2, gradient_norm=1.0, update_norm=bad)


def test_missing_gradients_with_positive_norm_is_invalid():
    with pytest.raises(ValueError):
        diagnose(present=0, total=2, gradient_norm=1.0, update_norm=0.0)
