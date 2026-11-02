import math

import pytest
import torch

from pytorch_lab.training_diagnostics import DiagnosticConfig, diagnose, run_experiment, run_path


@pytest.mark.parametrize("path", ["correct", "omit_backward", "omit_step"])
@pytest.mark.parametrize("lr", [0.01, 0.1, 0.25])
def test_scalar_equations(path, lr):
    config = DiagnosticConfig(steps=12, learning_rate=lr)
    rows = run_path(path, config)["traces"]
    w = b = 0.0
    for row in rows:
        # Symmetric x has mean zero and mean square 1/2.
        expected_loss = (w - 2) ** 2 / 2 + (b - 0.5) ** 2
        gradients = [w - 2, 2 * (b - 0.5)]
        assert row["loss_before"] == pytest.approx(expected_loss, rel=1e-12, abs=1e-12)
        if path == "omit_backward":
            assert row["gradients"] == [None, None]
        else:
            assert row["gradients"] == pytest.approx(gradients, rel=1e-12, abs=1e-12)
        if path == "correct":
            w -= lr * gradients[0]
            b -= lr * gradients[1]
        assert row["parameters_after"] == pytest.approx([w, b], rel=1e-12, abs=1e-12)
        assert row["loss_after"] == pytest.approx(
            (w - 2) ** 2 / 2 + (b - 0.5) ** 2, rel=1e-12, abs=1e-12
        )


@pytest.mark.parametrize(
    "present,gradient,update,expected",
    [
        (0, 0, 0, "missing_gradients_no_update"),
        (0, 0, 1, "missing_gradients_with_update"),
        (1, 0, 0, "partial_gradients_no_update"),
        (1, 1, 1, "partial_gradients_with_update"),
        (2, 0, 0, "zero_gradients_no_update"),
        (2, 0, 1, "zero_gradients_with_update"),
        (2, 1, 0, "gradient_without_update"),
        (2, 1, 1, "gradient_and_update_observed"),
    ],
)
def test_observation_only_states(present, gradient, update, expected):
    assert (
        diagnose(present=present, total=2, gradient_norm=gradient, update_norm=update) == expected
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"present": True},
        {"present": -1},
        {"present": 3},
        {"total": 0},
        {"total": 2.0},
        {"gradient_norm": math.nan},
        {"update_norm": math.inf},
        {"gradient_norm": -1},
        {"update_norm": True},
        {"present": 0, "gradient_norm": 1},
    ],
)
def test_invalid_observations(kwargs):
    values = dict(present=2, total=2, gradient_norm=1.0, update_norm=0.1)
    values.update(kwargs)
    with pytest.raises(ValueError):
        diagnose(**values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"steps": 101},
        {"steps": True},
        {"steps": 1.0},
        {"learning_rate": 0.0},
        {"learning_rate": 1},
        {"learning_rate": math.nan},
        {"learning_rate": math.inf},
        {"learning_rate": 0.5},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        run_path("correct", DiagnosticConfig(**kwargs))


def test_experiment_is_repeatable_and_rng_neutral():
    before = torch.get_rng_state().clone()
    a = run_experiment()
    assert torch.equal(before, torch.get_rng_state())
    torch.rand(10)
    b = run_experiment()
    assert a == b
    assert a["all_checks_passed"]
    assert sum(len(v["traces"]) for v in a["paths"].values()) == 36
    assert a["paths"]["correct"]["traces"][-1]["loss_after"] < 2.25


def test_bad_path_and_config():
    with pytest.raises(ValueError):
        run_path("unknown")
    with pytest.raises(ValueError):
        run_path("correct", {})
