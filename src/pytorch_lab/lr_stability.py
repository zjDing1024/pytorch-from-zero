"""Original bounded affine learning-rate experiment with a closed-form oracle."""

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class StabilityConfig:
    learning_rate: float = 0.1
    steps: int = 12

    def validate(self):
        if type(self.steps) is not int or not 1 <= self.steps <= 32:
            raise ValueError("steps must be an integer in [1, 32]")
        if (
            type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or not 0 <= self.learning_rate <= 1.5
        ):
            raise ValueError("learning_rate must be a finite float in [0, 1.5]")


def spectral_regime(learning_rate: float) -> str:
    """Worst-case error dynamics for this Hessian only, not observed convergence."""
    StabilityConfig(learning_rate).validate()
    if learning_rate == 0:
        return "stationary"
    if learning_rate < 1:
        return "contracting"
    if learning_rate == 1:
        return "critical"
    return "unstable"


def analytic_state(learning_rate: float, step: int) -> dict:
    StabilityConfig(learning_rate).validate()
    if type(step) is not int or not 0 <= step <= 32:
        raise ValueError("oracle step must be an integer in [0, 32]")
    ew = -2 * (1 - learning_rate) ** step
    eb = -0.5 * (1 - 2 * learning_rate) ** step
    return {"parameters": [2 + ew, 0.5 + eb], "error": [ew, eb], "loss": 0.5 * ew**2 + eb**2}


def _finite(values):
    if not all(math.isfinite(v) for v in values):
        raise ValueError("non-finite trajectory; bounded experiment rejected")


def run_rate(config: StabilityConfig | None = None) -> dict:
    config = StabilityConfig() if config is None else config
    if type(config) is not StabilityConfig:
        raise ValueError("config must be StabilityConfig")
    config.validate()
    x = torch.tensor([-1, -0.5, 0, 0.5, 1], dtype=torch.float64)
    y = 2 * x + 0.5
    weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    params = (weight, bias)
    optimizer = torch.optim.SGD(params, lr=config.learning_rate)
    trace = []
    for step in range(config.steps):
        before = torch.stack([p.detach().clone() for p in params])
        optimizer.zero_grad(set_to_none=True)
        loss = (weight * x + bias - y).square().mean()
        loss.backward()
        gradient = torch.stack([p.grad.detach().clone() for p in params])
        optimizer.step()
        after = torch.stack([p.detach().clone() for p in params])
        delta = after - before
        error = after - torch.tensor([2, 0.5], dtype=torch.float64)
        with torch.no_grad():
            loss_after = (weight * x + bias - y).square().mean().item()
        oracle = analytic_state(config.learning_rate, step + 1)
        row = {
            "step": step + 1,
            "parameters_before": before.tolist(),
            "parameters_after": after.tolist(),
            "parameter_error": error.tolist(),
            "loss_before": loss.item(),
            "loss_after": loss_after,
            "gradient": gradient.tolist(),
            "gradient_norm": torch.linalg.vector_norm(gradient).item(),
            "parameter_delta": delta.tolist(),
            "update_norm": torch.linalg.vector_norm(delta).item(),
            "error_norm": torch.linalg.vector_norm(error).item(),
            "oracle": oracle,
        }
        _finite(
            before.tolist()
            + after.tolist()
            + gradient.tolist()
            + delta.tolist()
            + [
                row["loss_before"],
                loss_after,
                row["gradient_norm"],
                row["update_norm"],
                row["error_norm"],
            ]
        )
        row["oracle_matches"] = all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
            for a, b in zip(
                after.tolist() + [loss_after], oracle["parameters"] + [oracle["loss"]], strict=True
            )
        )
        trace.append(row)
    return {
        "config": asdict(config),
        "spectral_regime": spectral_regime(config.learning_rate),
        "error_factors": [1 - config.learning_rate, 1 - 2 * config.learning_rate],
        "trace": trace,
        "all_oracle_matches": all(r["oracle_matches"] for r in trace),
    }


def run_experiment() -> dict:
    rates = [0.1, 0.75, 1.0, 1.25]
    cases = [run_rate(StabilityConfig(rate)) for rate in rates]
    stable, oscillating, critical, unstable = [c["trace"] for c in cases]
    checks = {
        "all_analytic_trajectories": all(c["all_oracle_matches"] for c in cases),
        "stable_loss_decreases": all(r["loss_after"] < r["loss_before"] for r in stable),
        "oscillating_loss_decreases": all(r["loss_after"] < r["loss_before"] for r in oscillating),
        "stable_bias_alternates": all(
            r["parameter_error"][1] * s["parameter_error"][1] < 0
            for r, s in zip(oscillating, oscillating[1:], strict=False)
        ),
        "critical_bias_two_cycle": all(
            abs(r["parameter_error"][1]) == 0.5 and r["loss_after"] == 0.25 for r in critical
        ),
        "unstable_bias_amplitude_grows": all(
            abs(s["parameter_error"][1]) > abs(r["parameter_error"][1])
            for r, s in zip(unstable, unstable[1:], strict=False)
        ),
        "unstable_final_loss_exceeds_initial": unstable[-1]["loss_after"] > 2.25,
    }
    return {
        "format": "affine-lr-stability-v1",
        "runtime": {"torch": str(torch.__version__), "device": "cpu", "dtype": "float64"},
        "fixture": {"x": [-1, -0.5, 0, 0.5, 1], "target": "2*x+0.5", "initial": [0, 0]},
        "hessian_eigenvalues": [1.0, 2.0],
        "oracle_tolerance": {"atol": 1e-12, "rtol": 1e-12},
        "cases": cases,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "limits": [
            "fixed quadratic only",
            "finite 12-step trajectories, not an infinite-time numerical proof",
            "unstable spectral direction need not be excited by every initial condition",
            "no universal learning-rate or performance recommendation",
        ],
    }
