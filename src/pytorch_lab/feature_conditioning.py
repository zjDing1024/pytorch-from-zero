"""Original equivalent-prediction scaling and diagonal SGD control experiment."""

import math
from dataclasses import asdict, dataclass

import torch

BASE_LR = 0.15
POLICIES = ("shared", "curvature", "mapped")


@dataclass(frozen=True)
class ConditioningConfig:
    scale: float = 1.0
    policy: str = "shared"
    steps: int = 12

    def validate(self):
        if (
            type(self.scale) is not float
            or not math.isfinite(self.scale)
            or not 0.25 <= self.scale <= 4
        ):
            raise ValueError("scale must be a finite float in [0.25, 4]")
        if type(self.policy) is not str or self.policy not in POLICIES:
            raise ValueError("unknown update policy")
        if type(self.steps) is not int or not 1 <= self.steps <= 16:
            raise ValueError("steps must be an integer in [1, 16]")


def geometry(config: ConditioningConfig) -> dict:
    if type(config) is not ConditioningConfig:
        raise ValueError("config must be ConditioningConfig")
    config.validate()
    eigenvalues = [config.scale**2, 2.0]
    if config.policy == "shared":
        rates = [BASE_LR, BASE_LR]
    elif config.policy == "curvature":
        rates = [0.5 / max(eigenvalues)] * 2
    else:
        rates = [BASE_LR / config.scale**2, BASE_LR]
    return {
        "hessian_eigenvalues": eigenvalues,
        "condition_number": max(eigenvalues) / min(eigenvalues),
        "scalar_stability_upper_exclusive": 2 / max(eigenvalues),
        "parameter_learning_rates": rates,
        "canonical_error_factors": [1 - rates[0] * config.scale**2, 1 - 2 * rates[1]],
    }


def analytic_state(config: ConditioningConfig, step: int) -> dict:
    info = geometry(config)
    if type(step) is not int or not 0 <= step <= 16:
        raise ValueError("oracle step must be an integer in [0, 16]")
    fw, fb = info["canonical_error_factors"]
    ew, eb = -2 * fw**step, -0.5 * fb**step
    return {"canonical_parameters": [2 + ew, 0.5 + eb], "loss": 0.5 * ew**2 + eb**2}


def _close(a, b):
    return math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12)


def run_case(config: ConditioningConfig | None = None) -> dict:
    config = ConditioningConfig() if config is None else config
    info = geometry(config)
    x = torch.tensor([-1, -0.5, 0, 0.5, 1], dtype=torch.float64)
    z, y = config.scale * x, 2 * x + 0.5
    theta = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    rates = info["parameter_learning_rates"]
    optimizer = torch.optim.SGD(
        [{"params": [theta], "lr": rates[0]}, {"params": [bias], "lr": rates[1]}]
    )
    traces = []
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        before = torch.stack([config.scale * theta.detach(), bias.detach()]).clone()
        loss = (theta * z + bias - y).square().mean()
        loss.backward()
        gradients = [theta.grad.item(), bias.grad.item()]
        optimizer.step()
        with torch.no_grad():
            canonical = torch.stack([config.scale * theta, bias])
            predictions = theta * z + bias
            loss_after = (predictions - y).square().mean().item()
            delta = canonical - before
            error = canonical - torch.tensor([2, 0.5], dtype=torch.float64)
            prediction_delta = delta[0] * x + delta[1]
            row = {
                "step": step + 1,
                "theta_bias": [theta.item(), bias.item()],
                "canonical_parameters": canonical.tolist(),
                "canonical_error": error.tolist(),
                "canonical_delta": delta.tolist(),
                "predictions": predictions.tolist(),
                "loss_before": loss.item(),
                "loss_after": loss_after,
                "prediction_rmse": math.sqrt(loss_after),
                "prediction_update_rms": prediction_delta.square().mean().sqrt().item(),
                "coordinate_gradients": gradients,
                "canonical_gradients": [gradients[0] / config.scale, gradients[1]],
            }
        values = (
            row["theta_bias"]
            + row["canonical_parameters"]
            + row["canonical_error"]
            + row["canonical_delta"]
            + row["predictions"]
            + gradients
            + row["canonical_gradients"]
            + [row["loss_before"], loss_after, row["prediction_rmse"], row["prediction_update_rms"]]
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite trajectory")
        row["oracle"] = analytic_state(config, step + 1)
        row["oracle_matches"] = all(
            _close(a, b)
            for a, b in zip(
                row["canonical_parameters"] + [loss_after],
                row["oracle"]["canonical_parameters"] + [row["oracle"]["loss"]],
                strict=True,
            )
        )
        traces.append(row)
    return {
        "config": asdict(config),
        "geometry": info,
        "trace": traces,
        "all_oracle_matches": all(row["oracle_matches"] for row in traces),
    }


def run_experiment() -> dict:
    cases = [
        run_case(ConditioningConfig(scale, policy))
        for scale in (0.25, 1.0, 4.0)
        for policy in POLICIES
    ]
    indexed = {(c["config"]["scale"], c["config"]["policy"]): c for c in cases}
    baseline = indexed[1.0, "shared"]["trace"]
    checks = {"all_analytic_trajectories": all(c["all_oracle_matches"] for c in cases)}
    for scale in (0.25, 1.0, 4.0):
        mapped = indexed[scale, "mapped"]["trace"]
        checks[f"mapped_prediction_equivalence_{scale}"] = all(
            _close(a, b)
            for row, ref in zip(mapped, baseline, strict=True)
            for a, b in zip(
                row["predictions"] + row["canonical_parameters"],
                ref["predictions"] + ref["canonical_parameters"],
                strict=True,
            )
        )
        checks[f"curvature_loss_decreases_{scale}"] = all(
            row["loss_after"] < row["loss_before"] for row in indexed[scale, "curvature"]["trace"]
        )
    checks["scale4_shared_growth"] = indexed[4.0, "shared"]["trace"][-1]["loss_after"] > 2.25
    checks["small_scale_shared_slower"] = (
        indexed[0.25, "shared"]["trace"][-1]["loss_after"] > baseline[-1]["loss_after"]
    )
    return {
        "format": "feature-conditioning-v1",
        "runtime": {"torch": str(torch.__version__), "device": "cpu", "dtype": "float64"},
        "fixture": {"x": [-1, -0.5, 0, 0.5, 1], "target": "2*x+0.5", "initial": [0, 0]},
        "steps_per_case": 12,
        "base_lr": BASE_LR,
        "oracle_tolerance": {"atol": 1e-12, "rtol": 1e-12},
        "cases": cases,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "limits": [
            "same function class does not make scalar SGD coordinate-invariant",
            "condition number depends on parameter coordinates",
            "fixed deterministic quadratic, not universal preprocessing advice",
            "no wall-time or generalization improvement claim",
        ],
    }
