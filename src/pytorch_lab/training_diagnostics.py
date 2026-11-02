"""Original fixed-batch training observations, not a root-cause classifier."""

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class DiagnosticConfig:
    steps: int = 12
    learning_rate: float = 0.1

    def validate(self):
        if type(self.steps) is not int or not 1 <= self.steps <= 100:
            raise ValueError("steps must be an integer in [1, 100]")
        if (
            type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or not 0 < self.learning_rate <= 0.25
        ):
            raise ValueError("learning_rate must be a finite float in (0, 0.25]")


def diagnose(*, present: int, total: int, gradient_norm: float, update_norm: float) -> str:
    """Classify measured symptoms only; no path name or optimizer action is input.

    Exact zero is an observation, not a tolerance-based convergence decision.
    Missing gradients do not rule out parameter mutation elsewhere.
    """
    if type(total) is not int or total < 1 or type(present) is not int or not 0 <= present <= total:
        raise ValueError("invalid gradient presence counts")
    for value in (gradient_norm, update_norm):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("norms must be finite nonnegative numbers")
    if present == 0 and gradient_norm != 0:
        raise ValueError("absent gradients cannot have a nonzero norm")
    if present == 0:
        return "missing_gradients_with_update" if update_norm else "missing_gradients_no_update"
    if present < total:
        return "partial_gradients_with_update" if update_norm else "partial_gradients_no_update"
    if gradient_norm == 0:
        return "zero_gradients_with_update" if update_norm else "zero_gradients_no_update"
    return "gradient_and_update_observed" if update_norm else "gradient_without_update"


def _norm(values):
    result = torch.linalg.vector_norm(torch.cat([v.detach().reshape(-1) for v in values])).item()
    if not math.isfinite(result):
        raise ValueError("non-finite measured norm")
    return result


def run_path(path: str, config: DiagnosticConfig | None = None) -> dict:
    """Run one controlled intervention on the same deterministic five-row fixture."""
    if path not in ("correct", "omit_backward", "omit_step"):
        raise ValueError("unknown controlled path")
    config = DiagnosticConfig() if config is None else config
    if type(config) is not DiagnosticConfig:
        raise ValueError("config must be DiagnosticConfig")
    config.validate()
    x = torch.tensor([-1, -0.5, 0, 0.5, 1], dtype=torch.float64)
    y = 2 * x + 0.5
    weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    parameters = (weight, bias)
    optimizer = torch.optim.SGD(parameters, lr=config.learning_rate)
    traces = []
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        before = tuple(p.detach().clone() for p in parameters)
        loss = ((weight * x + bias - y) ** 2).mean()
        if path != "omit_backward":
            loss.backward()
        gradients = tuple(None if p.grad is None else p.grad.detach().clone() for p in parameters)
        present = sum(g is not None for g in gradients)
        gradient_norm = _norm([g for g in gradients if g is not None]) if present else 0.0
        if path != "omit_step":
            optimizer.step()
        delta = tuple(p.detach() - old for p, old in zip(parameters, before, strict=True))
        update_norm = _norm(delta)
        with torch.no_grad():
            after_loss = ((weight * x + bias - y) ** 2).mean().item()
        row = {
            "step": step,
            "loss_before": loss.item(),
            "loss_after": after_loss,
            "parameters_before": [p.item() for p in before],
            "parameters_after": [p.item() for p in parameters],
            "gradients": [None if g is None else g.item() for g in gradients],
            "gradient_present": present,
            "gradient_total": len(parameters),
            "gradient_norm": gradient_norm,
            "update_norm": update_norm,
            "parameter_delta": [d.item() for d in delta],
            "parameter_norm_before": _norm(before),
            "parameter_norm_after": _norm(parameters),
        }
        if not math.isfinite(row["loss_before"]) or not math.isfinite(after_loss):
            raise ValueError("non-finite loss")
        row["diagnosis"] = diagnose(
            present=present,
            total=len(parameters),
            gradient_norm=gradient_norm,
            update_norm=update_norm,
        )
        traces.append(row)
    return {"traces": traces, "diagnostics": [r["diagnosis"] for r in traces]}


def run_experiment() -> dict:
    """Execute the frozen 12-step contract, retaining every observed measurement."""
    config = DiagnosticConfig()
    paths = {name: run_path(name, config) for name in ("correct", "omit_backward", "omit_step")}
    expected = {
        "correct": "gradient_and_update_observed",
        "omit_backward": "missing_gradients_no_update",
        "omit_step": "gradient_without_update",
    }
    checks = {}
    for name, result in paths.items():
        rows = result["traces"]
        checks[name + "_symptoms"] = all(r["diagnosis"] == expected[name] for r in rows)
        checks[name + "_loss"] = all(
            r["loss_after"] < r["loss_before"]
            if name == "correct"
            else r["loss_after"] == r["loss_before"]
            for r in rows
        )
    return {
        "format": "fixed-batch-training-diagnostics-v1",
        "config": asdict(config),
        "fixture": {"x": [-1, -0.5, 0, 0.5, 1], "y": [-1.5, -0.5, 0.5, 1.5, 2.5]},
        "runtime": {"torch": str(torch.__version__), "device": "cpu", "dtype": "float64"},
        "paths": paths,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "limits": [
            "controlled fault names are experimental labels, not inferred root causes",
            "zero gradient differs from absent gradient; "
            "nonzero gradient need not produce an update",
            "no arbitrary-model convergence, classification accuracy or performance claim",
        ],
    }
