"""Original, functional CPU float64 SGD arithmetic and an independent torch oracle.

The update below is written from the equations in docs/momentum-sgd.md. It is
deliberately not an Optimizer subclass and does not call torch.optim internally.
Only the separate reference experiment uses torch.optim.SGD.
"""

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def _nonnegative(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite nonnegative Python number")


def _tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a Tensor")
    if value.is_nested or value.layout != torch.strided:
        raise ValueError(f"{name} must be a dense strided Tensor")
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise ValueError(f"{name} must use CPU float64")
    if value.numel() == 0 or not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be nonempty and finite")


@torch.no_grad()
def momentum_sgd_step(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor | None],
    buffers: Sequence[Tensor | None],
    *,
    learning_rate: float,
    momentum: float,
    dampening: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    maximize: bool = False,
) -> tuple[list[Tensor], list[Tensor | None]]:
    """Return detached parameter/buffer snapshots without changing caller state.

    Lists/tuples have equal, nonzero length; parameter shapes can differ. Each
    parameter owns distinct storage. Gradients/buffers match its shape exactly.
    Scalar and noncontiguous tensors are supported. None gradients skip *all*
    arithmetic for that parameter, including decay and buffer initialization.

    The contract is narrower than torch: CPU float64, finite dense values,
    scalar Python hyperparameters, dampening in [0, 1], no autograd through the
    update, groups, closure, sparse, foreach, fused, AMP, or shared parameters.
    With momentum=0 an existing buffer is retained but not read or updated.
    """
    for name, value in (
        ("learning_rate", learning_rate),
        ("momentum", momentum),
        ("dampening", dampening),
        ("weight_decay", weight_decay),
    ):
        _nonnegative(value, name)
    if dampening > 1:
        raise ValueError("dampening must be in [0, 1]")
    if not isinstance(nesterov, bool) or not isinstance(maximize, bool):
        raise ValueError("nesterov and maximize must be bools")
    if nesterov and (momentum == 0 or dampening != 0):
        raise ValueError("nesterov requires positive momentum and zero dampening")
    if not all(isinstance(values, (list, tuple)) for values in (parameters, gradients, buffers)):
        raise ValueError("parameters, gradients, and buffers must be lists or tuples")
    if not parameters or len(parameters) != len(gradients) or len(parameters) != len(buffers):
        raise ValueError("parameters, gradients, and buffers must have equal nonzero lengths")

    storage_ids = set()
    for index, (parameter, gradient, buffer) in enumerate(
        zip(parameters, gradients, buffers, strict=True)
    ):
        _tensor(parameter, f"parameter[{index}]")
        storage_id = parameter.untyped_storage().data_ptr()
        if storage_id in storage_ids:
            raise ValueError("parameters must own distinct storage; shared views are unsupported")
        storage_ids.add(storage_id)
        for name, value in (("gradient", gradient), ("buffer", buffer)):
            if value is not None:
                _tensor(value, f"{name}[{index}]")
                if value.shape != parameter.shape:
                    raise ValueError(f"{name}[{index}] must match its parameter shape exactly")

    next_parameters, next_buffers = [], []
    for parameter, gradient, buffer in zip(parameters, gradients, buffers, strict=True):
        new_parameter = parameter.detach().clone()
        new_buffer = None if buffer is None else buffer.detach().clone()
        if gradient is not None:
            # Descent-form direction: maximize negates the loss gradient only.
            direction = (-gradient if maximize else gradient) + weight_decay * parameter
            _tensor(direction, "decay-adjusted gradient")
            if momentum > 0:
                # The first *observed gradient for this parameter* is undampened.
                new_buffer = (
                    direction.clone()
                    if buffer is None
                    else momentum * buffer + (1.0 - dampening) * direction
                )
                _tensor(new_buffer, "updated buffer")
                direction = direction + momentum * new_buffer if nesterov else new_buffer
                _tensor(direction, "update direction")
            new_parameter = parameter - learning_rate * direction
            _tensor(new_parameter, "updated parameter")
        next_parameters.append(new_parameter)
        next_buffers.append(new_buffer)
    return next_parameters, next_buffers


def run_momentum_reference_trace() -> dict:
    """Compare every parameter and buffer after every step with torch's loop SGD.

    Deterministic prescribed gradients isolate update semantics from autograd.
    No data fitting, RNG use, or performance claim is involved. All differences
    are absolute and compared with 1e-12; buffer-presence equality is separate.
    """
    rates = [0.2, 0.2, 0.05, 0.0, 0.025, 0.025]
    cases = {
        "plain_sgd": {"momentum": 0.0},
        "momentum": {"momentum": 0.8},
        "dampening_and_decay": {"momentum": 0.8, "dampening": 0.3, "weight_decay": 0.1},
        "nesterov": {"momentum": 0.8, "nesterov": True, "weight_decay": 0.1},
        "maximize_and_decay": {"momentum": 0.8, "maximize": True, "weight_decay": 0.1},
        "maximize_nesterov": {
            "momentum": 0.8,
            "maximize": True,
            "nesterov": True,
            "weight_decay": 0.1,
        },
    }
    initial = [
        torch.tensor([[1.0, -2.0], [0.5, 0.25]], dtype=torch.float64),
        torch.tensor([0.3, -0.7], dtype=torch.float64),
    ]
    traces = []
    for name, options in cases.items():
        manual = [value.clone() for value in initial]
        buffers = [None, None]
        reference = [value.clone().requires_grad_() for value in initial]
        oracle = torch.optim.SGD(reference, lr=rates[0], foreach=False, fused=False, **options)
        steps = []
        for step, rate in enumerate(rates, start=1):
            gradients = [
                torch.full_like(initial[0], 0.15 * step - 0.4),
                torch.tensor([0.2 * step, -0.1 * step], dtype=torch.float64),
            ]
            # Bias joins late. Later, each parameter skips a different step.
            if step == 1 or step == 5:
                gradients[1] = None
            if step == 3:
                gradients[0] = None
            if step == 4:
                gradients[1] = torch.zeros_like(initial[1])
            if step == 6:
                gradients[0] = torch.zeros_like(initial[0])
            manual, buffers = momentum_sgd_step(
                manual, gradients, buffers, learning_rate=rate, **options
            )
            oracle.param_groups[0]["lr"] = rate
            for parameter, gradient in zip(reference, gradients, strict=True):
                parameter.grad = None if gradient is None else gradient.clone()
            oracle.step()
            per_parameter = []
            for index, parameter in enumerate(reference):
                ref_buffer = oracle.state.get(parameter, {}).get("momentum_buffer")
                buffer = buffers[index]
                presence_matches = (ref_buffer is None) == (buffer is None)
                buffer_gap = (
                    (buffer - ref_buffer).abs().max().item()
                    if buffer is not None and ref_buffer is not None
                    else None
                )
                per_parameter.append(
                    {
                        "name": ("weight", "bias")[index],
                        "gradient_present": gradients[index] is not None,
                        "parameter_max_abs_difference": (manual[index] - parameter.detach())
                        .abs()
                        .max()
                        .item(),
                        "buffer_present": buffer is not None,
                        "reference_buffer_present": ref_buffer is not None,
                        "buffer_presence_matches": presence_matches,
                        "buffer_max_abs_difference": buffer_gap,
                    }
                )
            steps.append({"step": step, "learning_rate": rate, "parameters": per_parameter})
        traces.append({"name": name, "options": options, "steps": steps})
    rows = [row for case in traces for step in case["steps"] for row in step["parameters"]]
    parameter_gap = max(row["parameter_max_abs_difference"] for row in rows)
    buffer_gap = max(row["buffer_max_abs_difference"] or 0.0 for row in rows)
    checks = {
        "every_parameter_matches": parameter_gap <= 1e-12,
        "every_buffer_matches": buffer_gap <= 1e-12,
        "every_buffer_presence_matches": all(row["buffer_presence_matches"] for row in rows),
    }
    return {
        "reference": "torch.optim.SGD(foreach=False, fused=False, differentiable=False)",
        "pytorch": torch.__version__,
        "pytorch_git_version": torch.version.git_version,
        "device": "cpu",
        "dtype": "float64",
        "absolute_tolerance": 1e-12,
        "max_parameter_abs_difference": parameter_gap,
        "max_buffer_abs_difference": buffer_gap,
        "cases": traces,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
