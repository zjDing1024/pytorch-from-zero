"""Original CPU float64 affine/SGD gradient accumulation with element weighting.

The sum of each microbatch's squared errors is differentiated immediately. At an
update boundary gradients are divided by the ACTUAL number of target elements.
Only one microbatch graph is live. This is not an AMP/DDP/stochastic-layer loop.
"""

import math
from collections.abc import Iterable

import torch
from torch import Tensor

from pytorch_lab.minibatch import AffineRegressor


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_model_optimizer(model: AffineRegressor, optimizer: torch.optim.SGD) -> None:
    # Restrict the layer contract rather than imply equivalence for BatchNorm,
    # Dropout, or arbitrary models whose forward depends on microbatch size.
    if type(model) is not AffineRegressor:
        raise ValueError(
            "model must be an AffineRegressor without stochastic/batch-dependent layers"
        )
    if not isinstance(optimizer, torch.optim.SGD):
        raise ValueError("optimizer must be torch.optim.SGD")
    params = list(model.parameters())
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len(params) or {id(p) for p in actual} != {id(p) for p in params}:
        raise ValueError("optimizer must own each model parameter exactly once")
    for p in params:
        if p.device.type != "cpu" or p.dtype != torch.float64 or not p.requires_grad:
            raise ValueError("parameters must be trainable CPU float64 tensors")
        if not torch.isfinite(p).all().item():
            raise ValueError("model parameters must be finite")


def train_accumulated_epoch(
    model: AffineRegressor,
    batches: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.SGD,
    accumulation_steps: int,
) -> dict:
    """Train one epoch, retaining short microbatches and flushing a short window.

    ``accumulation_steps`` counts microbatches, not samples. Loss is mean squared
    error over N*K target elements; fixed output width K makes this equivalent
    to sample-weighted per-output MSE. Momentum/weight decay are applied once per
    effective update, not once per microbatch. No scheduler/clipping is invoked.

    Valid inputs are nonempty CPU float64 [N,D]/[N,K] tensor pairs. The stream is
    not materialized. Exceptions clear pending gradients but do NOT roll back
    optimizer updates already committed, or a failing optimizer's partial work.
    The caller must reload a known checkpoint after a partially completed epoch.
    Gradients are cleared on success, empty input, and iteration/backward errors.
    """
    _positive_int(accumulation_steps, "accumulation_steps")
    _validate_model_optimizer(model, optimizer)
    model.train()
    params = list(model.parameters())
    total_loss = 0.0
    samples = elements = microbatches = 0
    window_loss = 0.0
    window_samples = window_elements = window_batches = 0
    updates = []
    optimizer.zero_grad(set_to_none=True)

    def flush() -> None:
        nonlocal window_loss, window_samples, window_elements, window_batches
        for p in params:
            if p.grad is None or not torch.isfinite(p.grad).all().item():
                raise ValueError("accumulated gradient is missing or non-finite")
            p.grad.div_(window_elements)
        optimizer.step()
        if not all(torch.isfinite(p).all().item() for p in params):
            raise ValueError("parameters became non-finite after update")
        updates.append(
            {
                "samples": window_samples,
                "target_elements": window_elements,
                "microbatches": window_batches,
                "pre_update_mse": window_loss / window_elements,
            }
        )
        optimizer.zero_grad(set_to_none=True)
        window_loss = 0.0
        window_samples = window_elements = window_batches = 0

    try:
        for features, targets in batches:
            if not isinstance(features, Tensor) or not isinstance(targets, Tensor):
                raise ValueError("features and targets must be tensors")
            if features.ndim != 2 or targets.ndim != 2 or 0 in targets.shape:
                raise ValueError("batches must have nonempty [N,D] and [N,K] shapes")
            if targets.dtype != torch.float64 or targets.device.type != "cpu":
                raise ValueError("targets must be CPU float64")
            if not torch.isfinite(targets).all().item():
                raise ValueError("targets must be finite")
            prediction = model(features)
            if prediction.shape != targets.shape:
                raise ValueError("prediction and target shape must match; broadcasting forbidden")
            loss_sum = (prediction - targets).square().sum()
            if not torch.isfinite(loss_sum).item():
                raise ValueError("sum squared error became non-finite")
            loss_sum.backward()
            n, count = targets.shape[0], targets.numel()
            value = loss_sum.item()
            total_loss += value
            window_loss += value
            if not math.isfinite(total_loss) or not math.isfinite(window_loss):
                raise ValueError("accumulated loss became non-finite")
            samples += n
            elements += count
            microbatches += 1
            window_samples += n
            window_elements += count
            window_batches += 1
            if window_batches == accumulation_steps:
                flush()
        if window_batches:
            flush()
        if elements == 0:
            raise ValueError("batches produced no samples")
    finally:
        optimizer.zero_grad(set_to_none=True)
    return {
        "online_mse": total_loss / elements,
        "samples_seen": samples,
        "target_elements": elements,
        "microbatches": microbatches,
        "optimizer_steps": len(updates),
        "updates": updates,
    }
