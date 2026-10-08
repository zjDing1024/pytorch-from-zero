"""Reproducible assertions and machine-readable evidence, not a speed benchmark."""

import platform
from datetime import UTC, datetime

import torch

from pytorch_lab.regression import (
    TrainingConfig,
    affine_mse,
    analytical_gradients,
    finite_difference_gradients,
    run_training,
)
from pytorch_lab.tensors import inspect_accumulation, inspect_broadcast_gradient, inspect_layouts


def compare_gradients() -> dict:
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(5, 3, dtype=torch.float64, generator=generator)
    target = torch.randn(5, 2, dtype=torch.float64, generator=generator)
    weight = torch.randn(3, 2, dtype=torch.float64, generator=generator, requires_grad=True)
    bias = torch.randn(2, dtype=torch.float64, generator=generator, requires_grad=True)
    loss = affine_mse(x, target, weight, bias)
    automatic = torch.autograd.grad(loss, (weight, bias))
    analytical = analytical_gradients(x, target, weight, bias)
    numerical = finite_difference_gradients(x, target, weight, bias)

    def max_gap(left: tuple, right: tuple) -> float:
        return max((a - b).abs().max().item() for a, b in zip(left, right, strict=True))

    return {
        "seed": 17,
        "shapes": {"x": [5, 3], "target": [5, 2], "weight": [3, 2], "bias": [2]},
        "loss": loss.item(),
        "autograd_vs_analytical_max_abs_error": max_gap(automatic, analytical),
        "autograd_vs_finite_difference_max_abs_error": max_gap(automatic, numerical),
        "central_difference_epsilon": 1e-6,
        "torch_gradcheck_passed": torch.autograd.gradcheck(
            lambda w, b: affine_mse(x, target, w, b), (weight, bias)
        ),
    }


def run_all(config: TrainingConfig | None = None) -> dict:
    config = TrainingConfig() if config is None else config
    layouts = inspect_layouts()
    broadcasting = inspect_broadcast_gradient()
    accumulation = inspect_accumulation()
    gradients = compare_gradients()
    training = run_training(config)
    checks = {
        "transpose_aliases": layouts["mutation_visible_in_base"],
        "incompatible_view_rejected": layouts["view_flatten_rejected"],
        "reshape_copied_in_this_case": layouts["reshape_copy_in_this_case"],
        "broadcast_backward_sums": broadcasting["bias_gradient"]
        == broadcasting["expected_gradient"],
        "gradients_accumulate_and_reset": accumulation
        == {"first": 6.0, "without_reset": 12.0, "after_reset": 6.0},
        "analytical_matches_autograd": gradients["autograd_vs_analytical_max_abs_error"] < 1e-12,
        "finite_difference_matches_autograd": (
            gradients["autograd_vs_finite_difference_max_abs_error"] < 1e-7
        ),
        "torch_gradcheck": gradients["torch_gradcheck_passed"],
        "training_improves": training["final_train_mse"] < training["initial_train_mse"],
        "holdout_beats_mean_baseline": training["test_mse"] < training["test_mean_baseline_mse"],
    }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "executed_by": "AI assistant; learner self-check remains pending",
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "pytorch_git_version": torch.version.git_version,
        },
        "layouts": layouts,
        "broadcasting": broadcasting,
        "gradient_accumulation": accumulation,
        "gradient_comparison": gradients,
        "training": training,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
