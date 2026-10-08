"""Affine MSE: explicit shape contracts, analytical gradients, and a tiny optimizer."""

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor


def _validate(x: Tensor, target: Tensor, weight: Tensor, bias: Tensor) -> None:
    tensors = (x, target, weight, bias)
    if not all(t.is_floating_point() for t in tensors):
        raise ValueError("all inputs must use floating-point dtypes")
    if len({t.dtype for t in tensors}) != 1 or len({t.device for t in tensors}) != 1:
        raise ValueError("all inputs must have the same dtype and device")
    if x.ndim != 2 or weight.ndim != 2 or target.ndim != 2 or bias.ndim != 1:
        raise ValueError("expected x[N,D], target[N,K], weight[D,K], bias[K]")
    n, d = x.shape
    k = weight.shape[1]
    if n == 0 or d == 0 or k == 0:
        raise ValueError("N, D, and K must be nonzero")
    if weight.shape[0] != d or target.shape != (n, k) or bias.shape != (k,):
        raise ValueError("shape mismatch; implicit target broadcasting is forbidden")
    if not all(torch.isfinite(t).all().item() for t in tensors):
        raise ValueError("all inputs must be finite")


def affine_mse(x: Tensor, target: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Mean over all N*K entries, not just batch size. Preserves the autograd graph."""
    _validate(x, target, weight, bias)
    loss = (x @ weight + bias - target).square().mean()
    if not torch.isfinite(loss).item():
        raise ValueError("loss became non-finite; check input scale and learning rate")
    return loss


def analytical_gradients(
    x: Tensor, target: Tensor, weight: Tensor, bias: Tensor
) -> tuple[Tensor, Tensor]:
    """Closed-form first derivatives with respect to weight and bias."""
    _validate(x, target, weight, bias)
    with torch.no_grad():
        residual = x @ weight + bias - target
        d_prediction = 2.0 * residual / residual.numel()
        return x.T @ d_prediction, d_prediction.sum(dim=0)


def finite_difference_gradients(
    x: Tensor, target: Tensor, weight: Tensor, bias: Tensor, epsilon: float = 1e-6
) -> tuple[Tensor, Tensor]:
    """Central differences on independent clones. Intended for tiny float64 checks."""
    _validate(x, target, weight, bias)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if x.dtype != torch.float64:
        raise ValueError("finite-difference checking requires float64 for this experiment")
    parameters = [p.detach().clone(memory_format=torch.contiguous_format) for p in (weight, bias)]
    gradients = [torch.empty_like(p) for p in parameters]
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients, strict=True):
            flat = parameter.reshape(-1)
            flat_gradient = gradient.reshape(-1)
            for i in range(parameter.numel()):
                original = flat[i].item()
                flat[i] = original + epsilon
                plus = affine_mse(x, target, *parameters).item()
                flat[i] = original - epsilon
                minus = affine_mse(x, target, *parameters).item()
                flat[i] = original
                flat_gradient[i] = (plus - minus) / (2.0 * epsilon)
    return gradients[0], gradients[1]


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    steps: int = 200
    learning_rate: float = 0.1
    noise_std: float = 0.05

    def validate(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("seed must be an integer in [0, 2**63)")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be nonnegative and finite")


def run_training(config: TrainingConfig | None = None) -> dict:
    """Fit a known noisy synthetic process on CPU; report untouched holdout metrics.

    Hyperparameters are fixed beforehand. There is no test-driven model selection.
    A local Generator isolates this experiment from the caller's RNG state.
    """
    config = TrainingConfig() if config is None else config
    config.validate()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    x = torch.randn(160, 3, generator=generator, dtype=torch.float64)
    true_weight = torch.tensor([[1.75], [-2.0], [0.5]], dtype=torch.float64)
    true_bias = torch.tensor([-0.3], dtype=torch.float64)
    noise = torch.randn(160, 1, generator=generator, dtype=torch.float64)
    target = x @ true_weight + true_bias + config.noise_std * noise
    train_x, train_y = x[:96], target[:96]
    validation_x, validation_y = x[96:128], target[96:128]
    test_x, test_y = x[128:], target[128:]
    weight = torch.zeros(3, 1, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    initial = affine_mse(train_x, train_y, weight, bias).item()
    history = [{"step": 0, "train_mse": initial}]
    for step in range(1, config.steps + 1):
        loss = affine_mse(train_x, train_y, weight, bias)
        loss.backward()
        with torch.no_grad():
            weight -= config.learning_rate * weight.grad
            bias -= config.learning_rate * bias.grad
        weight.grad = None
        bias.grad = None
        if step % 20 == 0 or step == config.steps:
            history.append(
                {"step": step, "train_mse": affine_mse(train_x, train_y, weight, bias).item()}
            )
    with torch.no_grad():
        augmented = torch.cat([train_x, torch.ones(96, 1, dtype=torch.float64)], dim=1)
        reference = torch.linalg.lstsq(augmented, train_y).solution
        reference_train_mse = (augmented @ reference - train_y).square().mean().item()
        train_mean = train_y.mean()
        final_train = affine_mse(train_x, train_y, weight, bias).item()
        validation = affine_mse(validation_x, validation_y, weight, bias).item()
        test = affine_mse(test_x, test_y, weight, bias).item()
    return {
        "config": asdict(config),
        "device": "cpu",
        "dtype": "float64",
        "split_sizes": {"train": 96, "validation": 32, "test": 32},
        "initial_train_mse": initial,
        "final_train_mse": final_train,
        "validation_mse": validation,
        "test_mse": test,
        "test_mean_baseline_mse": (test_y - train_mean).square().mean().item(),
        "least_squares_train_mse": reference_train_mse,
        "gradient_descent_lstsq_max_parameter_gap": (
            torch.cat([weight.detach(), bias.detach().reshape(1, 1)]) - reference
        )
        .abs()
        .max()
        .item(),
        "learned_weight": weight.detach().flatten().tolist(),
        "learned_bias": bias.item(),
        "true_weight": true_weight.flatten().tolist(),
        "true_bias": true_bias.item(),
        "history": history,
    }
