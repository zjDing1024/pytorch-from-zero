"""An original Module/Dataset/DataLoader lab, retaining the handwritten baseline.

CPU-only, single-worker, float32/float64 teaching code. The experiment itself uses
float64. No preprocessing, model selection, or fitting ever sees holdout data.
"""

import math
import platform
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from pytorch_lab.regression import TrainingConfig, run_training


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _seed(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")


def _finite_number(value: float, name: str, *, allow_zero: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    ):
        bound = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {bound} and finite")


def _matrix(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a Tensor")
    if value.ndim != 2 or 0 in value.shape:
        raise ValueError(f"{name} must have nonempty shape [N, D] or [N, K]")
    if value.device.type != "cpu" or value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use CPU floating-point float32 or float64")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")


class RegressionDataset(Dataset):
    """Map-style paired samples, snapshotted independently of caller storage/graphs."""

    def __init__(self, features: Tensor, targets: Tensor) -> None:
        _matrix(features, "features")
        _matrix(targets, "targets")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must have the same sample count")
        if features.dtype != targets.dtype:
            raise ValueError("features and targets must have the same dtype")
        self.features = features.detach().clone()
        self.targets = targets.detach().clone()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.features[index], self.targets[index]


class AffineRegressor(nn.Module):
    """Registered W[D,K] and b[K] keep the original mathematical orientation.

    Zero initialization is suitable for this single affine layer, not generally
    for multilayer networks. It also avoids changing the caller's global RNG.
    """

    def __init__(
        self, in_features: int, out_features: int, dtype: torch.dtype = torch.float64
    ) -> None:
        super().__init__()
        _positive_integer(in_features, "in_features")
        _positive_integer(out_features, "out_features")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("model dtype must be floating-point float32 or float64")
        self.weight = nn.Parameter(torch.zeros(in_features, out_features, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))

    def forward(self, features: Tensor) -> Tensor:
        _matrix(features, "features")
        if features.shape[1] != self.weight.shape[0]:
            raise ValueError("features have the wrong input dimension")
        if features.dtype != self.weight.dtype or features.device != self.weight.device:
            raise ValueError("features and model must have the same dtype and device")
        if not all(torch.isfinite(p).all().item() for p in self.parameters()):
            raise ValueError("model parameters became non-finite")
        prediction = features @ self.weight + self.bias
        if not torch.isfinite(prediction).all().item():
            raise ValueError("prediction became non-finite")
        return prediction


def make_loader(
    dataset: RegressionDataset, batch_size: int, seed: int = 42, shuffle: bool = False
) -> DataLoader:
    """Even sequential loaders get a generator: iterator base seeds consume RNG."""
    _positive_integer(batch_size, "batch_size")
    _seed(seed)
    if not isinstance(shuffle, bool):
        raise ValueError("shuffle must be a bool")
    if len(dataset) < 1:
        raise ValueError("dataset must be nonempty")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )


def _batch_loss(model: nn.Module, features: Tensor, targets: Tensor) -> Tensor:
    _matrix(targets, "targets")
    prediction = model(features)
    if prediction.shape != targets.shape:
        raise ValueError("prediction and targets must have equal shape; broadcasting is forbidden")
    if prediction.dtype != targets.dtype or prediction.device != targets.device:
        raise ValueError("prediction and targets must have the same dtype and device")
    loss = (prediction - targets).square().mean()
    if not torch.isfinite(loss).item():
        raise ValueError("loss became non-finite; check input scale and learning rate")
    return loss


def _mean(total: float, samples: int) -> float:
    if samples == 0:
        raise ValueError("loader produced no samples")
    if not math.isfinite(total):
        raise ValueError("accumulated loss became non-finite")
    return total / samples


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Sample-weighted mean, with a fixed model; restore the caller's train/eval mode.

    All batches must have the same output width K. For this paired dataset that
    makes sum(batch_MSE * batch_N) / sum(batch_N) equal the all-element MSE.
    """
    was_training = model.training
    total, samples = 0.0, 0
    model.eval()
    try:
        for features, targets in loader:
            loss = _batch_loss(model, features, targets)
            batch_size = features.shape[0]
            total += loss.item() * batch_size
            samples += batch_size
        return _mean(total, samples)
    finally:
        model.train(was_training)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> dict:
    """Online loss uses each batch's pre-update parameters; it is not final-model MSE."""
    model.train()
    total, samples, batches = 0.0, 0, 0
    for features, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = _batch_loss(model, features, targets)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            optimizer.zero_grad(set_to_none=True)
            raise ValueError("gradient became non-finite")
        optimizer.step()
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            optimizer.zero_grad(set_to_none=True)
            raise ValueError("model parameters became non-finite after update")
        total += loss.item() * features.shape[0]
        samples += features.shape[0]
        batches += 1
    optimizer.zero_grad(set_to_none=True)
    return {"mse": _mean(total, samples), "samples_seen": samples, "batches": batches}


@dataclass(frozen=True)
class MinibatchConfig:
    seed: int = 42
    epochs: int = 40
    batch_size: int = 20
    learning_rate: float = 0.05
    noise_std: float = 0.05

    def validate(self) -> None:
        _seed(self.seed)
        _positive_integer(self.epochs, "epochs")
        _positive_integer(self.batch_size, "batch_size")
        _finite_number(self.learning_rate, "learning_rate")
        _finite_number(self.noise_std, "noise_std", allow_zero=True)


def make_splits(seed: int = 42, noise_std: float = 0.05) -> dict[str, RegressionDataset]:
    """Same data-generating process and fixed slices as the preserved first lab."""
    _seed(seed)
    _finite_number(noise_std, "noise_std", allow_zero=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features = torch.randn(160, 3, generator=generator, dtype=torch.float64)
    weight = torch.tensor([[1.75], [-2.0], [0.5]], dtype=torch.float64)
    bias = torch.tensor([-0.3], dtype=torch.float64)
    noise = torch.randn(160, 1, generator=generator, dtype=torch.float64)
    targets = features @ weight + bias + noise_std * noise
    return {
        "train": RegressionDataset(features[:96], targets[:96]),
        "validation": RegressionDataset(features[96:128], targets[96:128]),
        "test": RegressionDataset(features[128:], targets[128:]),
    }


def fit_minibatches(
    train_dataset: RegressionDataset, config: MinibatchConfig
) -> tuple[AffineRegressor, list[dict]]:
    """Only training data is accepted: no validation/test-driven hyperparameters."""
    config.validate()
    model = AffineRegressor(
        train_dataset.features.shape[1],
        train_dataset.targets.shape[1],
        train_dataset.features.dtype,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    # Independent loaders prevent metric evaluation from advancing shuffle state.
    train_loader = make_loader(train_dataset, config.batch_size, seed=config.seed, shuffle=True)
    metric_loader = make_loader(train_dataset, config.batch_size, seed=config.seed)
    history = [{"epoch": 0, "train_mse": evaluate(model, metric_loader)}]
    for epoch in range(1, config.epochs + 1):
        online = train_one_epoch(model, train_loader, optimizer)
        history.append(
            {
                "epoch": epoch,
                "train_mse": evaluate(model, metric_loader),
                "online_train_mse": online["mse"],
                "samples_seen": online["samples_seen"],
                "batches": online["batches"],
            }
        )
    return model, history


def run_minibatch_experiment(config: MinibatchConfig | None = None) -> dict:
    """Record real mini/full-batch runs, honest work counts, and numerical checks."""
    config = MinibatchConfig() if config is None else config
    config.validate()
    splits = make_splits(config.seed, config.noise_std)
    model, history = fit_minibatches(splits["train"], config)
    holdout = {
        name: evaluate(model, make_loader(splits[name], config.batch_size, seed=config.seed))
        for name in ("validation", "test")
    }
    # A constant baseline must be fitted on training targets only (per output).
    train_mean = splits["train"].targets.mean(dim=0)
    mean_baseline = (splits["test"].targets - train_mean).square().mean().item()
    reference = run_training(
        TrainingConfig(config.seed, config.epochs, config.learning_rate, config.noise_std)
    )
    full_model, full_history = fit_minibatches(
        splits["train"], replace(config, batch_size=len(splits["train"]))
    )
    reference_weight = torch.tensor(reference["learned_weight"], dtype=torch.float64).reshape(3, 1)
    equivalence_gap = max(
        (full_model.weight.detach() - reference_weight).abs().max().item(),
        abs(full_model.bias.item() - reference["learned_bias"]),
    )
    training = {
        "split_sizes": {name: len(dataset) for name, dataset in splits.items()},
        "initial_train_mse": history[0]["train_mse"],
        "final_train_mse": history[-1]["train_mse"],
        "validation_mse": holdout["validation"],
        "test_mse": holdout["test"],
        "test_mean_baseline_mse": mean_baseline,
        "learned_weight": model.weight.detach().flatten().tolist(),
        "learned_bias": model.bias.item(),
        "optimizer_steps": sum(row["batches"] for row in history[1:]),
        "training_sample_visits": sum(row["samples_seen"] for row in history[1:]),
        "last_batch_size": (len(splits["train"]) - 1) % config.batch_size + 1,
        "registered_parameters": {name: list(p.shape) for name, p in model.named_parameters()},
        "history": history,
    }
    checks = {
        "training_improves": training["final_train_mse"] < training["initial_train_mse"],
        "holdout_beats_train_mean": training["test_mse"] < mean_baseline,
        "every_epoch_covers_all_training_samples": all(
            row["samples_seen"] == len(splits["train"]) for row in history[1:]
        ),
        "module_full_batch_matches_handwritten": equivalence_gap < 1e-12,
        "parameters_are_registered": list(dict(model.named_parameters())) == ["weight", "bias"],
    }
    return {
        "schema_version": 1,
        "experiment": "module_dataset_minibatch",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "executed_by": "AI assistant; learner self-check remains pending",
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float64",
            "num_workers": 0,
            "pytorch_git_version": torch.version.git_version,
        },
        "config": asdict(config),
        "training": training,
        "comparison": {
            "protocol": "Same data, zero initialization, learning rate, and training data passes; "
            "optimizer steps differ. Not a throughput or equal-update-budget comparison.",
            "handwritten_full_batch": reference,
            "module_full_batch_final_train_mse": full_history[-1]["train_mse"],
            "module_full_batch_vs_handwritten_max_parameter_gap": equivalence_gap,
            "full_batch_optimizer_steps": config.epochs,
            "full_batch_training_sample_visits": config.epochs * len(splits["train"]),
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
