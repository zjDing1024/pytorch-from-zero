"""Offline, leakage-aware Wine classification with a predeclared evaluation protocol.

Original teaching implementation. Dataset bytes are redistributed with attribution,
not claimed as original code. CPU float64 only; no general training-framework API.
"""

import csv
import hashlib
import io
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.resources import files

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

DATA_SHA256 = "10e8a802908b34f86e5da8ce962f3c806694bc98450a18f61851af59f324bede"
SPLIT_SALT = "wine-split-v1:20261009"
SEEDS = (42, 7, 123)
ARCHITECTURES = ("linear", "mlp16")
FEATURE_NAMES = (
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_of_diluted_wines",
    "proline",
)


def _features(x: Tensor) -> None:
    if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("features must be a nonempty matrix")
    if x.device.type != "cpu" or x.dtype != torch.float64 or not torch.isfinite(x).all():
        raise ValueError("features must be finite CPU float64")


def _labels(y: Tensor, n: int) -> None:
    if not isinstance(y, Tensor) or y.shape != (n,) or y.dtype != torch.int64:
        raise ValueError("targets must be int64 with shape [N]")
    if y.device.type != "cpu" or not ((y >= 0) & (y < 3)).all():
        raise ValueError("targets must be CPU class indices in [0, 3)")


def load_wine() -> tuple[Tensor, Tensor]:
    """Verify pinned raw bytes before parsing; never fetch data during execution."""
    payload = files("pytorch_lab").joinpath("data/wine.csv").read_bytes()
    if hashlib.sha256(payload).hexdigest() != DATA_SHA256:
        raise ValueError("Wine CSV integrity check failed")
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8"))))
    if rows.pop(0) != ["178", "13", "class_0", "class_1", "class_2"]:
        raise ValueError("unexpected Wine CSV header")
    if len(rows) != 178 or any(len(row) != 14 for row in rows):
        raise ValueError("unexpected Wine CSV dimensions")
    x = torch.tensor([[float(v) for v in row[:13]] for row in rows], dtype=torch.float64)
    y = torch.tensor([int(row[13]) for row in rows], dtype=torch.int64)
    _features(x)
    _labels(y, len(x))
    if torch.bincount(y, minlength=3).tolist() != [59, 71, 48]:
        raise ValueError("unexpected Wine class counts")
    return x, y


def build_split(y: Tensor) -> dict[str, list[int]]:
    """Hash-rank stable zero-based source row IDs within class, with fixed rounding."""
    _labels(y, 178)
    split = {name: [] for name in ("train", "validation", "test")}
    for label in range(3):
        ids = [i for i, value in enumerate(y.tolist()) if value == label]
        ids.sort(key=lambda i: hashlib.sha256(f"{SPLIT_SALT}:{label}:{i}".encode()).hexdigest())
        train_end = math.floor(0.6 * len(ids) + 0.5)
        val_end = train_end + math.floor(0.2 * len(ids) + 0.5)
        for name, selected in zip(
            split, (ids[:train_end], ids[train_end:val_end], ids[val_end:]), strict=True
        ):
            split[name].extend(selected)
    return {name: sorted(ids) for name, ids in split.items()}


def audit_split(x: Tensor, y: Tensor, split: dict) -> dict:
    """Fail on overlap, omissions, duplicate features crossing splits, or absent classes."""
    _features(x)
    _labels(y, len(x))
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("split must have train, validation and test")
    joined = [index for indices in split.values() for index in indices]
    if any(type(i) is not int for i in joined) or sorted(joined) != list(range(len(x))):
        raise ValueError("split must partition each source row exactly once")
    seen = {}
    duplicates = 0
    counts = {}
    for name, ids in split.items():
        counts[name] = torch.bincount(y[ids], minlength=3).tolist()
        if any(count == 0 for count in counts[name]):
            raise ValueError("each split must contain every class")
        for index in ids:
            fingerprint = tuple(x[index].tolist())
            if fingerprint in seen:
                duplicates += 1
                if seen[fingerprint] != name:
                    raise ValueError("identical feature rows cross split boundaries")
            seen[fingerprint] = name
    return {
        "sizes": {name: len(ids) for name, ids in split.items()},
        "class_counts": counts,
        "duplicate_feature_rows": duplicates,
        "source_rows_partitioned_once": True,
        "cross_split_feature_duplicates": 0,
    }


@dataclass(frozen=True)
class Standardizer:
    mean: Tensor
    scale: Tensor

    @classmethod
    def fit(cls, train_x: Tensor) -> "Standardizer":
        """Population variance (correction=0); constant features use scale 1."""
        _features(train_x)
        mean = train_x.mean(dim=0)
        scale = train_x.std(dim=0, correction=0)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("non-finite fitted statistics")
        return cls(mean.detach().clone(), scale.detach().clone())

    def transform(self, x: Tensor) -> Tensor:
        _features(x)
        if x.shape[1:] != self.mean.shape or self.scale.shape != self.mean.shape:
            raise ValueError("standardizer feature width mismatch")
        result = (x - self.mean) / self.scale
        _features(result)
        return result


class WineClassifier(nn.Module):
    """Locally initialized affine softmax model or one ReLU hidden layer."""

    def __init__(self, architecture: str, seed: int):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError("unknown architecture")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        generator = torch.Generator().manual_seed(seed)
        widths = (13, 3) if architecture == "linear" else (13, 16, 3)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for width_in, width_out in zip(widths[:-1], widths[1:], strict=True):
            bound = 1 / math.sqrt(width_in)
            weight = torch.empty(width_out, width_in, dtype=torch.float64)
            weight.uniform_(-bound, bound, generator=generator)
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(torch.zeros(width_out, dtype=torch.float64)))

    def forward(self, x: Tensor) -> Tensor:
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = F.linear(x, weight, bias)
            if index < len(self.weights) - 1:
                x = F.relu(x)
        return x


@dataclass(frozen=True)
class WineConfig:
    epochs: int = 120
    batch_size: int = 16
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0001

    def validate(self) -> None:
        if type(self.epochs) is not int or not 1 <= self.epochs <= 1000:
            raise ValueError("epochs must be an integer in [1, 1000]")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 107:
            raise ValueError("batch size must be an integer in [1, 107]")
        for name in ("learning_rate", "momentum", "weight_decay"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or not 0 <= self.momentum < 1 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")


@torch.no_grad()
def metrics(logits: Tensor, y: Tensor, row_ids: list[int]) -> dict:
    """Rows=true and columns=predicted; zero-denominator precision/F1 is zero."""
    _features(logits)
    _labels(y, len(logits))
    if logits.shape[1] != 3 or len(row_ids) != len(y):
        raise ValueError("metrics require [N,3] logits and one row ID per sample")
    loss = F.cross_entropy(logits, y).item()
    if not math.isfinite(loss):
        raise ValueError("non-finite cross entropy")
    prediction = logits.argmax(dim=1)
    probabilities = logits.softmax(dim=1)
    confusion = torch.bincount(3 * y + prediction, minlength=9).reshape(3, 3)
    classes = []
    for label in range(3):
        tp = int(confusion[label, label])
        support = int(confusion[label].sum())
        predicted = int(confusion[:, label].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        classes.append(
            {"class": label, "support": support, "precision": precision, "recall": recall, "f1": f1}
        )
    errors = [
        {
            "row_id": row_ids[i],
            "true": int(y[i]),
            "predicted": int(prediction[i]),
            "predicted_probability": float(probabilities[i, prediction[i]]),
            "true_probability": float(probabilities[i, y[i]]),
        }
        for i in range(len(y))
        if y[i] != prediction[i]
    ]
    errors.sort(key=lambda item: (-item["predicted_probability"], item["row_id"]))
    return {
        "cross_entropy": loss,
        "accuracy": float((prediction == y).double().mean()),
        "macro_f1": sum(item["f1"] for item in classes) / 3,
        "confusion_matrix": confusion.tolist(),
        "per_class": classes,
        "errors": errors,
        "row_ids": row_ids,
        "predictions": prediction.tolist(),
        "probabilities": probabilities.tolist(),
    }


def fit_candidate(
    train_x: Tensor, train_y: Tensor, architecture: str, seed: int, config: WineConfig
) -> tuple[WineClassifier, dict]:
    """Only the training partition is accepted; no test-driven stopping or tuning."""
    config.validate()
    _features(train_x)
    _labels(train_y, len(train_x))
    if train_x.shape[1] != 13:
        raise ValueError("Wine classifiers require 13 features")
    model = WineClassifier(architecture, seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    history, updates, samples = [], 0, 0
    model.train()
    for epoch in range(1, config.epochs + 1):
        online_sum = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_batch), y_batch)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("non-finite gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("non-finite parameter")
            updates += 1
            samples += len(y_batch)
            online_sum += loss.item() * len(y_batch)
        if epoch == 1 or epoch % 20 == 0 or epoch == config.epochs:
            with torch.no_grad():
                final_loss = F.cross_entropy(model(train_x), train_y).item()
            history.append(
                {
                    "epoch": epoch,
                    "online_cross_entropy": online_sum / len(train_y),
                    "final_train_cross_entropy": final_loss,
                }
            )
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return model, {
        "architecture": architecture,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "updates": updates,
        "samples_seen": samples,
        "history": history,
    }


def select_architecture(validation: dict[str, list[float]]) -> str:
    """One choice from mean final validation CE across the three prespecified seeds."""
    if set(validation) != set(ARCHITECTURES):
        raise ValueError("validation must contain both prespecified architectures")
    for values in validation.values():
        if len(values) != len(SEEDS) or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("three finite nonnegative validation losses required")
    parameters = {"linear": 42, "mlp16": 275}
    return min(
        validation, key=lambda name: (sum(validation[name]) / len(SEEDS), parameters[name], name)
    )


def _summary(runs: list[dict], partition: str) -> dict:
    result = {}
    for key in ("accuracy", "macro_f1", "cross_entropy"):
        values = [run[partition][key] for run in runs]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        result[key] = {"mean": mean, "sample_std": std, "values": values}
    return result


def run_wine_experiment(config: WineConfig | None = None) -> dict:
    """Train and select before final test evaluation; restore caller CPU thread count."""
    if config is None:
        config = WineConfig()
    config.validate()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _run_wine_experiment(config)
    finally:
        torch.set_num_threads(old_threads)


@torch.no_grad()
def evaluate_classifier(model: nn.Module, x: Tensor, y: Tensor, ids: list[int]) -> dict:
    was_training = model.training
    model.eval()
    try:
        return metrics(model(x), y, ids)
    finally:
        model.train(was_training)


def _run_wine_experiment(config: WineConfig) -> dict:
    x, y = load_wine()
    split = build_split(y)
    expected = json.loads(files("pytorch_lab").joinpath("data/wine_split.json").read_text())
    if split != expected["row_ids"] or expected["data_sha256"] != DATA_SHA256:
        raise ValueError("pinned split manifest mismatch")
    audit = audit_split(x, y, split)
    scaler = Standardizer.fit(x[split["train"]])
    train_x, train_y = scaler.transform(x[split["train"]]), y[split["train"]]
    val_x, val_y = scaler.transform(x[split["validation"]]), y[split["validation"]]
    candidates, losses = {}, {}
    for architecture in ARCHITECTURES:
        candidates[architecture], losses[architecture] = [], []
        for seed in SEEDS:
            model, run = fit_candidate(train_x, train_y, architecture, seed, config)
            run["validation"] = evaluate_classifier(model, val_x, val_y, split["validation"])
            losses[architecture].append(run["validation"]["cross_entropy"])
            candidates[architecture].append((model, run))
    selected = select_architecture(losses)
    # Test features, labels and predictions are not used until selection is final.
    test_x, test_y = scaler.transform(x[split["test"]]), y[split["test"]]
    results = {}
    for architecture in ARCHITECTURES:
        runs = []
        for model, run in candidates[architecture]:
            run["train"] = evaluate_classifier(model, train_x, train_y, split["train"])
            run["test"] = evaluate_classifier(model, test_x, test_y, split["test"])
            runs.append(run)
        results[architecture] = {
            "runs": runs,
            "validation": _summary(runs, "validation"),
            "test": _summary(runs, "test"),
        }
    priors = torch.bincount(train_y, minlength=3).double() / len(train_y)
    baseline = metrics(priors.log().expand(len(test_y), -1), test_y, split["test"])
    checks = {
        "all_rows_disjoint_and_covered": audit["source_rows_partitioned_once"],
        "no_cross_split_exact_feature_duplicates": audit["cross_split_feature_duplicates"] == 0,
        "train_scaled_mean_zero": bool(train_x.mean(dim=0).abs().max() < 1e-12),
        "train_scaled_variance_one": bool(
            (train_x.var(dim=0, correction=0) - 1).abs().max() < 1e-12
        ),
        "equal_update_and_sample_budgets": all(
            run["updates"] == math.ceil(len(train_y) / config.batch_size) * config.epochs
            and run["samples_seen"] == len(train_y) * config.epochs
            for item in results.values()
            for run in item["runs"]
        ),
    }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float64",
            "threads": 1,
        },
        "dataset": {
            "name": "UCI Wine",
            "sha256": DATA_SHA256,
            "source": "https://archive.ics.uci.edu/dataset/109/wine",
            "license": "CC-BY-4.0",
            "features": list(FEATURE_NAMES),
        },
        "protocol": {
            "config": asdict(config),
            "seeds": list(SEEDS),
            "split_salt": SPLIT_SALT,
            "split_row_ids": split,
            "selection": (
                "lowest mean final validation cross entropy across seeds; "
                "ties: fewer parameters, name"
            ),
            "selected_architecture": selected,
            "test_usage": (
                "final evaluation of all prespecified candidates after selection; no retuning"
            ),
            "seed_variation": "initialization and minibatch order, fixed data split",
        },
        "audit": audit,
        "preprocessing": {
            "fit_partition": "train",
            "correction": 0,
            "mean": scaler.mean.tolist(),
            "scale": scaler.scale.tolist(),
        },
        "training_prior_baseline": {"priors": priors.tolist(), "test": baseline},
        "models": results,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "limits": [
            "Small easy historical dataset; no production or population-generalization claim.",
            "Three seeds on one split do not estimate data-sampling uncertainty.",
            "No group/time metadata: exact duplicate and row checks cannot exclude latent leakage.",
            "Equal sample/update budgets do not mean equal FLOPs or wall time.",
            "Uncalibrated softmax scores are not reliability guarantees.",
            "Assistant-generated evidence does not establish independent learner mastery.",
        ],
    }
