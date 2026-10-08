"""Versioned recovery for this CPU affine lab at completed epoch boundaries only.

Read only your own/trusted checkpoints. Restricted weights-only loading, size
limits and a checksum are defense in depth, not a sandbox or authentication.
"""

import copy
import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from pytorch_lab.minibatch import (
    AffineRegressor,
    _finite_number,
    _positive_integer,
    _seed,
    evaluate,
    make_loader,
    make_splits,
    train_one_epoch,
)

MAX_CHECKPOINT_BYTES = 8 * 1024 * 1024
MAX_EPOCHS = 10_000
FORMAT = "pytorch-lab-epoch-checkpoint"


class CheckpointError(ValueError):
    """Unsupported, inconsistent or corrupt lab checkpoint."""


@dataclass(frozen=True)
class RecoveryConfig:
    seed: int = 42
    batch_size: int = 20
    learning_rate: float = 0.05
    momentum: float = 0.8
    noise_std: float = 0.05

    def validate(self) -> None:
        _seed(self.seed)
        _positive_integer(self.batch_size, "batch_size")
        _finite_number(self.learning_rate, "learning_rate")
        _finite_number(self.momentum, "momentum")
        if self.momentum >= 1:
            raise ValueError("momentum must be strictly between zero and one")
        _finite_number(self.noise_std, "noise_std", allow_zero=True)


def _epoch(value: int) -> None:
    if type(value) is not int or not 0 <= value <= MAX_EPOCHS:
        raise CheckpointError(f"epoch must be an integer in [0, {MAX_EPOCHS}]")


def _keys(value: dict, expected: set, name: str) -> None:
    if type(value) is not dict or set(value) != expected:
        raise CheckpointError(f"{name} has invalid keys or type")


def _tensor(value: torch.Tensor, shape: tuple, dtype: torch.dtype, name: str) -> None:
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.device.type != "cpu"
        or value.dtype != dtype
        or tuple(value.shape) != shape
        or not value.is_contiguous()
        or value.requires_grad
    ):
        raise CheckpointError(f"{name} has invalid tensor shape, dtype, device or layout")
    if not torch.isfinite(value).all().item():
        raise CheckpointError(f"{name} must contain finite values")


def _canonical(value):
    """Typed encoding avoids bool/int, dict-order and tensor-storage ambiguity."""
    if type(value) is torch.Tensor:
        return ["tensor", str(value.dtype), list(value.shape), value.view(torch.uint8).tolist()]
    if type(value) is dict:
        entries = [[_canonical(k), _canonical(v)] for k, v in value.items()]
        return ["dict", sorted(entries, key=lambda pair: json.dumps(pair[0]))]
    if type(value) is list:
        return ["list", [_canonical(item) for item in value]]
    if type(value) not in (str, int, float, bool, type(None)):
        raise CheckpointError("unsupported checkpoint value type")
    return [type(value).__name__, value]


def _content_digest(payload: dict) -> str:
    encoded = json.dumps(_canonical(payload), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dataset_digest(splits: dict) -> str:
    return _content_digest({name: [data.features, data.targets] for name, data in splits.items()})


class EpochTrainer:
    """Owned, single-threaded state. Do not mutate private model/optimizer/loaders.

    train_to accepts a TOTAL epoch count, not an additional count. An interrupted
    epoch invalidates this object; recover from the last successfully saved file.
    """

    def __init__(self, config: RecoveryConfig | None = None) -> None:
        self._config = RecoveryConfig() if config is None else config
        if type(self.config) is not RecoveryConfig:
            raise ValueError("config must be RecoveryConfig")
        self.config.validate()
        self._splits = make_splits(self.config.seed, self.config.noise_std)
        self._model = AffineRegressor(3, 1, torch.float64)
        self._optimizer = torch.optim.SGD(
            self._model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum
        )
        self._train_loader = make_loader(
            self._splits["train"], self.config.batch_size, self.config.seed, shuffle=True
        )
        self._metric_loader = make_loader(
            self._splits["train"], self.config.batch_size, self.config.seed
        )
        self._epoch = 0
        self._history = [{"epoch": 0, "train_mse": evaluate(self._model, self._metric_loader)}]
        self._boundary = True

    @property
    def config(self) -> RecoveryConfig:
        """Read-only: saved settings cannot diverge from owned loader/optimizer state."""
        return self._config

    @property
    def epoch(self) -> int:
        return self._epoch

    def train_to(self, total_epochs: int) -> None:
        _epoch(total_epochs)
        if not self._boundary:
            raise CheckpointError("failed/in-progress epoch: reload a completed boundary")
        if total_epochs < self.epoch:
            raise ValueError("total_epochs cannot go backwards")
        while self._epoch < total_epochs:
            self._boundary = False
            online = train_one_epoch(self._model, self._train_loader, self._optimizer)
            metric = evaluate(self._model, self._metric_loader)
            self._epoch += 1
            self._history.append(
                {
                    "epoch": self._epoch,
                    "train_mse": metric,
                    "online_train_mse": online["mse"],
                    "samples_seen": online["samples_seen"],
                    "batches": online["batches"],
                }
            )
            self._boundary = True

    def snapshot(self) -> dict:
        if not self._boundary:
            raise CheckpointError("checkpoint requires a completed epoch boundary")
        payload = {
            "format": FORMAT,
            "schema_version": 1,
            "torch_version": str(torch.__version__),
            "config": asdict(self.config),
            "dataset_sha256": _dataset_digest(self._splits),
            "epoch": self.epoch,
            "model": dict(self._model.state_dict()),
            "optimizer": self._optimizer.state_dict(),
            "generators": {
                "train": self._train_loader.generator.get_state(),
                "metric": self._metric_loader.generator.get_state(),
            },
            "history": self._history,
        }
        payload = copy.deepcopy(payload)
        payload["content_sha256"] = _content_digest(payload)
        return payload

    def report(self) -> dict:
        """Observational only: evaluation uses fresh generators, never training state."""
        snapshot = self.snapshot()
        holdout = {
            name: evaluate(
                self._model,
                make_loader(self._splits[name], self.config.batch_size, self.config.seed),
            )
            for name in ("validation", "test")
        }
        train_mean = self._splits["train"].targets.mean(dim=0)
        baseline = (self._splits["test"].targets - train_mean).square().mean().item()
        return {
            "config": asdict(self.config),
            "epoch": self.epoch,
            "content_sha256": snapshot["content_sha256"],
            "learned_weight": self._model.weight.detach().flatten().tolist(),
            "learned_bias": self._model.bias.item(),
            "train_mse": self._history[-1]["train_mse"],
            "validation_mse": holdout["validation"],
            "test_mse": holdout["test"],
            "test_mean_baseline_mse": baseline,
            "history": copy.deepcopy(self._history),
        }


def _validate_payload(payload: dict, expected_config: RecoveryConfig | None) -> EpochTrainer:
    _keys(
        payload,
        {
            "format",
            "schema_version",
            "torch_version",
            "config",
            "dataset_sha256",
            "epoch",
            "model",
            "optimizer",
            "generators",
            "history",
            "content_sha256",
        },
        "checkpoint",
    )
    if payload["format"] != FORMAT or type(payload["format"]) is not str:
        raise CheckpointError("checkpoint format is unsupported")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise CheckpointError("schema_version is unsupported")
    if type(payload["torch_version"]) is not str or payload["torch_version"] != str(
        torch.__version__
    ):
        raise CheckpointError("torch_version differs; cross-version recovery is unsupported")
    _keys(payload["config"], set(asdict(RecoveryConfig())), "config")
    config = RecoveryConfig(**payload["config"])
    config.validate()
    if expected_config is not None:
        if type(expected_config) is not RecoveryConfig:
            raise CheckpointError("expected_config must be RecoveryConfig")
        expected_config.validate()
        if _canonical(asdict(expected_config)) != _canonical(payload["config"]):
            raise CheckpointError("checkpoint config does not match expected_config")
    _epoch(payload["epoch"])
    # New candidate only; no existing trainer or caller/global RNG is ever touched.
    candidate = EpochTrainer(config)
    if payload["dataset_sha256"] != _dataset_digest(candidate._splits):
        raise CheckpointError("dataset_sha256 does not match regenerated data")
    _keys(payload["model"], {"weight", "bias"}, "model")
    for name, shape in (("weight", (3, 1)), ("bias", (1,))):
        _tensor(payload["model"][name], shape, torch.float64, f"model.{name}")
    _keys(payload["optimizer"], {"state", "param_groups"}, "optimizer")
    template = candidate._optimizer.state_dict()["param_groups"]
    if _canonical(payload["optimizer"]["param_groups"]) != _canonical(template):
        raise CheckpointError("optimizer param_groups/config/parameter order mismatch")
    expected_ids = {0, 1} if payload["epoch"] else set()
    state = payload["optimizer"]["state"]
    _keys(state, expected_ids, "optimizer state")
    if any(type(key) is not int for key in state):
        raise CheckpointError("optimizer parameter IDs must be integers")
    for index, shape in enumerate(((3, 1), (1,))):
        if index in state:
            _keys(state[index], {"momentum_buffer"}, "optimizer state entry")
            _tensor(state[index]["momentum_buffer"], shape, torch.float64, "momentum_buffer")
    _keys(payload["generators"], {"train", "metric"}, "generators")
    rng_shape = tuple(torch.Generator().get_state().shape)
    for name, value in payload["generators"].items():
        _tensor(value, rng_shape, torch.uint8, f"generators.{name}")
        try:
            probe = torch.Generator().set_state(value)
            torch.randperm(96, generator=probe)
        except RuntimeError as error:
            raise CheckpointError(f"generators.{name} has invalid RNG state") from error
    history = payload["history"]
    if type(history) is not list or len(history) != payload["epoch"] + 1:
        raise CheckpointError("history length does not match epoch")
    for epoch, row in enumerate(history):
        fields = {"epoch", "train_mse"}
        if epoch:
            fields |= {"online_train_mse", "samples_seen", "batches"}
        _keys(row, fields, "history row")
        if type(row["epoch"]) is not int or row["epoch"] != epoch:
            raise CheckpointError("history epoch sequence is invalid")
        for name in ("train_mse", "online_train_mse"):
            if name in row and (
                type(row[name]) is not float or not math.isfinite(row[name]) or row[name] < 0
            ):
                raise CheckpointError(f"history {name} must be a finite nonnegative float")
        if epoch and (
            type(row["samples_seen"]) is not int
            or row["samples_seen"] != 96
            or type(row["batches"]) is not int
            or row["batches"] != (96 + config.batch_size - 1) // config.batch_size
        ):
            raise CheckpointError("history batch/sample counts are inconsistent")
    digest = payload["content_sha256"]
    if type(digest) is not str or len(digest) != 64:
        raise CheckpointError("content_sha256 is invalid")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if _content_digest(body) != digest:
        raise CheckpointError("content_sha256 checksum mismatch")
    # Applying to this private candidate cannot partially mutate any live trainer.
    candidate._model.load_state_dict(payload["model"], strict=True)
    candidate._optimizer.load_state_dict(payload["optimizer"])
    candidate._train_loader.generator.set_state(payload["generators"]["train"])
    candidate._metric_loader.generator.set_state(payload["generators"]["metric"])
    candidate._epoch = payload["epoch"]
    candidate._history = copy.deepcopy(history)
    candidate._model.train()
    return candidate


def _atomic_write(path: Path, writer) -> None:
    """Publish a complete fsynced file by hard link; never replace an existing name.

    Requires a local filesystem supporting same-directory hard links. No fallback
    to a partial direct write. This does not promise power-loss directory durability.
    """
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            writer(output)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def save_checkpoint(trainer: EpochTrainer, path: str | Path) -> None:
    """Save a detached, validated snapshot; no clobber even for concurrent writers."""
    if type(trainer) is not EpochTrainer:
        raise CheckpointError("trainer must be EpochTrainer")
    payload = trainer.snapshot()
    _validate_payload(payload, trainer.config)

    def write(output) -> None:
        torch.save(payload, output)
        if output.tell() > MAX_CHECKPOINT_BYTES:
            raise CheckpointError("checkpoint exceeds size limit")

    _atomic_write(Path(path), write)


def load_checkpoint(
    path: str | Path, expected_config: RecoveryConfig | None = None
) -> EpochTrainer:
    """Restricted deserialization + strict validation; return a NEW owned trainer.

    This is not a hostile-file sandbox. Never retry with weights_only=False.
    """
    try:
        with Path(path).open("rb") as source:
            if os.fstat(source.fileno()).st_size > MAX_CHECKPOINT_BYTES:
                raise CheckpointError("checkpoint exceeds size limit")
            data = source.read(MAX_CHECKPOINT_BYTES + 1)
        if len(data) > MAX_CHECKPOINT_BYTES:
            raise CheckpointError("checkpoint exceeds size limit")
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        return _validate_payload(payload, expected_config)
    except CheckpointError:
        raise
    except Exception as error:
        raise CheckpointError(f"checkpoint could not be loaded or validated: {error}") from error
