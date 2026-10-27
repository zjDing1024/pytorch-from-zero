"""Original trusted-file recovery at completed microbatch safe points.

Fixed ordered CPU float64 affine fixture only. Pending gradients are unnormalized
loss SUM gradients. No RNG is consumed during training; fixture generation uses
an isolated local generator and is revalidated by its content fingerprint.
"""

import copy
import io
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from pytorch_lab.accumulation_cli import _fixture
from pytorch_lab.checkpoint import (
    MAX_CHECKPOINT_BYTES,
    CheckpointError,
    _atomic_write,
    _content_digest,
    _keys,
    _tensor,
)
from pytorch_lab.minibatch import AffineRegressor

FORMAT = "pytorch-lab-accumulation-microbatch-checkpoint"
MAX_EPOCHS = 100


@dataclass(frozen=True)
class WindowConfig:
    seed: int = 42
    samples: int = 23
    microbatch_size: int = 5
    accumulation_steps: int = 3
    learning_rate: float = 0.03
    momentum: float = 0.8
    weight_decay: float = 0.01

    def validate(self) -> None:
        for name, upper in (
            ("seed", 2**32 - 1),
            ("samples", 1000),
            ("microbatch_size", 1000),
            ("accumulation_steps", 1000),
        ):
            value = getattr(self, name)
            low = 0 if name == "seed" else 1
            if type(value) is not int or not low <= value <= upper:
                raise CheckpointError(f"{name} must be an integer in [{low}, {upper}]")
        for name in ("learning_rate", "momentum", "weight_decay"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise CheckpointError(f"{name} must be a finite float")
        if not 0 < self.learning_rate <= 1 or not 0 < self.momentum < 1:
            raise CheckpointError("learning rate or momentum is outside supported bounds")
        if not 0 <= self.weight_decay <= 1:
            raise CheckpointError("weight_decay must be in [0, 1]")


class WindowTrainer:
    """Owned single-threaded state; do not mutate private members.

    step_microbatch performs backward, optional update, and cursor advance as one
    safe-point operation. An exception invalidates the object: no rollback is
    promised, and save/continue then fail. Reload a prior trusted checkpoint.
    """

    def __init__(self, config: WindowConfig | None = None):
        self._config = WindowConfig() if config is None else config
        if type(self._config) is not WindowConfig:
            raise CheckpointError("config must be WindowConfig")
        self._config.validate()
        self._data = _fixture(self.config.samples, self.config.seed)
        self._data_hash = _content_digest([self._data.features, self._data.targets])
        self._model = AffineRegressor(3, 2)
        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        self._epoch = self._cursor = self._updates = 0
        self._pending_batches = self._pending_samples = self._pending_elements = 0
        self._safe = True

    @property
    def config(self):
        return self._config

    @property
    def epoch(self):
        return self._epoch

    @property
    def cursor(self):
        return self._cursor

    def step_microbatch(self) -> None:
        if not self._safe:
            raise CheckpointError("failed/in-progress operation: reload a safe checkpoint")
        if self.epoch >= MAX_EPOCHS:
            raise CheckpointError("maximum epoch limit reached")
        self._safe = False
        try:
            end = min(self.cursor + self.config.microbatch_size, self.config.samples)
            x, y = self._data.features[self.cursor : end], self._data.targets[self.cursor : end]
            loss = (self._model(x) - y).square().sum()
            if not torch.isfinite(loss).item():
                raise CheckpointError("non-finite loss")
            loss.backward()
            if any(
                p.grad is None or not torch.isfinite(p.grad).all().item()
                for p in self._model.parameters()
            ):
                raise CheckpointError("non-finite or missing pending gradient")
            self._pending_batches += 1
            self._pending_samples += end - self.cursor
            self._pending_elements += y.numel()
            if (
                self._pending_batches == self.config.accumulation_steps
                or end == self.config.samples
            ):
                for p in self._model.parameters():
                    p.grad.div_(self._pending_elements)
                self._optimizer.step()
                if any(not torch.isfinite(p).all().item() for p in self._model.parameters()):
                    raise CheckpointError("non-finite updated parameter")
                self._optimizer.zero_grad(set_to_none=True)
                self._updates += 1
                self._pending_batches = self._pending_samples = self._pending_elements = 0
            self._cursor = end
            if end == self.config.samples:
                self._epoch += 1
                self._cursor = 0
            self._safe = True
        except Exception:
            self._optimizer.zero_grad(set_to_none=True)
            raise

    def train_to(self, total_epochs: int) -> None:
        if type(total_epochs) is not int or not self.epoch <= total_epochs <= MAX_EPOCHS:
            raise CheckpointError("total_epochs must be a bounded nondecreasing integer")
        if not self._safe:
            raise CheckpointError("failed/in-progress operation: reload a safe checkpoint")
        while self.epoch < total_epochs:
            self.step_microbatch()

    def snapshot(self) -> dict:
        if not self._safe:
            raise CheckpointError("cannot snapshot a failed/in-progress operation")
        body = {
            "format": FORMAT,
            "schema_version": 1,
            "torch_version": str(torch.__version__),
            "config": asdict(self.config),
            "data_sha256": self._data_hash,
            "order": "fixed-sequential",
            "training_rng": "none",
            "epoch": self.epoch,
            "cursor": self.cursor,
            "updates": self._updates,
            "pending_batches": self._pending_batches,
            "pending_samples": self._pending_samples,
            "pending_elements": self._pending_elements,
            "model": dict(self._model.state_dict()),
            "optimizer": self._optimizer.state_dict(),
            "pending_gradients": {
                name: None if p.grad is None else p.grad.detach().clone()
                for name, p in self._model.named_parameters()
            },
        }
        body = copy.deepcopy(body)
        body["content_sha256"] = _content_digest(body)
        return body


def restore_snapshot(payload: dict, expected_config: WindowConfig | None = None) -> WindowTrainer:
    """Validate a trusted snapshot before returning a newly owned trainer.

    Structural checks and checksum detect accidental damage, not a malicious
    author able to rewrite valid tensor values and recompute the checksum.
    """
    fields = {
        "format",
        "schema_version",
        "torch_version",
        "config",
        "data_sha256",
        "order",
        "training_rng",
        "epoch",
        "cursor",
        "updates",
        "pending_batches",
        "pending_samples",
        "pending_elements",
        "model",
        "optimizer",
        "pending_gradients",
        "content_sha256",
    }
    _keys(payload, fields, "snapshot")
    if (
        payload["format"] != FORMAT
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
    ):
        raise CheckpointError("unsupported snapshot format/schema")
    if payload["torch_version"] != str(torch.__version__):
        raise CheckpointError("exact PyTorch version mismatch")
    if payload["order"] != "fixed-sequential" or payload["training_rng"] != "none":
        raise CheckpointError("unsupported order/RNG contract")
    _keys(payload["config"], set(asdict(WindowConfig())), "config")
    config = WindowConfig(**payload["config"])
    config.validate()
    if expected_config is not None:
        if type(expected_config) is not WindowConfig:
            raise CheckpointError("expected_config must be WindowConfig")
        expected_config.validate()
        if config != expected_config:
            raise CheckpointError("expected config mismatch")
    for name in (
        "epoch",
        "cursor",
        "updates",
        "pending_batches",
        "pending_samples",
        "pending_elements",
    ):
        if type(payload[name]) is not int or payload[name] < 0:
            raise CheckpointError(f"{name} must be a nonnegative integer")
    epoch, cursor = payload["epoch"], payload["cursor"]
    if epoch > MAX_EPOCHS or cursor >= config.samples or cursor % config.microbatch_size:
        raise CheckpointError("epoch/cursor outside supported microbatch safe points")
    if epoch == MAX_EPOCHS and cursor:
        raise CheckpointError("cursor exceeds maximum training horizon")
    per_epoch = math.ceil(config.samples / config.microbatch_size)
    k = cursor // config.microbatch_size
    pending = k % config.accumulation_steps
    expected_updates = (
        epoch * math.ceil(per_epoch / config.accumulation_steps) + k // config.accumulation_steps
    )
    if (
        payload["updates"] != expected_updates
        or payload["pending_batches"] != pending
        or payload["pending_samples"] != pending * config.microbatch_size
        or payload["pending_elements"] != pending * config.microbatch_size * 2
    ):
        raise CheckpointError("cursor, window counters and updates disagree")
    shapes = {"weight": (3, 2), "bias": (2,)}
    _keys(payload["model"], set(shapes), "model")
    _keys(payload["pending_gradients"], set(shapes), "pending_gradients")
    for name, shape in shapes.items():
        _tensor(payload["model"][name], shape, torch.float64, f"model.{name}")
        grad = payload["pending_gradients"][name]
        if pending:
            _tensor(grad, shape, torch.float64, f"pending_gradients.{name}")
        elif grad is not None:
            raise CheckpointError("zero pending window must have no gradients")
    # Check content integrity; this is not authentication.
    digest = payload["content_sha256"]
    if type(digest) is not str or len(digest) != 64:
        raise CheckpointError("invalid content checksum")
    if _content_digest({k: v for k, v in payload.items() if k != "content_sha256"}) != digest:
        raise CheckpointError("content checksum mismatch")
    data = _fixture(config.samples, config.seed)
    if payload["data_sha256"] != _content_digest([data.features, data.targets]):
        raise CheckpointError("fixture fingerprint mismatch")
    optimizer = payload["optimizer"]
    _keys(optimizer, {"state", "param_groups"}, "optimizer")
    expected_groups = [
        {
            "lr": config.learning_rate,
            "momentum": config.momentum,
            "dampening": 0,
            "weight_decay": config.weight_decay,
            "nesterov": False,
            "maximize": False,
            "foreach": None,
            "differentiable": False,
            "fused": None,
            "params": [0, 1],
        }
    ]
    if type(optimizer["param_groups"]) is not list or optimizer["param_groups"] != expected_groups:
        raise CheckpointError("optimizer parameter groups mismatch")
    # Typed digest comparison rejects bool/int substitutions hidden by Python equality.
    if _content_digest(optimizer["param_groups"]) != _content_digest(expected_groups):
        raise CheckpointError("optimizer parameter-group types mismatch")
    expected_ids = {0, 1} if expected_updates else set()
    _keys(optimizer["state"], expected_ids, "optimizer state")
    if any(type(i) is not int for i in optimizer["state"]):
        raise CheckpointError("optimizer IDs must be integers")
    for index, shape in enumerate(shapes.values()):
        if expected_updates:
            _keys(optimizer["state"][index], {"momentum_buffer"}, "momentum state")
            _tensor(optimizer["state"][index]["momentum_buffer"], shape, torch.float64, "momentum")
    candidate = WindowTrainer(config)
    candidate._model.load_state_dict(payload["model"], strict=True)
    candidate._optimizer.load_state_dict(copy.deepcopy(optimizer))
    for name, p in candidate._model.named_parameters():
        value = payload["pending_gradients"][name]
        p.grad = None if value is None else value.clone()
    candidate._epoch, candidate._cursor, candidate._updates = epoch, cursor, expected_updates
    candidate._pending_batches = pending
    candidate._pending_samples = payload["pending_samples"]
    candidate._pending_elements = payload["pending_elements"]
    return candidate


def save_window_checkpoint(trainer: WindowTrainer, path: str | Path) -> None:
    if type(trainer) is not WindowTrainer:
        raise CheckpointError("trainer must be WindowTrainer")
    payload = trainer.snapshot()
    restore_snapshot(payload, trainer.config)

    def write(output):
        torch.save(payload, output)
        if output.tell() > MAX_CHECKPOINT_BYTES:
            raise CheckpointError("snapshot exceeds size limit")

    _atomic_write(Path(path), write)


def load_window_checkpoint(
    path: str | Path, expected_config: WindowConfig | None = None
) -> WindowTrainer:
    """Only own/trusted files; weights_only is not a hostile-file sandbox."""
    try:
        with Path(path).open("rb") as source:
            if os.fstat(source.fileno()).st_size > MAX_CHECKPOINT_BYTES:
                raise CheckpointError("snapshot exceeds size limit")
            data = source.read(MAX_CHECKPOINT_BYTES + 1)
        if len(data) > MAX_CHECKPOINT_BYTES:
            raise CheckpointError("snapshot exceeds size limit")
        payload = torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")
        return restore_snapshot(payload, expected_config)
    except CheckpointError:
        raise
    except Exception as error:
        raise CheckpointError("snapshot could not be loaded or validated") from error
