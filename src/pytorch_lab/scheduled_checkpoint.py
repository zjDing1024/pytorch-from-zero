"""An isolated, strict StepLR extension of the version-1 epoch recovery lab.

Single CPU float64 parameter group; constant LR inside an epoch; scheduler.step
once AFTER all optimizer updates. Only load checkpoints from trusted sources.
"""

import copy
import io
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from pytorch_lab.checkpoint import (
    FORMAT as BASE_FORMAT,
)
from pytorch_lab.checkpoint import (
    MAX_CHECKPOINT_BYTES,
    CheckpointError,
    EpochTrainer,
    RecoveryConfig,
    _atomic_write,
    _canonical,
    _content_digest,
    _epoch,
    _keys,
    _validate_payload,
)
from pytorch_lab.minibatch import _finite_number, _positive_integer, evaluate, train_one_epoch

FORMAT = "pytorch-lab-step-lr-checkpoint"


@dataclass(frozen=True)
class ScheduleConfig:
    seed: int = 42
    batch_size: int = 20
    learning_rate: float = 0.05
    momentum: float = 0.8
    noise_std: float = 0.05
    step_size: int = 5
    gamma: float = 0.5

    def recovery_config(self) -> RecoveryConfig:
        return RecoveryConfig(
            seed=self.seed,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            noise_std=self.noise_std,
        )

    def validate(self) -> None:
        self.recovery_config().validate()
        _positive_integer(self.step_size, "step_size")
        _finite_number(self.gamma, "gamma")
        if self.gamma > 1:
            raise ValueError("gamma must be in (0, 1]; gamma=1 is the fixed-LR control")


def learning_rates(config: ScheduleConfig, epochs: int) -> list[float]:
    """LR at each boundary (index 0 is initial), matching repeated multiplication."""
    config.validate()
    _epoch(epochs)
    current = config.learning_rate
    result = [current]
    for epoch in range(1, epochs + 1):
        if epoch % config.step_size == 0:
            current *= config.gamma
        result.append(current)
    return result


class ScheduledTrainer(EpochTrainer):
    """Owned StepLR run. Public config is immutable; private state is not an API."""

    def __init__(self, config: ScheduleConfig | None = None) -> None:
        config = ScheduleConfig() if config is None else config
        if type(config) is not ScheduleConfig:
            raise ValueError("config must be ScheduleConfig")
        config.validate()
        super().__init__(config.recovery_config())
        self._schedule_config = config
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, step_size=config.step_size, gamma=config.gamma
        )
        self._history[0]["next_lr"] = self._optimizer.param_groups[0]["lr"]

    @property
    def schedule_config(self) -> ScheduleConfig:
        return self._schedule_config

    def train_to(self, total_epochs: int) -> None:
        _epoch(total_epochs)
        if not self._boundary:
            raise CheckpointError("failed/in-progress epoch: reload a completed boundary")
        if total_epochs < self.epoch:
            raise ValueError("total_epochs cannot go backwards")
        while self._epoch < total_epochs:
            self._boundary = False
            lr_used = self._optimizer.param_groups[0]["lr"]
            online = train_one_epoch(self._model, self._train_loader, self._optimizer)
            metric = evaluate(self._model, self._metric_loader)
            # The next epoch's rate is selected only after this epoch's updates.
            self._scheduler.step()
            self._epoch += 1
            self._history.append(
                {
                    "epoch": self.epoch,
                    "train_mse": metric,
                    "online_train_mse": online["mse"],
                    "samples_seen": online["samples_seen"],
                    "batches": online["batches"],
                    "lr_used": lr_used,
                    "next_lr": self._optimizer.param_groups[0]["lr"],
                }
            )
            self._boundary = True

    def snapshot(self) -> dict:
        payload = super().snapshot()
        payload["format"] = FORMAT
        payload["config"] = asdict(self.schedule_config)
        payload["scheduler"] = copy.deepcopy(self._scheduler.state_dict())
        payload.pop("content_sha256")
        payload["content_sha256"] = _content_digest(payload)
        return payload

    def report(self) -> dict:
        result = super().report()
        result["config"] = asdict(self.schedule_config)
        result["scheduler"] = copy.deepcopy(self._scheduler.state_dict())
        result["next_lr"] = self._optimizer.param_groups[0]["lr"]
        result["optimizer_updates"] = sum(row.get("batches", 0) for row in self._history)
        result["samples_seen"] = sum(row.get("samples_seen", 0) for row in self._history)
        return result


def _validate_scheduled_payload(
    payload: dict, expected_config: ScheduleConfig | None = None
) -> ScheduledTrainer:
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
            "scheduler",
            "generators",
            "history",
            "content_sha256",
        },
        "scheduled checkpoint",
    )
    if type(payload["format"]) is not str or payload["format"] != FORMAT:
        raise CheckpointError("scheduled checkpoint format is unsupported")
    _keys(payload["config"], set(asdict(ScheduleConfig())), "config")
    config = ScheduleConfig(**payload["config"])
    config.validate()
    if expected_config is not None:
        if type(expected_config) is not ScheduleConfig:
            raise CheckpointError("expected_config must be ScheduleConfig")
        expected_config.validate()
        if _canonical(asdict(expected_config)) != _canonical(payload["config"]):
            raise CheckpointError("checkpoint config does not match expected_config")
    _epoch(payload["epoch"])
    epoch = payload["epoch"]
    rates = learning_rates(config, epoch)
    candidate = ScheduledTrainer(config)
    # Lock the supported scheduler's entire state, including version-specific fields.
    expected_scheduler = candidate._scheduler.state_dict()
    expected_scheduler.update(last_epoch=epoch, _step_count=epoch + 1, _last_lr=[rates[-1]])
    if _canonical(payload["scheduler"]) != _canonical(expected_scheduler):
        raise CheckpointError("scheduler state/counters/config/LR mismatch")
    _keys(payload["optimizer"], {"state", "param_groups"}, "optimizer")
    expected_groups = candidate._optimizer.state_dict()["param_groups"]
    expected_groups[0]["lr"] = rates[-1]
    if _canonical(payload["optimizer"]["param_groups"]) != _canonical(expected_groups):
        raise CheckpointError("optimizer param_groups/current LR/initial_lr mismatch")
    history = payload["history"]
    if type(history) is not list or len(history) != epoch + 1:
        raise CheckpointError("history length does not match epoch")
    base_history = []
    for index, row in enumerate(history):
        fields = {"epoch", "train_mse", "next_lr"}
        if index:
            fields |= {"online_train_mse", "samples_seen", "batches", "lr_used"}
        _keys(row, fields, "scheduled history row")
        for key, expected in (("next_lr", rates[index]), ("lr_used", rates[max(0, index - 1)])):
            if key in row and _canonical(row[key]) != _canonical(expected):
                raise CheckpointError(f"history {key} does not match epoch schedule")
        base_history.append(
            {key: value for key, value in row.items() if key not in {"lr_used", "next_lr"}}
        )
    digest = payload["content_sha256"]
    if type(digest) is not str or len(digest) != 64:
        raise CheckpointError("content_sha256 is invalid")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if _content_digest(body) != digest:
        raise CheckpointError("content_sha256 checksum mismatch")
    # Reuse the proven v1 data/model/momentum/RNG/history contracts without relaxing
    # them or changing the old format. Only a new private candidate can be mutated.
    base = copy.deepcopy(body)
    base.pop("scheduler")
    base["format"] = BASE_FORMAT
    base["config"] = asdict(config.recovery_config())
    base["history"] = base_history
    group = base["optimizer"]["param_groups"][0]
    group.pop("initial_lr")
    group["lr"] = config.learning_rate
    base["content_sha256"] = _content_digest(base)
    validated = _validate_payload(base, config.recovery_config())
    candidate._model.load_state_dict(validated._model.state_dict(), strict=True)
    # Construct scheduler FIRST; restore its state, then restore optimizer LR/state.
    # No training update or scheduler.step is allowed between these operations.
    candidate._scheduler.load_state_dict(copy.deepcopy(payload["scheduler"]))
    candidate._optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
    candidate._train_loader.generator.set_state(payload["generators"]["train"])
    candidate._metric_loader.generator.set_state(payload["generators"]["metric"])
    candidate._epoch = epoch
    candidate._history = copy.deepcopy(history)
    candidate._model.train()
    return candidate


def save_scheduled_checkpoint(trainer: ScheduledTrainer, path: str | Path) -> None:
    if type(trainer) is not ScheduledTrainer:
        raise CheckpointError("trainer must be ScheduledTrainer")
    payload = trainer.snapshot()
    _validate_scheduled_payload(payload, trainer.schedule_config)

    def write(output) -> None:
        torch.save(payload, output)
        if output.tell() > MAX_CHECKPOINT_BYTES:
            raise CheckpointError("checkpoint exceeds size limit")

    _atomic_write(Path(path), write)


def load_scheduled_checkpoint(
    path: str | Path, expected_config: ScheduleConfig | None = None
) -> ScheduledTrainer:
    """Restricted trusted-file load, returning a new owned trainer; never fall back."""
    try:
        with Path(path).open("rb") as source:
            if os.fstat(source.fileno()).st_size > MAX_CHECKPOINT_BYTES:
                raise CheckpointError("checkpoint exceeds size limit")
            data = source.read(MAX_CHECKPOINT_BYTES + 1)
        if len(data) > MAX_CHECKPOINT_BYTES:
            raise CheckpointError("checkpoint exceeds size limit")
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        return _validate_scheduled_payload(payload, expected_config)
    except CheckpointError:
        raise
    except Exception as error:
        raise CheckpointError(f"checkpoint could not be loaded or validated: {error}") from error
