"""Original joint safe points for shuffled CPU float64 affine SGD.

Issuance, backward/update and progress commit are distinct. A failed operation is
not rolled back: discard that owner and reload a whole previously saved joint
checkpoint. Recomputed work may happen twice; external effects are not covered.
"""

import copy
import io
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from pytorch_lab.accumulation_cli import _fixture
from pytorch_lab.accumulation_recovery import MAX_EPOCHS, WindowConfig
from pytorch_lab.checkpoint import (
    MAX_CHECKPOINT_BYTES,
    CheckpointError,
    _atomic_write,
    _content_digest,
    _keys,
    _tensor,
)
from pytorch_lab.minibatch import AffineRegressor
from pytorch_lab.shuffle_stream import (
    ShuffleConfig,
    ShuffledBatchStream,
    ShuffleStateError,
)
from pytorch_lab.shuffle_stream import (
    restore_snapshot as restore_stream,
)

FORMAT = "pytorch-lab-joint-shuffled-training"
FAULTS = ("after_fetch", "after_backward", "after_update", "before_commit")


class InjectedFailure(CheckpointError):
    """An explicitly requested experiment failure, not a recoverable owner state."""


@dataclass(frozen=True)
class JointConfig(WindowConfig):
    shuffle_seed: int = 19

    def validate(self) -> None:
        super().validate()
        if type(self.shuffle_seed) is not int or not 0 <= self.shuffle_seed < 2**32:
            raise CheckpointError("shuffle_seed must be an integer in [0, 2**32)")


class JointTrainer:
    """Single-threaded exclusive owner; no direct mutation of private members.

    A completed step is an in-memory safe point, not a durable acknowledgement.
    Save one whole snapshot at this point. Failures before the commit leave this
    owner unusable, even if SGD or epoch
    advance mutated it. Arbitrary asynchronous interruption at the final
    commit/return boundary is not a supported acknowledgement protocol.
    """

    def __init__(self, config: JointConfig | None = None):
        self._config = JointConfig() if config is None else config
        if type(self._config) is not JointConfig:
            raise CheckpointError("config must be JointConfig")
        self.config.validate()
        self._data = _fixture(self.config.samples, self.config.seed)
        self._data_hash = _content_digest([self._data.features, self._data.targets])
        self._stream = ShuffledBatchStream(self._stream_config())
        self._model = AffineRegressor(3, 2)
        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        self._epoch = self._cursor = self._updates = self._committed_batches = 0
        self._pending_batches = self._pending_samples = self._pending_elements = 0
        self._safe = True

    def _stream_config(self):
        return ShuffleConfig(
            dataset_sha256=self._data_hash,
            samples=self.config.samples,
            batch_size=self.config.microbatch_size,
            seed=self.config.shuffle_seed,
            drop_last=False,
        )

    @property
    def config(self):
        return self._config

    @property
    def epoch(self):
        return self._epoch

    @property
    def cursor(self):
        return self._cursor

    @property
    def committed_batches(self):
        return self._committed_batches

    def _require_safe(self):
        if not self._safe:
            raise CheckpointError("failed/in-progress owner: reload a whole joint checkpoint")

    def step_microbatch(self, *, fault_at: str | None = None) -> tuple[int, ...]:
        """Return committed indices; experimental failpoints never return a batch.

        No callback/user side effect runs inside the transaction. Failpoint
        after_update is accepted only when this batch will trigger an update.
        """
        self._require_safe()
        if fault_at is not None and (type(fault_at) is not str or fault_at not in FAULTS):
            raise CheckpointError("unknown fault point")
        if self.epoch >= MAX_EPOCHS:
            raise CheckpointError("maximum epoch limit reached")
        end = min(self.cursor + self.config.microbatch_size, self.config.samples)
        update = (
            self._pending_batches + 1 == self.config.accumulation_steps
            or end == self.config.samples
        )
        if fault_at == "after_update" and not update:
            raise CheckpointError("after_update requires an update-triggering batch")

        def fail(point):
            if fault_at == point:
                raise InjectedFailure(point)

        # Deliberately no finally resetting the guard and no partial rollback.
        self._safe = False
        indices = self._stream.next_batch()
        fail("after_fetch")
        selected = torch.tensor(indices, dtype=torch.int64, device="cpu")
        x = self._data.features.index_select(0, selected)
        y = self._data.targets.index_select(0, selected)
        loss = (self._model(x) - y).square().sum()
        if not torch.isfinite(loss).item():
            raise CheckpointError("non-finite loss")
        loss.backward()
        if any(
            p.grad is None or not torch.isfinite(p.grad).all().item()
            for p in self._model.parameters()
        ):
            raise CheckpointError("non-finite or missing pending gradient")
        fail("after_backward")
        pending_batches = self._pending_batches + 1
        pending_samples = self._pending_samples + len(indices)
        pending_elements = self._pending_elements + y.numel()
        if update:
            for parameter in self._model.parameters():
                parameter.grad.div_(pending_elements)
            self._optimizer.step()
            if any(not torch.isfinite(p).all().item() for p in self._model.parameters()):
                raise CheckpointError("non-finite updated parameter")
            fail("after_update")
            self._optimizer.zero_grad(set_to_none=True)
            pending_batches = pending_samples = pending_elements = 0
        if self._stream.exhausted:
            # Canonical safe point: tail flushed, next permutation prepared.
            self._stream.advance_epoch()
        fail("before_commit")
        self._epoch, self._cursor = self._stream.epoch, self._stream.cursor
        self._updates += int(update)
        self._committed_batches += 1
        self._pending_batches = pending_batches
        self._pending_samples = pending_samples
        self._pending_elements = pending_elements
        self._safe = True
        return indices

    def train_to(self, total_epochs: int) -> None:
        self._require_safe()
        if type(total_epochs) is not int or not self.epoch <= total_epochs <= MAX_EPOCHS:
            raise CheckpointError("total_epochs must be a bounded nondecreasing integer")
        while self.epoch < total_epochs:
            self.step_microbatch()

    def snapshot(self) -> dict:
        self._require_safe()
        body = copy.deepcopy(
            {
                "format": FORMAT,
                "schema_version": 1,
                "torch_version": str(torch.__version__),
                "config": asdict(self.config),
                "data_sha256": self._data_hash,
                "stream": self._stream.snapshot(),
                "epoch": self.epoch,
                "cursor": self.cursor,
                "updates": self._updates,
                "committed_batches": self.committed_batches,
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
        )
        body["content_sha256"] = _content_digest(body)
        return body


def restore_snapshot(payload: dict, expected_config: JointConfig | None = None) -> JointTrainer:
    """Validate and restore one joint object, never a caller-assembled split pair.

    The nested/outer hashes detect accidental corruption, not forged history or
    a malicious author who can recompute hashes. Only own trusted input files.
    """
    _keys(
        payload,
        {
            "format",
            "schema_version",
            "torch_version",
            "config",
            "data_sha256",
            "stream",
            "epoch",
            "cursor",
            "updates",
            "committed_batches",
            "pending_batches",
            "pending_samples",
            "pending_elements",
            "model",
            "optimizer",
            "pending_gradients",
            "content_sha256",
        },
        "joint snapshot",
    )
    if (
        payload["format"] != FORMAT
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["torch_version"] != str(torch.__version__)
    ):
        raise CheckpointError("unsupported joint format/schema/PyTorch version")
    _keys(payload["config"], set(asdict(JointConfig())), "joint config")
    config = JointConfig(**payload["config"])
    config.validate()
    if expected_config is not None:
        if type(expected_config) is not JointConfig:
            raise CheckpointError("expected_config must be JointConfig")
        expected_config.validate()
        if config != expected_config:
            raise CheckpointError("expected joint config mismatch")
    candidate = JointTrainer(config)
    if payload["data_sha256"] != candidate._data_hash:
        raise CheckpointError("fixture fingerprint mismatch")
    try:
        stream = restore_stream(payload["stream"], candidate._stream_config())
    except ShuffleStateError as error:
        raise CheckpointError("invalid or mismatched joint stream") from error
    for key in (
        "epoch",
        "cursor",
        "updates",
        "committed_batches",
        "pending_batches",
        "pending_samples",
        "pending_elements",
    ):
        if type(payload[key]) is not int or payload[key] < 0:
            raise CheckpointError(f"{key} must be a nonnegative integer")
    epoch, cursor = payload["epoch"], payload["cursor"]
    if (
        epoch > MAX_EPOCHS
        or cursor >= config.samples
        or cursor % config.microbatch_size
        or (epoch == MAX_EPOCHS and cursor)
        or (epoch, cursor) != (stream.epoch, stream.cursor)
    ):
        raise CheckpointError("trainer and stream are not at the same canonical safe point")
    per_epoch = math.ceil(config.samples / config.microbatch_size)
    within = cursor // config.microbatch_size
    pending = within % config.accumulation_steps
    updates = epoch * math.ceil(per_epoch / config.accumulation_steps)
    updates += within // config.accumulation_steps
    expected = {
        "committed_batches": epoch * per_epoch + within,
        "updates": updates,
        "pending_batches": pending,
        "pending_samples": pending * config.microbatch_size,
        "pending_elements": pending * config.microbatch_size * 2,
    }
    if any(payload[key] != value for key, value in expected.items()):
        raise CheckpointError("joint progress/window counters disagree")
    shapes = {"weight": (3, 2), "bias": (2,)}
    _keys(payload["model"], set(shapes), "model")
    _keys(payload["pending_gradients"], set(shapes), "pending gradients")
    for name, shape in shapes.items():
        _tensor(payload["model"][name], shape, torch.float64, f"model.{name}")
        grad = payload["pending_gradients"][name]
        if pending:
            _tensor(grad, shape, torch.float64, f"pending_gradients.{name}")
        elif grad is not None:
            raise CheckpointError("empty window must have no gradients")
    optimizer = payload["optimizer"]
    _keys(optimizer, {"state", "param_groups"}, "optimizer")
    groups = candidate._optimizer.state_dict()["param_groups"]
    if _content_digest(optimizer["param_groups"]) != _content_digest(groups):
        raise CheckpointError("optimizer parameter groups/types mismatch")
    _keys(optimizer["state"], {0, 1} if updates else set(), "momentum state")
    if any(type(key) is not int for key in optimizer["state"]):
        raise CheckpointError("optimizer IDs must be integers")
    for key, shape in enumerate(shapes.values()):
        if updates:
            _keys(optimizer["state"][key], {"momentum_buffer"}, "momentum")
            _tensor(optimizer["state"][key]["momentum_buffer"], shape, torch.float64, "momentum")
    digest = payload["content_sha256"]
    if type(digest) is not str or digest != _content_digest(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    ):
        raise CheckpointError("joint content checksum mismatch")
    candidate._stream = stream
    candidate._model.load_state_dict(payload["model"], strict=True)
    candidate._optimizer.load_state_dict(copy.deepcopy(optimizer))
    for name, parameter in candidate._model.named_parameters():
        value = payload["pending_gradients"][name]
        parameter.grad = None if value is None else value.clone()
    candidate._epoch, candidate._cursor = epoch, cursor
    for name, value in expected.items():
        setattr(candidate, "_" + name, value)
    return candidate


def save_joint_checkpoint(trainer: JointTrainer, path: str | Path) -> None:
    """No-clobber single-file publication; local hard links, no durability claim."""
    if type(trainer) is not JointTrainer:
        raise CheckpointError("trainer must be JointTrainer")
    payload = trainer.snapshot()
    restore_snapshot(payload, trainer.config)

    def write(output):
        torch.save(payload, output)
        if output.tell() > MAX_CHECKPOINT_BYTES:
            raise CheckpointError("joint snapshot exceeds size limit")

    _atomic_write(Path(path), write)


def load_joint_checkpoint(path: str | Path, expected_config: JointConfig | None = None):
    """Bounded weights_only loading of trusted own files, not a hostile-file sandbox."""
    try:
        with Path(path).open("rb") as source:
            if os.fstat(source.fileno()).st_size > MAX_CHECKPOINT_BYTES:
                raise CheckpointError("joint snapshot exceeds size limit")
            encoded = source.read(MAX_CHECKPOINT_BYTES + 1)
        if len(encoded) > MAX_CHECKPOINT_BYTES:
            raise CheckpointError("joint snapshot exceeds size limit")
        payload = torch.load(io.BytesIO(encoded), weights_only=True, map_location="cpu")
        return restore_snapshot(payload, expected_config)
    except CheckpointError:
        raise
    except Exception as error:
        raise CheckpointError("joint snapshot could not be loaded or validated") from error
