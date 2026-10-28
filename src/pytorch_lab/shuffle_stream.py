"""Original, single-owner CPU index emission; deliberately not training recovery.

A successful next_batch call commits *issuance*, not completion by a consumer.
No Dataset, DataLoader, workers, prefetch queue, or training state is included.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from pytorch_lab.checkpoint import _atomic_write

FORMAT = "pytorch-lab-shuffled-index-stream"
VERSION = 1
MAX_SAMPLES = 10_000
MAX_EPOCH = 1_000_000
MAX_SNAPSHOT_BYTES = 1_000_000


class ShuffleStateError(ValueError):
    """The bounded index-stream contract was violated."""


def _integer(value, low: int, high: int, name: str) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ShuffleStateError(f"{name} must be an integer in [{low}, {high}]")


def _sha256(value) -> bool:
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class ShuffleConfig:
    # Caller supplies an ordered dataset/schema fingerprint, not a filename.
    # The index-only stream binds this value but cannot inspect actual samples.
    dataset_sha256: str
    samples: int = 23
    batch_size: int = 5
    seed: int = 42
    drop_last: bool = False

    def validate(self) -> None:
        _integer(self.samples, 0, MAX_SAMPLES, "samples")
        _integer(self.batch_size, 1, MAX_SAMPLES, "batch_size")
        _integer(self.seed, 0, 2**32 - 1, "seed")
        if type(self.drop_last) is not bool:
            raise ShuffleStateError("drop_last must be bool")
        if not _sha256(self.dataset_sha256):
            raise ShuffleStateError("dataset_sha256 must be a lowercase SHA-256 string")

    @property
    def usable_samples(self) -> int:
        if self.drop_last:
            return (self.samples // self.batch_size) * self.batch_size
        return self.samples


def _payload_digest(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ShuffledBatchStream:
    """Issue immutable batches from one permutation, advancing epochs explicitly.

    Single-threaded, exclusively owned object. Do not mutate private state.
    next_batch advances before returning; failure *after* return in downstream
    work does not roll it back. Save at caller-chosen quiescent emission points.
    """

    def __init__(self, config: ShuffleConfig):
        if type(config) is not ShuffleConfig:
            raise ShuffleStateError("config must be ShuffleConfig")
        config.validate()
        self._config = config
        self._generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self._permutation = tuple(
            torch.randperm(config.samples, generator=self._generator, device="cpu").tolist()
        )
        self._epoch = self._cursor = 0

    @property
    def config(self) -> ShuffleConfig:
        return self._config

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def exhausted(self) -> bool:
        return self.cursor == self.config.usable_samples

    def next_batch(self) -> tuple[int, ...]:
        """Return the next batch and commit issuance, or leave state unchanged."""
        if self.exhausted:
            raise StopIteration
        end = min(self.cursor + self.config.batch_size, self.config.usable_samples)
        batch = self._permutation[self.cursor : end]
        self._cursor = end
        return batch

    def advance_epoch(self) -> None:
        """Only after exhaustion; draw once, with no reseed or implicit skip.

        Empty and drop-all epochs are exhausted immediately. Preparing the next
        permutation uses a detached generator, so a failed draw changes nothing.
        """
        if not self.exhausted:
            raise ShuffleStateError("cannot advance an epoch with unissued batches")
        if self.epoch == MAX_EPOCH:
            raise ShuffleStateError("epoch limit reached")
        generator = torch.Generator(device="cpu")
        generator.set_state(self._generator.get_state())
        permutation = tuple(
            torch.randperm(self.config.samples, generator=generator, device="cpu").tolist()
        )
        self._generator, self._permutation = generator, permutation
        self._epoch += 1
        self._cursor = 0

    def snapshot(self) -> dict:
        """Detached JSON-only values, including the post-permutation RNG state."""
        payload = {
            "format": FORMAT,
            "version": VERSION,
            "torch_version": str(torch.__version__),
            "config": asdict(self.config),
            "epoch": self.epoch,
            "cursor": self.cursor,
            "permutation": list(self._permutation),
            "generator_state": self._generator.get_state().tolist(),
        }
        payload["content_sha256"] = _payload_digest(payload)
        return payload


def restore_snapshot(payload: dict, expected_config: ShuffleConfig | None = None):
    """Validate completely, then return a new owned stream without drawing RNG.

    The digest detects accidental corruption, not intentional manipulation.
    A structurally valid cursor/order/RNG edit with a recomputed digest can be
    accepted; validation is not proof of the history of emitted or trained data.
    """
    fields = {
        "format",
        "version",
        "torch_version",
        "config",
        "epoch",
        "cursor",
        "permutation",
        "generator_state",
        "content_sha256",
    }
    if type(payload) is not dict or payload.keys() != fields:
        raise ShuffleStateError("unexpected snapshot fields")
    if payload["format"] != FORMAT or type(payload["format"]) is not str:
        raise ShuffleStateError("unsupported format")
    if type(payload["version"]) is not int or payload["version"] != VERSION:
        raise ShuffleStateError("unsupported version")
    if type(payload["torch_version"]) is not str or payload["torch_version"] != str(
        torch.__version__
    ):
        raise ShuffleStateError("exact PyTorch version required")
    config_data = payload["config"]
    if type(config_data) is not dict or config_data.keys() != {
        "dataset_sha256",
        "samples",
        "batch_size",
        "seed",
        "drop_last",
    }:
        raise ShuffleStateError("unexpected config fields")
    config = ShuffleConfig(**config_data)
    config.validate()
    if expected_config is not None:
        if type(expected_config) is not ShuffleConfig:
            raise ShuffleStateError("expected_config must be ShuffleConfig")
        expected_config.validate()
        if config != expected_config:
            raise ShuffleStateError("snapshot differs from expected config/dataset identity")
    _integer(payload["epoch"], 0, MAX_EPOCH, "epoch")
    cursor = payload["cursor"]
    _integer(cursor, 0, config.usable_samples, "cursor")
    if cursor != config.usable_samples and cursor % config.batch_size:
        raise ShuffleStateError("cursor must be an issued-batch boundary")
    permutation = payload["permutation"]
    if type(permutation) is not list or len(permutation) != config.samples:
        raise ShuffleStateError("wrong permutation length/type")
    if any(type(i) is not int for i in permutation) or sorted(permutation) != list(
        range(config.samples)
    ):
        raise ShuffleStateError("permutation must contain every index exactly once")
    state = payload["generator_state"]
    generator = torch.Generator(device="cpu")
    if type(state) is not list or len(state) != generator.get_state().numel():
        raise ShuffleStateError("wrong generator state length/type")
    if any(type(i) is not int or not 0 <= i <= 255 for i in state):
        raise ShuffleStateError("generator state must contain bytes")
    try:
        generator.set_state(torch.tensor(state, dtype=torch.uint8, device="cpu"))
    except RuntimeError as error:
        raise ShuffleStateError("invalid CPU generator state") from error
    if generator.initial_seed() != config.seed:
        raise ShuffleStateError("generator seed differs from config")
    if not _sha256(payload["content_sha256"]) or payload["content_sha256"] != _payload_digest(
        payload
    ):
        raise ShuffleStateError("snapshot digest mismatch")
    stream = ShuffledBatchStream.__new__(ShuffledBatchStream)
    stream._config = config
    stream._epoch, stream._cursor = payload["epoch"], cursor
    stream._permutation = tuple(permutation)
    stream._generator = generator
    return stream


def save_shuffle_checkpoint(stream: ShuffledBatchStream, path: str | Path) -> None:
    """Publish a validated JSON snapshot without replacing an existing name.

    Uses the project's same-directory fsync + hard-link publication helper.
    Requires a local filesystem with hard links; no power-loss durability claim.
    """
    if type(stream) is not ShuffledBatchStream:
        raise ShuffleStateError("stream must be ShuffledBatchStream")
    payload = stream.snapshot()
    restore_snapshot(payload, stream.config)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(encoded) > MAX_SNAPSHOT_BYTES:
        raise ShuffleStateError("snapshot exceeds byte limit")
    _atomic_write(Path(path), lambda output: output.write(encoded))


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ShuffleStateError("duplicate JSON key")
        result[key] = value
    return result


def load_shuffle_checkpoint(path: str | Path, expected_config: ShuffleConfig | None = None):
    """Read bounded JSON from a trusted own file; never deserialize pickle.

    Input bounds and schema checks are not a hostile-file security sandbox.
    Pass independently known expected_config to bind the real dataset/order.
    """
    try:
        with Path(path).open("rb") as source:
            encoded = source.read(MAX_SNAPSHOT_BYTES + 1)
        if len(encoded) > MAX_SNAPSHOT_BYTES:
            raise ShuffleStateError("snapshot exceeds byte limit")
        payload = json.loads(encoded, object_pairs_hook=_unique_object)
    except (OSError, ValueError, RecursionError) as error:
        raise ShuffleStateError("cannot read bounded snapshot JSON") from error
    return restore_snapshot(payload, expected_config)
