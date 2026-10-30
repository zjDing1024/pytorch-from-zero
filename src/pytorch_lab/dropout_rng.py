"""Single-threaded CPU stochastic forward/backward replay, not a training checkpoint."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class DropoutConfig:
    seed: int = 42
    model_seed: int = 17
    p: float = 0.35
    training: bool = True

    def __post_init__(self):
        for name in ("seed", "model_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError(f"{name} must be an integer in [0, 2**63)")
        if type(self.p) not in (float, int) or not math.isfinite(self.p) or not 0 <= self.p <= 1:
            raise ValueError("p must be finite and in [0, 1]")
        if type(self.training) is not bool:
            raise ValueError("training must be bool")


def cpu_seed_state(seed: int) -> torch.Tensor:
    return torch.Generator(device="cpu").manual_seed(seed).get_state()


def digest(state: torch.Tensor) -> str:
    return hashlib.sha256(bytes(state.tolist())).hexdigest()


def fixture(step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """No RNG: five examples, four inputs and two regression targets."""
    x = torch.arange(20, dtype=torch.float64).reshape(5, 4) / 8 - 1 + step / 32
    y = torch.arange(10, dtype=torch.float64).reshape(5, 2) / 16 - 0.25
    return x, y


def make_model(config: DropoutConfig) -> nn.Sequential:
    # Module constructors consume the default RNG. Keep this separate from replay state.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(cpu_seed_state(config.model_seed))
        model = nn.Sequential(
            nn.Linear(4, 8, dtype=torch.float64),
            nn.Dropout(config.p),
            nn.Linear(8, 2, dtype=torch.float64),
        )
    return model.train(config.training)


MAX_STEPS = 1_000_000


class DropoutReplay:
    """Own the RNG of a fixed, buffer-free model; never perform optimizer updates.

    The context temporarily changes process-global CPU RNG. Do not use concurrently
    with other random consumers. This is not arbitrary-module failure rollback.
    """

    def __init__(self, config: DropoutConfig | None = None):
        if config is None:
            config = DropoutConfig()
        if not isinstance(config, DropoutConfig):
            raise ValueError("config must be DropoutConfig")
        self.config = config
        self.model = make_model(config)
        self._rng = cpu_seed_state(config.seed)
        self.steps = 0

    def _check_mode(self):
        if (
            type(self.model) is not nn.Sequential
            or len(self.model) != 3
            or type(self.model[0]) is not nn.Linear
            or type(self.model[1]) is not nn.Dropout
            or type(self.model[2]) is not nn.Linear
        ):
            raise ValueError("model architecture differs from fixture")
        shapes = {"0.weight": (8, 4), "0.bias": (8,), "2.weight": (2, 8), "2.bias": (2,)}
        weights = self.model.state_dict()
        if set(weights) != set(shapes):
            raise ValueError("model state differs from fixture")
        for name, value in weights.items():
            if (
                tuple(value.shape) != shapes[name]
                or value.dtype != torch.float64
                or value.device.type != "cpu"
                or not torch.isfinite(value).all()
            ):
                raise ValueError("model requires finite CPU float64 fixture weights")
        if any(m.training != self.config.training for m in self.model.modules()):
            raise ValueError("model mode differs from config")
        if self.model[1].p != self.config.p:
            raise ValueError("Dropout p differs from config")

    def step(self) -> dict:
        self._check_mode()
        if self.steps >= MAX_STEPS:
            raise ValueError("maximum probe steps reached")
        # A prior failed backward can leave gradients; clear them before each attempt.
        self.model.zero_grad(set_to_none=True)
        x, y = fixture(self.steps)
        x.requires_grad_(True)
        before = self._rng.clone()
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(before)
            output = self.model(x)
            loss = (output - y).square().sum()
            loss.backward()
            after = torch.get_rng_state().clone()
            result = {
                "step": self.steps,
                "output": output.detach().tolist(),
                "loss": loss.item(),
                "input_grad": x.grad.tolist(),
                "parameter_grads": {n: p.grad.tolist() for n, p in self.model.named_parameters()},
                "rng_before_digest": digest(before),
                "rng_after_digest": digest(after),
            }
            if not all(torch.isfinite(p.grad).all() for p in self.model.parameters()):
                raise ValueError("non-finite gradients")
            if (
                not torch.isfinite(output).all()
                or not torch.isfinite(loss)
                or not torch.isfinite(x.grad).all()
            ):
                raise ValueError("non-finite forward")
        # Only a completed probe advances private state; global state is restored above.
        self._rng = after
        self.steps += 1
        return result

    def snapshot(self) -> dict:
        self._check_mode()
        return {
            "format": "cpu-dropout-replay-v1",
            "torch_version": torch.__version__,
            "fixture": "arange-5x4-step-offset-sum-mse-v1",
            "config": asdict(self.config),
            "steps": self.steps,
            "model": {n: v.tolist() for n, v in self.model.state_dict().items()},
            "rng": self._rng.tolist(),
        }

    @classmethod
    def from_snapshot(cls, state: dict, *, expected_config: DropoutConfig | None = None):
        keys = {"format", "torch_version", "fixture", "config", "steps", "model", "rng"}
        if type(state) is not dict or set(state) != keys:
            raise ValueError("invalid snapshot keys")
        if state["format"] != "cpu-dropout-replay-v1":
            raise ValueError("unsupported snapshot format")
        if state["torch_version"] != torch.__version__:
            raise ValueError("snapshot requires the same torch version")
        if state["fixture"] != "arange-5x4-step-offset-sum-mse-v1":
            raise ValueError("fixture identity mismatch")
        if type(state["config"]) is not dict or set(state["config"]) != set(
            asdict(DropoutConfig())
        ):
            raise ValueError("invalid config")
        config = DropoutConfig(**state["config"])
        if expected_config is not None and config != expected_config:
            raise ValueError("expected config mismatch")
        if type(state["steps"]) is not int or not 0 <= state["steps"] <= MAX_STEPS:
            raise ValueError("invalid step counter")
        rng = state["rng"]
        if type(rng) is not list or len(rng) != len(cpu_seed_state(0)):
            raise ValueError("invalid RNG length")
        if any(type(x) is not int or not 0 <= x <= 255 for x in rng):
            raise ValueError("invalid RNG bytes")
        rng = torch.tensor(rng, dtype=torch.uint8)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(rng)
        except RuntimeError as error:
            raise ValueError("invalid RNG state") from error
        obj = cls(config)  # Construct first, then restore the captured replay RNG.
        model = state["model"]
        if type(model) is not dict or set(model) != set(obj.model.state_dict()):
            raise ValueError("invalid model keys")
        restored = {}
        for name, ref in obj.model.state_dict().items():

            def valid_numbers(value, shape):
                if shape:
                    return (
                        type(value) is list
                        and len(value) == shape[0]
                        and all(valid_numbers(x, shape[1:]) for x in value)
                    )
                return type(value) in (int, float) and math.isfinite(value)

            if not valid_numbers(model[name], tuple(ref.shape)):
                raise ValueError("invalid model numeric schema")
            try:
                value = torch.tensor(model[name], dtype=torch.float64)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError("invalid model values") from error
            if value.shape != ref.shape or not torch.isfinite(value).all():
                raise ValueError("invalid model shape or non-finite values")
            restored[name] = value
        obj.model.load_state_dict(restored)
        obj._rng = rng.clone()
        obj.steps = state["steps"]
        return obj

    def save(self, path: str | Path):
        payload = self.snapshot()
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        envelope = {"payload": payload, "sha256": hashlib.sha256(raw.encode()).hexdigest()}
        # A failed write may leave an unusable file. No durability/atomic-save claim.
        with Path(path).open("x", encoding="utf-8") as handle:
            json.dump(envelope, handle, allow_nan=False)

    @classmethod
    def load(cls, path: str | Path, *, expected_config: DropoutConfig | None = None):
        path = Path(path)
        if path.stat().st_size > 100_000:
            raise ValueError("snapshot too large")
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if type(envelope) is not dict or set(envelope) != {"payload", "sha256"}:
            raise ValueError("invalid envelope")
        raw = json.dumps(
            envelope["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if hashlib.sha256(raw.encode()).hexdigest() != envelope["sha256"]:
            raise ValueError("snapshot checksum mismatch")
        return cls.from_snapshot(envelope["payload"], expected_config=expected_config)


def mode_probe(seed: int = 42, p: float = 0.35) -> dict:
    """Compare module mode and gradient recording under the same initial CPU state."""
    config = DropoutConfig(seed=seed, p=p)
    model = make_model(config)
    x, _ = fixture(0)
    traces = {}
    for mode, training, grad_enabled in (
        ("train", True, True),
        ("no_grad", True, False),
        ("eval", False, True),
    ):
        model.train(training)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(cpu_seed_state(seed))
            before = torch.get_rng_state()
            with torch.set_grad_enabled(grad_enabled):
                output = model(x)
            traces[mode] = {
                "output": output.detach().tolist(),
                "requires_grad": output.requires_grad,
                "rng_changed": not torch.equal(before, torch.get_rng_state()),
            }
    return traces
