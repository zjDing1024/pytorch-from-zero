"""Training-only measurements on a fixed two-epoch prefix of the existing Wine protocol.

Original implementation; diagnostics are read-only, not a causal fault classifier.
The existing full Wine experiment and its held-out results are never rerun here.
"""

import hashlib
import json
import math
import platform
from dataclasses import asdict
from importlib.resources import files

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lab import wine

PREFIX_EPOCHS = 2
ATOL = RTOL = 1e-12


def tensor_digest(tensor: Tensor) -> str:
    """Canonical dtype/shape/raw CPU bytes, without a NumPy dependency."""
    value = tensor.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)]).encode()
    payload = bytes(value.reshape(-1).view(torch.uint8).tolist())
    return hashlib.sha256(header + b"\0" + payload).hexdigest()


def prepare_train() -> tuple[Tensor, Tensor, dict]:
    """Read only pinned training IDs for preprocessing and optimization.

    Loading pinned bundled CSV bytes is not scoring held-out examples. Do not
    derive a new partition from labels or fit any statistic on held-out features.
    """
    x, y = wine.load_wine()
    manifest = json.loads(files("pytorch_lab").joinpath("data/wine_split.json").read_text())
    if manifest["data_sha256"] != wine.DATA_SHA256:
        raise ValueError("pinned split data digest mismatch")
    ids = manifest["row_ids"]["train"]
    if len(ids) != 107 or len(set(ids)) != 107 or any(type(i) is not int for i in ids):
        raise ValueError("invalid pinned training IDs")
    train_x, train_y = x[ids], y[ids]
    scaler = wine.Standardizer.fit(train_x)
    return (
        scaler.transform(train_x),
        train_y,
        {
            "row_ids": ids,
            "mean": scaler.mean.tolist(),
            "scale": scaler.scale.tolist(),
            "fit_partition": "train",
            "correction": 0,
        },
    )


def observe_parameter(
    before: Tensor,
    parameter: Tensor,
    gradient: Tensor | None,
    momentum: Tensor,
    learning_rate: float,
) -> dict:
    """Measure actual state, never substitute -lr*gradient for an observed update."""
    with torch.no_grad():
        delta = parameter.detach() - before
        expected_delta = -learning_rate * momentum
        norm = float(torch.linalg.vector_norm(before))
        delta_norm = float(torch.linalg.vector_norm(delta))
        grad_norm = None if gradient is None else float(torch.linalg.vector_norm(gradient))
        result = {
            "gradient_present": gradient is not None,
            "gradient_l2": grad_norm,
            "parameter_before_l2": norm,
            "actual_delta_l2": delta_norm,
            "update_to_parameter_ratio": delta_norm / norm if norm else None,
            "ratio_status": "defined" if norm else "undefined_zero_parameter_norm",
            "momentum_after_l2": float(torch.linalg.vector_norm(momentum)),
            "sgd_delta_max_abs_residual": float((delta - expected_delta).abs().max()),
            "sgd_delta_matches": bool(torch.allclose(delta, expected_delta, atol=ATOL, rtol=RTOL)),
        }
    numbers = [v for v in result.values() if type(v) in (float, int)]
    if not all(math.isfinite(v) for v in numbers):
        raise ValueError("non-finite observation")
    return result


def train_prefix(
    train_x: Tensor, train_y: Tensor, architecture: str, seed: int, *, observe: bool
) -> tuple[wine.WineClassifier, dict, list[dict]]:
    """Fixed protocol; common audit hashes remain in both diagnostic modes.

    Hashing is a verifier, not a claim that the reference has zero CPU overhead.
    Legacy fit_candidate supplies a separate genuinely uninstrumented reference.
    """
    if type(observe) is not bool:
        raise ValueError("observe must be bool")
    wine._features(train_x)
    wine._labels(train_y, len(train_x))
    if train_x.shape != (107, 13):
        raise ValueError("requires the pinned 107 x 13 training partition")
    config = wine.WineConfig(epochs=PREFIX_EPOCHS)
    model = wine.WineClassifier(architecture, seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    trace, history, updates, samples = [], [], 0, 0
    model.train()
    for epoch in range(1, config.epochs + 1):
        online_sum = 0.0
        for batch_index, (xb, yb) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            named = dict(model.named_parameters())
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in named.values()):
                raise ValueError("missing or non-finite training gradient")
            audit = {
                "batch_x": tensor_digest(xb),
                "batch_y": tensor_digest(yb),
                "logits": tensor_digest(logits),
                "loss": tensor_digest(loss),
                "gradients": {n: tensor_digest(p.grad) for n, p in named.items()},
                "parameters_before": {n: tensor_digest(p) for n, p in named.items()},
            }
            # detach alone shares storage; clone is required before the in-place optimizer step.
            before = {n: p.detach().clone() for n, p in named.items()} if observe else {}
            gradients = {n: p.grad.detach().clone() for n, p in named.items()} if observe else {}
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in named.values()):
                raise ValueError("non-finite parameter")
            buffers = {n: optimizer.state[p]["momentum_buffer"] for n, p in named.items()}
            if any(not torch.isfinite(v).all() for v in buffers.values()):
                raise ValueError("non-finite momentum")
            audit.update(
                {
                    "parameters_after": {n: tensor_digest(p) for n, p in named.items()},
                    "momentum_after": {n: tensor_digest(v) for n, v in buffers.items()},
                    "loader_rng_after": tensor_digest(generator.get_state()),
                }
            )
            updates += 1
            samples += len(yb)
            online_sum += loss.item() * len(yb)
            record = {
                "step": updates,
                "epoch": epoch,
                "batch_index": batch_index,
                "batch_size": len(yb),
                "pre_update_batch_cross_entropy": loss.item(),
                "audit": audit,
            }
            if observe:
                record["parameters"] = {
                    n: observe_parameter(
                        before[n], p, gradients[n], buffers[n], config.learning_rate
                    )
                    for n, p in named.items()
                }
            trace.append(record)
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
    return (
        model,
        {
            "architecture": architecture,
            "seed": seed,
            "parameters": sum(p.numel() for p in model.parameters()),
            "updates": updates,
            "samples_seen": samples,
            "history": history,
        },
        trace,
    )


def _run_experiment() -> dict:
    train_x, train_y, preprocessing = prepare_train()
    config = wine.WineConfig(epochs=PREFIX_EPOCHS)
    runs = []
    for architecture in wine.ARCHITECTURES:
        for seed in wine.SEEDS:
            observed, info, trace = train_prefix(train_x, train_y, architecture, seed, observe=True)
            plain, plain_info, plain_trace = train_prefix(
                train_x, train_y, architecture, seed, observe=False
            )
            legacy, legacy_info = wine.fit_candidate(train_x, train_y, architecture, seed, config)
            equal_state = lambda a, b: all(  # noqa: E731
                torch.equal(v, b.state_dict()[n]) for n, v in a.state_dict().items()
            )
            checks = {
                "audit_trajectory_equal": [
                    {k: v for k, v in row.items() if k != "parameters"} for row in trace
                ]
                == plain_trace,
                "diagnostics_disabled_final_equal": equal_state(observed, plain),
                "legacy_uninstrumented_final_equal": equal_state(observed, legacy),
                "legacy_history_and_budget_equal": info == plain_info == legacy_info,
                "exact_prefix_budget": info["updates"] == 14 and info["samples_seen"] == 214,
                "actual_updates_match_momentum": all(
                    p["sgd_delta_matches"] for row in trace for p in row["parameters"].values()
                ),
                "gradients_present": all(
                    p["gradient_present"] for row in trace for p in row["parameters"].values()
                ),
            }
            runs.append(
                {
                    **info,
                    "trace": trace,
                    "checks": checks,
                    "all_checks_passed": all(checks.values()),
                }
            )
    checks = {
        "all_run_checks": all(run["all_checks_passed"] for run in runs),
        "train_centered": bool(train_x.mean(0).abs().max() < 1e-12),
        "train_unit_variance": bool((train_x.var(0, correction=0) - 1).abs().max() < 1e-12),
    }
    return {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float64",
            "threads": 1,
        },
        "dataset": {"name": "UCI Wine", "sha256": wine.DATA_SHA256, "license": "CC-BY-4.0"},
        "protocol": {
            "config": asdict(config),
            "seeds": list(wine.SEEDS),
            "architectures": list(wine.ARCHITECTURES),
            "split_salt": wine.SPLIT_SALT,
            "scored_partitions": ["train"],
            "historical_full_epochs": 120,
            "prefix_only": True,
            "comparison_passes": 3,
            "total_optimizer_updates": 252,
            "total_training_sample_visits": 3852,
            "delta_atol": ATOL,
            "delta_rtol": RTOL,
        },
        "preprocessing": preprocessing,
        "runs": runs,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "limits": [
            "Two-epoch prefix, not a rerun of the 120-epoch benchmark or a convergence test.",
            "No validation/test scoring, tuning, architecture selection or accuracy claim.",
            "Both diagnostic modes use audit hashes; legacy fit is an uninstrumented reference.",
            "Within-environment exact comparison is not cross-device/version reproducibility.",
            "Norms and ratios describe symptoms, not unique root causes or layer quality.",
            "CPU measurement overhead is not benchmarked; no GPU, AMP or distributed claim.",
        ],
    }


def run_experiment() -> dict:
    """Restore caller thread count even on failure; local generators isolate CPU RNG."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _run_experiment()
    finally:
        torch.set_num_threads(previous_threads)
