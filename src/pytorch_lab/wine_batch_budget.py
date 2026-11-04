"""Original train-only batch-grouping experiment with two explicit budgets."""

import math
import platform

import torch
from torch.nn import functional as F

from pytorch_lab import wine
from pytorch_lab.wine_observations import prepare_train, tensor_digest

ARMS = {
    "batch16_14updates": (16, 14, 214),
    "batch32_2epochs": (32, 8, 214),
    "batch32_14updates": (32, 14, 385),
}


def make_orders(seed):
    """Freeze four permutations independently of batch grouping and global RNG."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    generator = torch.Generator().manual_seed(seed)
    return [torch.randperm(107, generator=generator).tolist() for _ in range(4)]


def _flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def train_run(x, y, architecture, seed, arm, orders):
    wine._features(x)
    wine._labels(y, len(x))
    if x.shape != (107, 13) or arm not in ARMS:
        raise ValueError("requires pinned training shape and known arm")
    if (
        not isinstance(orders, list)
        or len(orders) != 4
        or any(
            not isinstance(order, list)
            or any(type(i) is not int for i in order)
            or sorted(order) != list(range(107))
            for order in orders
        )
    ):
        raise ValueError("orders must contain four exact permutations")
    batch_size, target_updates, target_visits = ARMS[arm]
    model = wine.WineClassifier(architecture, seed)
    initial = tensor_digest(_flat(model))
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    trace, visits, online_sum = [], 0, 0.0
    for epoch, order in enumerate(orders):
        for offset in range(0, len(order), batch_size):
            if len(trace) == target_updates:
                break
            ids = order[offset : offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[ids]), y[ids])
            if not torch.isfinite(loss):
                raise ValueError("non-finite loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("missing or non-finite gradient")
            before = _flat(model).clone()
            optimizer.step()
            after = _flat(model)
            if not torch.isfinite(after).all():
                raise ValueError("non-finite parameters")
            visits += len(ids)
            online_sum += loss.item() * len(ids)
            trace.append(
                {
                    "step": len(trace) + 1,
                    "epoch": epoch + 1,
                    "row_positions": ids,
                    "batch_size": len(ids),
                    "loss_before_update": loss.item(),
                    "actual_update_l2": float(torch.linalg.vector_norm(after - before)),
                    "parameters_after_sha256": tensor_digest(after),
                }
            )
    model.eval()
    with torch.no_grad():
        final_ce = F.cross_entropy(model(x), y).item()
    if not math.isfinite(final_ce):
        raise ValueError("non-finite final loss")
    optimizer.zero_grad(set_to_none=True)
    return model, {
        "arm": arm,
        "batch_size": batch_size,
        "initial_parameters_sha256": initial,
        "final_parameters_sha256": tensor_digest(_flat(model)),
        "updates": len(trace),
        "sample_visits": visits,
        "sample_weighted_online_cross_entropy": online_sum / visits,
        "final_train_cross_entropy": final_ce,
        "trace": trace,
        "budget_matches": len(trace) == target_updates and visits == target_visits,
    }


def run_experiment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x, y, preprocessing = prepare_train()
        configs = []
        for architecture in wine.ARCHITECTURES:
            for seed in wine.SEEDS:
                orders = make_orders(seed)
                initial = wine.WineClassifier(architecture, seed)
                initial.eval()
                with torch.no_grad():
                    initial_ce = F.cross_entropy(initial(x), y).item()
                if not math.isfinite(initial_ce):
                    raise ValueError("non-finite initial loss")
                runs = [train_run(x, y, architecture, seed, arm, orders)[1] for arm in ARMS]
                sequences = [[i for t in r["trace"] for i in t["row_positions"]] for r in runs]
                checks = {
                    "identical_initial_parameters": len(
                        {r["initial_parameters_sha256"] for r in runs}
                    )
                    == 1,
                    "exact_budgets": all(r["budget_matches"] for r in runs),
                    "equal_visits_identical_samples": sequences[0] == sequences[1],
                    "equal_updates_shared_prefix": sequences[2][:214] == sequences[0],
                    "batch32_common_prefix": runs[1]["trace"] == runs[2]["trace"][:8],
                }
                configs.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "orders": orders,
                        "initial_train_cross_entropy": initial_ce,
                        "runs": runs,
                        "checks": checks,
                        "all_checks_passed": all(checks.values()),
                    }
                )
        return {
            "schema_version": 1,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
                "dtype": "float64",
                "threads": 1,
            },
            "preprocessing": preprocessing,
            "protocol": {
                "scored_partitions": ["train"],
                "learning_rate": 0.05,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "unique_runs": 18,
                "optimizer_updates": sum(r["updates"] for c in configs for r in c["runs"]),
                "training_sample_visits": sum(
                    r["sample_visits"] for c in configs for r in c["runs"]
                ),
                "evaluation_forwards": 24,
                "evaluation_sample_visits": 2568,
            },
            "configurations": configs,
            "all_checks_passed": all(c["all_checks_passed"] for c in configs),
            "limits": [
                "Two budgets answer different questions; neither equals compute time.",
                "Momentum and weight decay apply per update; not pure gradient-noise attribution.",
                "No heldout scoring, selection, universal ranking, or convergence claim.",
                "Online CE mixes changing models and is not endpoint CE.",
            ],
        }
    finally:
        torch.set_num_threads(threads)
