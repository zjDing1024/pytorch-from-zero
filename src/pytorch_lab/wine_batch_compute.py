"""Original bounded CPU update-loop measurements; no speed ranking acceptance gate."""

import math
import platform
import statistics
import time

import torch
from torch.nn import functional as F

from pytorch_lab import wine
from pytorch_lab import wine_batch_budget as budget
from pytorch_lab.wine_observations import prepare_train, tensor_digest

REPEATS = 3


def train_timed(x, y, architecture, seed, arm, orders):
    wine._features(x)
    wine._labels(y, len(x))
    if x.shape != (107, 13) or arm not in budget.ARMS:
        raise ValueError("requires pinned train shape and known arm")
    if (
        not isinstance(orders, list)
        or len(orders) != 4
        or any(
            not isinstance(o, list)
            or any(type(i) is not int for i in o)
            or sorted(o) != list(range(107))
            for o in orders
        )
    ):
        raise ValueError("requires four permutations")
    size, steps, visits = budget.ARMS[arm]
    ids = [o[start : start + size] for o in orders for start in range(0, 107, size)][:steps]
    batches = [(x[i], y[i]) for i in ids]
    model = wine.WineClassifier(architecture, seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    model.train()
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    for xb, yb in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()
    cpu_ns = time.process_time_ns() - cpu_start
    wall_ns = time.perf_counter_ns() - wall_start
    if cpu_ns < 0 or wall_ns <= 0:
        raise ValueError("invalid clock duration")
    model.eval()
    with torch.no_grad():
        ce = F.cross_entropy(model(x), y).item()
    if not math.isfinite(ce) or any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("non-finite training result")
    optimizer.zero_grad(set_to_none=True)
    return model, {
        "arm": arm,
        "updates": len(batches),
        "sample_visits": sum(len(yb) for _, yb in batches),
        "row_positions": ids,
        "final_train_cross_entropy": ce,
        "final_parameters_sha256": tensor_digest(budget._flat(model)),
        "wall_ns": wall_ns,
        "process_cpu_ns": cpu_ns,
        "budget_matches": len(batches) == steps and sum(len(yb) for _, yb in batches) == visits,
    }


def run_experiment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x, y, preprocessing = prepare_train()
        configurations = []
        arms = list(budget.ARMS)
        for architecture in wine.ARCHITECTURES:
            for seed in wine.SEEDS:
                orders = budget.make_orders(seed)
                references = {
                    a: budget.train_run(x, y, architecture, seed, a, orders) for a in arms
                }
                observations = []
                for phase, repeat in [("warmup", 0)] + [("measured", i) for i in range(REPEATS)]:
                    order = arms[repeat:] + arms[:repeat]
                    for position, arm in enumerate(order):
                        model, row = train_timed(x, y, architecture, seed, arm, orders)
                        ref, info = references[arm]
                        row.update(phase=phase, repeat=repeat, position=position)
                        row["reference_exact"] = (
                            all(
                                torch.equal(v, ref.state_dict()[n])
                                for n, v in model.state_dict().items()
                            )
                            and row["final_train_cross_entropy"]
                            == info["final_train_cross_entropy"]
                            and row["row_positions"] == [t["row_positions"] for t in info["trace"]]
                        )
                        observations.append(row)
                summaries = []
                for arm in arms:
                    rows = [r for r in observations if r["arm"] == arm and r["phase"] == "measured"]
                    summaries.append(
                        {
                            "arm": arm,
                            "repeats": len(rows),
                            "wall_ns_min": min(r["wall_ns"] for r in rows),
                            "wall_ns_median": statistics.median(r["wall_ns"] for r in rows),
                            "wall_ns_max": max(r["wall_ns"] for r in rows),
                            "process_cpu_ns_median": statistics.median(
                                r["process_cpu_ns"] for r in rows
                            ),
                            "final_train_cross_entropy": rows[0]["final_train_cross_entropy"],
                        }
                    )
                configurations.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "observations": observations,
                        "summaries": summaries,
                        "all_checks_passed": all(
                            r["reference_exact"] and r["budget_matches"] for r in observations
                        ),
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
                "machine": platform.machine(),
                "wall_clock": vars(time.get_clock_info("perf_counter")),
                "cpu_clock": vars(time.get_clock_info("process_time")),
            },
            "protocol": {
                "repeats": REPEATS,
                "reference_runs": 18,
                "warmup_runs": 18,
                "measured_runs": 54,
                "total_updates": 1080,
                "training_sample_visits": 24390,
                "evaluation_forwards": 90,
                "evaluation_sample_visits": 9630,
                "scored_partitions": ["train"],
            },
            "preprocessing": preprocessing,
            "configurations": configurations,
            "all_checks_passed": all(c["all_checks_passed"] for c in configurations),
            "limits": [
                "Timers cover prepared-batch update loops, not end-to-end training.",
                "Hash/evaluation excluded; Python loop and clock overhead remain.",
                "Three within-process repeats are descriptive, not confidence intervals.",
                "Shared-host noise uncontrolled; CPU only, no GPU synchronization claim.",
                "Budgets differ; timings never select hyperparameters.",
            ],
        }
    finally:
        torch.set_num_threads(threads)
