"""Fixed-budget development curves with validation-only endpoint selection."""

import copy
import hashlib
import json
import math
import platform
from importlib.resources import files

import torch
from torch.nn import functional as F

from pytorch_lab import wine
from pytorch_lab.wine_observations import tensor_digest

EPOCHS = 20
CANDIDATES = (2, 5, 10, 20)
ARMS = ("constant", "step7")


def prepare_data():
    """Only train/validation rows enter tensors; fit only on training rows."""
    x, y = wine.load_wine()
    manifest = json.loads(files("pytorch_lab").joinpath("data/wine_split.json").read_text())
    if manifest["data_sha256"] != wine.DATA_SHA256:
        raise ValueError("data digest mismatch")
    train, validation = (manifest["row_ids"][key] for key in ("train", "validation"))
    if len(train) != 107 or len(validation) != 36 or len(set(train + validation)) != 143:
        raise ValueError("invalid development partition")
    scaler = wine.Standardizer.fit(x[train])
    return (
        scaler.transform(x[train]),
        y[train],
        scaler.transform(x[validation]),
        y[validation],
        {
            "train_ids": train,
            "validation_ids": validation,
            "mean": scaler.mean.tolist(),
            "scale": scaler.scale.tolist(),
            "fit_partition": "train",
        },
    )


def evaluate(model, x, y):
    """Evaluation changes neither gradients nor the caller's train/eval mode."""
    mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            value = F.cross_entropy(model(x), y).item()
    finally:
        model.train(mode)
    if not math.isfinite(value):
        raise ValueError("nonfinite evaluation")
    return value


def digest_model(model):
    return tensor_digest(torch.cat([p.detach().reshape(-1) for p in model.parameters()]))


def train_run(x, y, vx, vy, architecture, seed, arm):
    if architecture not in wine.ARCHITECTURES or seed not in wine.SEEDS or arm not in ARMS:
        raise ValueError("unknown frozen configuration")
    model = wine.WineClassifier(architecture, seed)
    model.train()
    initial = digest_model(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 7, gamma=0.5) if arm == "step7" else None
    generator = torch.Generator().manual_seed(seed)
    curves, snapshots, order_log = [], {}, []
    updates = visits = 0
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(107, generator=generator).tolist()
        order_log.append(order)
        for offset in range(0, 107, 16):
            ids = order[offset : offset + 16]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[ids]), y[ids])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("nonfinite or missing gradient")
            lr_used = optimizer.param_groups[0]["lr"]
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("nonfinite parameters")
            updates += 1
            visits += len(ids)
        optimizer.zero_grad(set_to_none=True)
        curves.append(
            {
                "epoch": epoch,
                "updates": updates,
                "sample_visits": visits,
                "train_ce": evaluate(model, x, y),
                "validation_ce": evaluate(model, vx, vy),
                "lr_last_used": lr_used,
                "lr_next": optimizer.param_groups[0]["lr"],
                "parameters_sha256": digest_model(model),
            }
        )
        if epoch in CANDIDATES:
            snapshots[epoch] = copy.deepcopy(model.state_dict())
    # Verify retained endpoint states after the entire run has finished.
    checks = []
    for epoch, state in snapshots.items():
        model.load_state_dict(state)
        point = curves[epoch - 1]
        checks.append(
            digest_model(model) == point["parameters_sha256"]
            and evaluate(model, vx, vy) == point["validation_ce"]
        )
    if not all(checks):
        raise AssertionError("endpoint snapshot was changed or does not reproduce validation CE")
    return {
        "architecture": architecture,
        "seed": seed,
        "arm": arm,
        "initial_parameters_sha256": initial,
        "order_sha256": hashlib.sha256(json.dumps(order_log).encode()).hexdigest(),
        "curve": curves,
        "endpoint_reload_checks": checks,
    }


def select_endpoint(runs):
    """Require the entire frozen candidate grid; never discard failed seeds."""
    expected = {(a, s, b) for a in wine.ARCHITECTURES for s in wine.SEEDS for b in ARMS}
    keys = [(r["architecture"], r["seed"], r["arm"]) for r in runs]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("missing or duplicate frozen run")
    indexed = dict(zip(keys, runs, strict=True))
    ranking = []
    for architecture in wine.ARCHITECTURES:
        for arm in ARMS:
            for epoch in CANDIDATES:
                values = []
                for seed in wine.SEEDS:
                    curve = indexed[architecture, seed, arm]["curve"]
                    if [p["epoch"] for p in curve] != list(range(1, EPOCHS + 1)):
                        raise ValueError("incomplete curve")
                    if any(
                        not math.isfinite(p[k])
                        for p in curve
                        for k in ("train_ce", "validation_ce")
                    ):
                        raise ValueError("nonfinite curve")
                    values.append(curve[epoch - 1]["validation_ce"])
                ranking.append(
                    {
                        "architecture": architecture,
                        "arm": arm,
                        "epoch": epoch,
                        "mean_validation_ce": sum(v / len(values) for v in values),
                        "seed_validation_ce": values,
                    }
                )
    ranking.sort(
        key=lambda r: (
            r["mean_validation_ce"],
            r["epoch"],
            wine.ARCHITECTURES.index(r["architecture"]),
            ARMS.index(r["arm"]),
        )
    )
    return {"selected": ranking[0], "all_candidates": ranking}


def run_experiment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x, y, vx, vy, preprocessing = prepare_data()
        runs = [
            train_run(x, y, vx, vy, a, s, b)
            for a in wine.ARCHITECTURES
            for s in wine.SEEDS
            for b in ARMS
        ]
        paired = all(
            runs[i][k] == runs[i + 1][k]
            for i in range(0, len(runs), 2)
            for k in ("initial_parameters_sha256", "order_sha256")
        )
        if not paired:
            raise AssertionError("paired controls differ")
        return {
            "protocol": "wine-learning-curves-v1",
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
                "dtype": "float64",
                "threads": 1,
            },
            "preprocessing": preprocessing,
            "runs": runs,
            "selection": select_endpoint(runs),
            "paired_controls_equal": paired,
            "all_checks_passed": paired and all(all(r["endpoint_reload_checks"]) for r in runs),
            "work": {
                "optimizer_updates": sum(r["curve"][-1]["updates"] for r in runs),
                "training_sample_visits": sum(r["curve"][-1]["sample_visits"] for r in runs),
                "curve_evaluation_calls": 2 * sum(len(r["curve"]) for r in runs),
                "curve_evaluation_sample_visits": (len(y) + len(vy))
                * sum(len(r["curve"]) for r in runs),
                "reload_evaluation_calls": sum(len(r["endpoint_reload_checks"]) for r in runs),
                "reload_evaluation_sample_visits": len(vy)
                * sum(len(r["endpoint_reload_checks"]) for r in runs),
            },
            "limitations": [
                "Reused development validation; no new held-out claim.",
                "No test scoring; no uncertainty over dataset splits.",
                "Assistant implementation is not learner mastery.",
            ],
        }
    finally:
        torch.set_num_threads(threads)
