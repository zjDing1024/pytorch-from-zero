"""Original paired, training-only momentum experiment with fixed nominal step size.

Reuse pinned preprocessing/digests, not the historical observation loop. Pairing
controls initialization and examples, not gradients once parameters diverge.
"""

import math
import platform

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lab import wine
from pytorch_lab.wine_observations import prepare_train, tensor_digest

EPOCHS = 2
MOMENTA = (0.0, 0.9)
ATOL = RTOL = 1e-12


def _flat(values):
    return torch.cat([v.reshape(-1) for v in values])


def _close(a, b):
    return bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL))


def train_pair(train_x, train_y, architecture, seed):
    """Train two real SGD optimizers on one common stream; measure cloned state."""
    wine._features(train_x)
    wine._labels(train_y, len(train_x))
    if train_x.shape != (107, 13):
        raise ValueError("requires pinned 107 x 13 training partition")
    config = wine.WineConfig(epochs=EPOCHS)
    models = [wine.WineClassifier(architecture, seed) for _ in MOMENTA]
    optimizers = [
        torch.optim.SGD(
            m.parameters(), lr=config.learning_rate, momentum=mu, weight_decay=config.weight_decay
        )
        for m, mu in zip(models, MOMENTA, strict=True)
    ]
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    initial = [_flat([p.detach() for p in m.parameters()]).clone() for m in models]
    checks = {"identical_initial_parameters": torch.equal(*initial)}
    histories = [[], []]
    traces = []
    path_lengths = [0.0, 0.0]
    first_effective = None
    steps = samples = 0
    for epoch in range(1, EPOCHS + 1):
        online = [0.0, 0.0]
        for batch_index, (xb, yb) in enumerate(loader):
            arms, before_vectors, gradient_vectors, delta_vectors = [], [], [], []
            for arm, (model, optimizer, mu) in enumerate(
                zip(models, optimizers, MOMENTA, strict=True)
            ):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                loss.backward()
                named = dict(model.named_parameters())
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in named.values()):
                    raise ValueError("missing or non-finite gradient")
                before = {n: p.detach().clone() for n, p in named.items()}
                gradients = {n: p.grad.detach().clone() for n, p in named.items()}
                effective = {n: gradients[n] + config.weight_decay * before[n] for n in named}
                expected = {}
                for n, p in named.items():
                    old = optimizer.state.get(p, {}).get("momentum_buffer")
                    expected[n] = (
                        effective[n] if old is None else mu * old.detach().clone() + effective[n]
                    )
                optimizer.step()
                before_vector = _flat(before.values())
                raw_gradient = _flat(gradients.values())
                delta = _flat([p.detach() - before[n] for n, p in named.items()])
                expected_delta = -config.learning_rate * _flat(expected.values())
                actual_buffers = [
                    optimizer.state.get(p, {}).get("momentum_buffer") for p in named.values()
                ]
                buffer_ok = (
                    all(v is None for v in actual_buffers)
                    if mu == 0
                    else all(
                        v is not None and _close(v, expected[n])
                        for (n, _), v in zip(named.items(), actual_buffers, strict=True)
                    )
                )
                if not all(torch.isfinite(p).all() for p in named.values()):
                    raise ValueError("non-finite parameter")
                delta_l2 = float(torch.linalg.vector_norm(delta))
                path_lengths[arm] += delta_l2
                online[arm] += loss.item() * len(yb)
                arms.append(
                    {
                        "momentum": mu,
                        "pre_update_batch_cross_entropy": loss.item(),
                        "raw_gradient_l2": float(torch.linalg.vector_norm(raw_gradient)),
                        "effective_gradient_l2": float(
                            torch.linalg.vector_norm(_flat(effective.values()))
                        ),
                        "actual_delta_l2": delta_l2,
                        "cumulative_path_length": path_lengths[arm],
                        "buffer_l2": None
                        if mu == 0
                        else float(torch.linalg.vector_norm(_flat(actual_buffers))),
                        "buffer_matches_recurrence": buffer_ok,
                        "delta_matches_sgd": _close(delta, expected_delta),
                        "delta_max_abs_residual": float((delta - expected_delta).abs().max()),
                        "parameters_before_sha256": tensor_digest(before_vector),
                        "raw_gradients_sha256": tensor_digest(raw_gradient),
                        "parameters_after_sha256": tensor_digest(
                            _flat([p.detach() for p in named.values()])
                        ),
                    }
                )
                before_vectors.append(before_vector)
                gradient_vectors.append(raw_gradient)
                delta_vectors.append(delta)
                if steps == 0 and arm == 0:
                    first_effective = _flat(effective.values()).clone()
            steps += 1
            samples += len(yb)
            if steps == 1:
                checks["identical_first_step"] = torch.equal(*delta_vectors)
            if steps == 2:
                checks["identical_second_step_preparameters"] = torch.equal(*before_vectors)
                checks["identical_second_step_raw_gradients"] = torch.equal(*gradient_vectors)
                checks["second_step_momentum_explains_delta_difference"] = _close(
                    delta_vectors[1] - delta_vectors[0],
                    -config.learning_rate * MOMENTA[1] * first_effective,
                )
            traces.append(
                {
                    "step": steps,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "batch_size": len(yb),
                    "batch_x_sha256": tensor_digest(xb),
                    "batch_y_sha256": tensor_digest(yb),
                    "arms": arms,
                    "paired_parameter_distance_l2": float(
                        torch.linalg.vector_norm(
                            _flat([p.detach() for p in models[1].parameters()])
                            - _flat([p.detach() for p in models[0].parameters()])
                        )
                    ),
                    "paired_raw_gradients_equal": torch.equal(*gradient_vectors),
                }
            )
        for arm, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                ce = F.cross_entropy(model(train_x), train_y).item()
            if not math.isfinite(ce):
                raise ValueError("non-finite final training loss")
            histories[arm].append(
                {
                    "epoch": epoch,
                    "online_cross_entropy": online[arm] / len(train_y),
                    "final_train_cross_entropy": ce,
                }
            )
    checks.update(
        {
            "equal_exact_budget": steps == 14 and samples == 214,
            "all_sgd_equations": all(a["delta_matches_sgd"] for t in traces for a in t["arms"]),
            "all_buffer_recurrences": all(
                a["buffer_matches_recurrence"] for t in traces for a in t["arms"]
            ),
        }
    )
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    outcomes = [
        {
            "momentum": mu,
            "updates": steps,
            "sample_visits": samples,
            "history": history,
            "cumulative_path_length": length,
        }
        for mu, history, length in zip(MOMENTA, histories, path_lengths, strict=True)
    ]
    return models, {
        "architecture": architecture,
        "seed": seed,
        "arms": outcomes,
        "trace": traces,
        "train_ce_difference_momentum_minus_plain": histories[1][-1]["final_train_cross_entropy"]
        - histories[0][-1]["final_train_cross_entropy"],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def run_experiment():
    """Run fixed protocol, restoring caller thread count on success or failure."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x, y, preprocessing = prepare_train()
        runs = [
            train_pair(x, y, architecture, seed)[1]
            for architecture in wine.ARCHITECTURES
            for seed in wine.SEEDS
        ]
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
                "epochs": EPOCHS,
                "momenta": list(MOMENTA),
                "learning_rate": 0.05,
                "weight_decay": 1e-4,
                "batch_size": 16,
                "scored_partitions": ["train"],
                "total_optimizer_updates": sum(a["updates"] for r in runs for a in r["arms"]),
                "total_training_sample_visits": sum(
                    a["sample_visits"] for r in runs for a in r["arms"]
                ),
                "full_train_evaluation_forwards": 24,
                "atol": ATOL,
                "rtol": RTOL,
            },
            "preprocessing": preprocessing,
            "runs": runs,
            "all_checks_passed": all(r["all_checks_passed"] for r in runs),
            "limits": [
                "Fixed nominal lr does not equalize effective displacement.",
                "Only first and second step have matched pre-update parameters and gradients.",
                "Two-epoch train-only comparison, not convergence or generalization evidence.",
                "No heldout scoring, tuning, model selection or universal optimizer ranking.",
                "Observation overhead and GPU/AMP/distributed behavior are not benchmarked.",
            ],
        }
    finally:
        torch.set_num_threads(threads)
