"""Reconstructed, original fixed-state Wine gradient sampling study.

This is a new implementation and run, not recovered bytes or evidence from Nov 3.
"""

import platform

import torch
from torch.nn import functional as F

from pytorch_lab import wine
from pytorch_lab.wine_observations import prepare_train, tensor_digest

BATCH_SIZES = (4, 16, 64)
DRAWS = 32
ATOL = RTOL = 1e-12


def gradient(model, x, y):
    """Return a detached flat mean-CE gradient without changing .grad buffers."""
    loss = F.cross_entropy(model(x), y)
    values = torch.autograd.grad(loss, tuple(model.parameters()))
    result = torch.cat([v.detach().reshape(-1) for v in values])
    if not torch.isfinite(result).all():
        raise ValueError("non-finite gradient")
    return result


def expected_squared_error(per_example, batch_size):
    """Uniform subsets without replacement; population variance denominator N."""
    n = len(per_example)
    if type(batch_size) is not int or not 1 <= batch_size <= n or n < 2:
        raise ValueError("invalid batch size or population")
    if per_example.ndim != 2 or not torch.isfinite(per_example).all():
        raise ValueError("invalid gradient population")
    variance = (per_example - per_example.mean(0)).square().sum(1).mean()
    return float((n - batch_size) / (batch_size * (n - 1)) * variance)


def analyze_model(x, y, architecture, seed):
    wine._features(x)
    wine._labels(y, len(x))
    if x.shape != (107, 13):
        raise ValueError("requires pinned 107 x 13 training partition")
    model = wine.WineClassifier(architecture, seed)
    model.eval()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    full = gradient(model, x, y)
    individual = torch.stack([gradient(model, x[i : i + 1], y[i : i + 1]) for i in range(len(x))])
    batches = [gradient(model, x[i : i + 16], y[i : i + 16]) for i in range(0, len(x), 16)]
    sizes = [min(16, len(x) - i) for i in range(0, len(x), 16)]
    weighted = sum(g * (size / len(x)) for g, size in zip(batches, sizes, strict=True))
    naive = torch.stack(batches).mean(0)
    sample_results = []
    for size in BATCH_SIZES:
        rng = torch.Generator().manual_seed(seed * 1000 + size)
        estimates, draws = [], []
        for _ in range(DRAWS):
            ids = torch.randperm(len(x), generator=rng)[:size]
            g = gradient(model, x[ids], y[ids])
            estimates.append(g)
            draws.append(
                {
                    "indices": ids.tolist(),
                    "gradient_sha256": tensor_digest(g),
                    "squared_error": float((g - full).square().sum()),
                }
            )
        theory = expected_squared_error(individual, size)
        empirical = sum(d["squared_error"] for d in draws) / DRAWS
        sample_results.append(
            {
                "batch_size": size,
                "draws": draws,
                "expected_squared_error": theory,
                "empirical_mean_squared_error": empirical,
                "empirical_to_expected_ratio": empirical / theory if theory else None,
                "finite_sample_mean_error_l2": float(
                    torch.linalg.vector_norm(torch.stack(estimates).mean(0) - full)
                ),
            }
        )
    checks = {
        "per_example_mean_matches_full": bool(
            torch.allclose(individual.mean(0), full, atol=ATOL, rtol=RTOL)
        ),
        "weighted_partition_matches_full": bool(
            torch.allclose(weighted, full, atol=ATOL, rtol=RTOL)
        ),
        "parameters_unchanged": all(torch.equal(before[n], p) for n, p in model.named_parameters()),
        "grad_buffers_unset": all(p.grad is None for p in model.parameters()),
        "tail_preserved": sizes == [16] * 6 + [11],
    }
    return {
        "architecture": architecture,
        "seed": seed,
        "full_gradient": full.tolist(),
        "per_example_gradient_hashes": [tensor_digest(g) for g in individual],
        "partition_sizes": sizes,
        "weighted_max_abs_error": float((weighted - full).abs().max()),
        "naive_partition_error_l2": float(torch.linalg.vector_norm(naive - full)),
        "sampling": sample_results,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def run_experiment():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x, y, preprocessing = prepare_train()
        runs = [
            analyze_model(x, y, architecture, seed)
            for architecture in wine.ARCHITECTURES
            for seed in wine.SEEDS
        ]
        return {
            "schema_version": 1,
            "reconstruction": "New Nov 4 implementation and execution; not recovered Nov 3 bytes",
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
                "dtype": "float64",
                "threads": 1,
            },
            "dataset": {"name": "UCI Wine", "sha256": wine.DATA_SHA256, "license": "CC-BY-4.0"},
            "protocol": {
                "batch_sizes": list(BATCH_SIZES),
                "draws_per_size": DRAWS,
                "gradient_calls": 6 * (1 + 107 + 7 + DRAWS * sum(1 for _ in BATCH_SIZES)),
                "sample_visits": 6 * (107 * 3 + DRAWS * sum(BATCH_SIZES)),
                "optimizer_updates": 0,
                "scored_partitions": ["train"],
                "atol": ATOL,
                "rtol": RTOL,
            },
            "preprocessing": preprocessing,
            "runs": runs,
            "all_checks_passed": all(r["all_checks_passed"] for r in runs),
            "limits": [
                "Finite draws are descriptive; no statistical pass threshold.",
                "A finite sample mean error is not proof of estimator bias.",
                "Fixed parameters and mean CE, no optimizer or decay; no generalization claim.",
                "No heldout scoring here; delivery separately reruns historical Wine evaluation.",
            ],
        }
    finally:
        torch.set_num_threads(old_threads)
