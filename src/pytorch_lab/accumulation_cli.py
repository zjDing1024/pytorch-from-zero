"""Auditable accumulation/effective-batch comparison; original implementation."""

import argparse
import json
import math
import platform
from pathlib import Path

import torch

from pytorch_lab.accumulation import train_accumulated_epoch
from pytorch_lab.minibatch import AffineRegressor, RegressionDataset, make_loader, train_one_epoch

# Declared before experiments: bounded CPU float64 affine fixture, not a universal
# floating-point bound, GPU guarantee, or replacement for existing ULP diagnostics.
ATOL = 1e-12
RTOL = 1e-12


class _RecordingSGD(torch.optim.SGD):
    def __init__(self, model: AffineRegressor) -> None:
        super().__init__(model.parameters(), lr=0.03, momentum=0.8, weight_decay=0.01)
        self.model = model
        self.trace = []

    def step(self, closure=None):
        result = super().step(closure)
        self.trace.append(
            {
                "parameters": {
                    name: p.detach().clone() for name, p in self.model.named_parameters()
                },
                "momentum": {
                    name: self.state[p]["momentum_buffer"].detach().clone()
                    for name, p in self.model.named_parameters()
                },
            }
        )
        return result


def _fixture(n: int, seed: int) -> RegressionDataset:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=gen, dtype=torch.float64) / 3
    w = torch.tensor([[0.5, -0.25], [-0.75, 0.125], [0.25, 0.75]], dtype=torch.float64)
    # Heterogeneous sample residuals make unequal microbatch weighting observable.
    y = x @ w + torch.linspace(-0.2, 0.4, n, dtype=torch.float64).unsqueeze(1)
    return RegressionDataset(x, y)


def _effective_batches(microbatches: list, steps: int) -> list:
    return [
        (torch.cat([x for x, _ in group]), torch.cat([y for _, y in group]))
        for start in range(0, len(microbatches), steps)
        if (group := microbatches[start : start + steps])
    ]


def _naive_epoch(model, batches, optimizer, steps, fixed_divisor):
    """Deliberately WRONG controls only; never used by the actual training API."""
    for start in range(0, len(batches), steps):
        group = batches[start : start + steps]
        optimizer.zero_grad(set_to_none=True)
        divisor = steps if fixed_divisor else len(group)
        for x, y in group:
            ((model(x) - y).square().mean() / divisor).backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _compare(left, right):
    if len(left) != len(right):
        return {"all_close": False, "max_absolute_gap": None, "updates": []}
    gaps = []
    okay = True
    for a, b in zip(left, right, strict=True):
        row = {}
        for field in ("parameters", "momentum"):
            row[field] = max((a[field][key] - b[field][key]).abs().max().item() for key in a[field])
            okay &= all(
                torch.allclose(a[field][key], b[field][key], atol=ATOL, rtol=RTOL)
                for key in a[field]
            )
        gaps.append(row)
    return {
        "all_close": bool(okay),
        "max_absolute_gap": max(max(row.values()) for row in gaps),
        "updates": gaps,
    }


def run_experiment() -> dict:
    """Nine bounded 3-epoch comparisons; all input order/config choices fixed."""
    cases = []
    for seed in (42, 7, 123):
        for n, micro_size, steps in ((23, 5, 3), (11, 4, 2), (10, 4, 3)):
            dataset = _fixture(n, seed)
            micro = list(make_loader(dataset, micro_size, seed=seed, shuffle=False))
            effective = _effective_batches(micro, steps)
            models = {
                name: AffineRegressor(3, 2) for name in ("weighted", "reference", "naive", "fixed")
            }
            optimizers = {name: _RecordingSGD(model) for name, model in models.items()}
            reports = []
            for _ in range(3):
                reports.append(
                    train_accumulated_epoch(
                        models["weighted"], micro, optimizers["weighted"], steps
                    )
                )
                train_one_epoch(models["reference"], effective, optimizers["reference"])
                _naive_epoch(models["naive"], micro, optimizers["naive"], steps, False)
                _naive_epoch(models["fixed"], micro, optimizers["fixed"], steps, True)
            cases.append(
                {
                    "seed": seed,
                    "samples": n,
                    "output_width": 2,
                    "microbatch_size": micro_size,
                    "accumulation_steps": steps,
                    "microbatch_sizes": [len(x) for x, _ in micro],
                    "effective_batch_sizes": [len(x) for x, _ in effective],
                    "epochs": 3,
                    "weighted_epoch_reports": reports,
                    "weighted_vs_effective": _compare(
                        optimizers["weighted"].trace, optimizers["reference"].trace
                    ),
                    "naive_mean_of_means_vs_effective": _compare(
                        optimizers["naive"].trace, optimizers["reference"].trace
                    ),
                    "naive_fixed_divisor_vs_effective": _compare(
                        optimizers["fixed"].trace, optimizers["reference"].trace
                    ),
                }
            )
    checks = {
        "every_weighted_parameter_and_momentum_update_matches": all(
            c["weighted_vs_effective"]["all_close"] for c in cases
        ),
        "all_samples_seen_each_epoch": all(
            r["samples_seen"] == c["samples"] for c in cases for r in c["weighted_epoch_reports"]
        ),
        "optimizer_step_count_matches_effective_windows": all(
            r["optimizer_steps"] == math.ceil(len(c["microbatch_sizes"]) / c["accumulation_steps"])
            for c in cases
            for r in c["weighted_epoch_reports"]
        ),
        "naive_unequal_microbatch_control_detected": all(
            not c["naive_mean_of_means_vs_effective"]["all_close"]
            for c in cases
            if c["samples"] in (23, 10)
        ),
        "naive_fixed_divisor_short_window_control_detected": all(
            not c["naive_fixed_divisor_vs_effective"]["all_close"]
            for c in cases
            if c["samples"] == 11
        ),
    }
    return {
        "schema_version": 1,
        "experiment": "sample_weighted_gradient_accumulation",
        "executed_by": "AI assistant; learner mastery is unassessed",
        "environment": {"python": platform.python_version(), "pytorch": torch.__version__},
        "protocol": {
            "device": "cpu",
            "dtype": "float64",
            "loss": "sum squared error per microbatch; divide gradients by actual window N*K",
            "comparison": "same ordered samples, effective windows, initialization, SGD updates",
            "optimizer": {"lr": 0.03, "momentum": 0.8, "weight_decay": 0.01},
            "atol": ATOL,
            "rtol": RTOL,
            "tolerance_scope": "predeclared bounded affine fixture, not a universal error bound",
            "excluded": ["BatchNorm", "Dropout", "AMP", "DDP", "GPU", "memory/speed claims"],
        },
        "cases": cases,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    # Never overwrite a prior evidence file.
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    result = run_experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    print(encoded, end="")
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
