"""Independent analytic oracle for the bounded accumulation training contract.

This original verifier differentiates the affine all-element MSE by hand and
applies SGD equations without using the training loop's autograd reference.
It verifies a fixed CPU float64 grid; it is not a proof for arbitrary optimizers,
stochastic layers, devices, data magnitudes, or learning-rate schedules.
Run after installing the project: python scripts/verify_accumulation_analytic.py
JSON is written to stdout only; redirect it to a NEW evidence file.
"""

import itertools
import json
import math

import torch

from pytorch_lab.accumulation import train_accumulated_epoch
from pytorch_lab.minibatch import AffineRegressor


def main() -> None:
    torch.set_num_threads(1)
    checked = 0
    max_gap = 0.0
    for n, k, steps, variant in itertools.product(
        (1, 10, 18, 23), (1, 3), (1, 3, 6), ("plain", "momentum", "nesterov", "maximize")
    ):
        d = 3
        gen = torch.Generator().manual_seed(901 + n + k)
        x = torch.randn(n, d, generator=gen, dtype=torch.float64) / 4
        y = torch.randn(n, k, generator=gen, dtype=torch.float64) / 5
        model = AffineRegressor(d, k)
        with torch.no_grad():
            model.weight.copy_(torch.randn(d, k, generator=gen, dtype=torch.float64) / 8)
            model.bias.copy_(torch.randn(k, generator=gen, dtype=torch.float64) / 8)
        weight = model.weight.detach().clone()
        bias = model.bias.detach().clone()
        bufw = bufb = None
        kw = {
            "lr": 0.02,
            "weight_decay": 0.03,
            "momentum": 0.0 if variant == "plain" else 0.7,
            "dampening": 0.2 if variant == "momentum" else 0.0,
            "nesterov": variant == "nesterov",
            "maximize": variant == "maximize",
        }
        optimizer = torch.optim.SGD(model.parameters(), **kw)
        parts = list(zip(x.split(4), y.split(4), strict=True))
        expected_steps = math.ceil(len(parts) / steps)
        for epoch in range(4):
            for start in range(0, len(parts), steps):
                group = parts[start : start + steps]
                xx = torch.cat([p[0] for p in group])
                yy = torch.cat([p[1] for p in group])
                residual = xx @ weight + bias - yy
                gw = 2 * (xx.T @ residual) / yy.numel()
                gb = 2 * residual.sum(0) / yy.numel()
                if kw["maximize"]:
                    gw = -gw
                    gb = -gb
                gw = gw + kw["weight_decay"] * weight
                gb = gb + kw["weight_decay"] * bias
                if kw["momentum"]:
                    bufw = (
                        gw.clone()
                        if bufw is None
                        else kw["momentum"] * bufw + (1 - kw["dampening"]) * gw
                    )
                    bufb = (
                        gb.clone()
                        if bufb is None
                        else kw["momentum"] * bufb + (1 - kw["dampening"]) * gb
                    )
                    if kw["nesterov"]:
                        gw = gw + kw["momentum"] * bufw
                        gb = gb + kw["momentum"] * bufb
                    else:
                        gw = bufw
                        gb = bufb
                weight = weight - kw["lr"] * gw
                bias = bias - kw["lr"] * gb
            report = train_accumulated_epoch(model, iter(parts), optimizer, steps)
            assert report["optimizer_steps"] == expected_steps
            assert report["samples_seen"] == n and report["target_elements"] == n * k
            for actual, want in ((model.weight, weight), (model.bias, bias)):
                gap = (actual - want).abs().max().item()
                max_gap = max(gap, max_gap)
                assert torch.allclose(actual, want, rtol=1e-12, atol=1e-12), (
                    n,
                    k,
                    steps,
                    variant,
                    epoch,
                    gap,
                )
            if kw["momentum"]:
                for p, want in ((model.weight, bufw), (model.bias, bufb)):
                    actual = optimizer.state[p]["momentum_buffer"]
                    gap = (actual - want).abs().max().item()
                    max_gap = max(gap, max_gap)
                    assert torch.allclose(actual, want, rtol=1e-12, atol=1e-12)
            assert all(p.grad is None for p in model.parameters())
        checked += 1

    # Plain iterable short failure after a completed window leaves exactly that update,
    # with no parameter change from the partial window and no residual gradient.
    m = AffineRegressor(3, 3)
    o = torch.optim.SGD(m.parameters(), lr=0.04, momentum=0.8)
    xs = torch.ones(1, 3, dtype=torch.float64)
    ys = torch.full((1, 3), 2.0, dtype=torch.float64)
    seen = {}

    def broken():
        yield xs, ys
        yield xs * 2, ys
        seen["parameters"] = [p.detach().clone() for p in m.parameters()]
        yield xs * 3, ys
        raise RuntimeError("independent injected iterator error")

    try:
        train_accumulated_epoch(m, broken(), o, 2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("failure swallowed")
    assert all(p.grad is None for p in m.parameters())
    assert all(torch.equal(p, a) for p, a in zip(m.parameters(), seen["parameters"], strict=True))

    print(
        json.dumps(
            {
                "analytical_oracle_configurations": checked,
                "epochs_per_configuration": 4,
                "max_parameter_or_momentum_absolute_gap": max_gap,
                "rtol": 1e-12,
                "atol": 1e-12,
                "failure_cleanup_passed": True,
                "scope": (
                    "CPU float64 affine; analytic gradients and manually computed SGD "
                    "including weight_decay/dampening/nesterov/maximize; "
                    "not a general optimizer proof"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
