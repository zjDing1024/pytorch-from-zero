"""Independent reviewer checks; no scheduler used in the manual update oracle."""

import torch
from torch.nn import functional as F

from pytorch_lab import wine
from pytorch_lab import wine_update_schedule as schedule
from pytorch_lab.wine_batch_budget import make_orders, train_run
from pytorch_lab.wine_observations import prepare_train, tensor_digest


def test_independent_full_protocol(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x, y, prep = prepare_train()
        raw_x, raw_y = wine.load_wine()
        held = sorted(set(range(len(raw_x))) - set(prep["row_ids"]))
        raw_x[held] = float("nan")
        raw_y[held] = -100
        counts = dict(updates=0, train_visits=0, evals=0, eval_visits=0)
        old_step = torch.optim.SGD.step
        old_forward = wine.WineClassifier.forward

        def step(self, *args, **kwargs):
            counts["updates"] += 1
            return old_step(self, *args, **kwargs)

        def forward(self, z):
            assert torch.isfinite(z).all()
            if self.training:
                counts["train_visits"] += len(z)
            else:
                counts["eval_visits"] += len(z)
                counts["evals"] += 1
            return old_forward(self, z)

        rng = torch.get_rng_state().clone()
        with monkeypatch.context() as patch:
            patch.setattr(wine, "load_wine", lambda: (raw_x, raw_y))
            patch.setattr(torch.optim.SGD, "step", step)
            patch.setattr(wine.WineClassifier, "forward", forward)
            report = schedule.run_experiment()
        assert report["all_checks_passed"]
        assert torch.equal(rng, torch.get_rng_state())
        assert counts == dict(updates=588, train_visits=12486, evals=60, eval_visits=6420)
        assert report["protocol"]["actual_optimizer_updates"] == counts["updates"]
        assert report["protocol"]["actual_training_sample_visits"] == counts["train_visits"]
        assert report["protocol"]["endpoint_evaluations"] == counts["evals"]
        assert report["protocol"]["evaluation_sample_visits"] == counts["eval_visits"]
        for config in report["configurations"]:
            arch, seed, size = (
                config["architecture"],
                config["seed"],
                config["batch_size"],
            )
            arm = "batch16_14updates" if size == 16 else "batch32_14updates"
            baseline = train_run(x, y, arch, seed, arm, make_orders(seed))[1]
            constant, scheduled = config["runs"]
            assert baseline["final_train_cross_entropy"] == constant["final_train_cross_entropy"]
            assert [t["parameters_after_sha256"] for t in baseline["trace"]] == [
                t["parameters_after_sha256"] for t in constant["trace"]
            ]
            model = wine.WineClassifier(arch, seed)
            buffers = [torch.zeros_like(p) for p in model.parameters()]
            independent_rows = [
                order[offset : offset + size]
                for order in make_orders(seed)
                for offset in range(0, 107, size)
            ][:14]
            for update, (trace, ids) in enumerate(
                zip(scheduled["trace"], independent_rows, strict=True), 1
            ):
                assert trace["row_positions"] == ids
                model.zero_grad(set_to_none=True)
                F.cross_entropy(model(x[ids]), y[ids]).backward()
                learning_rate = 0.05 if update <= 7 else 0.025
                assert trace["lr_used"] == learning_rate
                with torch.no_grad():
                    for parameter, buffer in zip(model.parameters(), buffers, strict=True):
                        gradient = parameter.grad.add(parameter, alpha=1e-4)
                        if update == 1:
                            buffer.copy_(gradient)
                        else:
                            buffer.mul_(0.9).add_(gradient)
                        parameter.add_(buffer, alpha=-learning_rate)
                parameters = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
                momentum = torch.cat([b.reshape(-1) for b in buffers])
                assert tensor_digest(parameters) == trace["parameters_after_sha256"]
                assert tensor_digest(momentum) == trace["momentum_after_sha256"]
    finally:
        torch.set_num_threads(old_threads)
