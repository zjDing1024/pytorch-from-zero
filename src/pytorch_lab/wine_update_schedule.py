"""Original Wine experiment: optimizer-update clocks and bounded state restoration.

The snapshots are trusted internal experiment objects, not a public file format.
Only complete successful updates are restoration boundaries.
"""

import copy
import math
import platform

import torch
from torch.nn import functional as F

from pytorch_lab import wine
from pytorch_lab.wine_batch_budget import make_orders
from pytorch_lab.wine_observations import prepare_train, tensor_digest

CUTS = (6, 7, 8)
TARGET = 14


def _flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def equal_state(left, right):
    """Exact recursive comparison including optimizer tensors, types and keys."""
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            equal_state(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


class UpdateRun:
    """Pinned CPU experiment runner, not an arbitrary-model checkpoint service."""

    def __init__(self, x, y, architecture, seed, batch_size, arm):
        wine._features(x)
        wine._labels(y, len(x))
        if x.shape != (107, 13) or type(batch_size) is not int or batch_size not in (16, 32):
            raise ValueError("requires pinned training shape and batch size 16 or 32")
        if arm not in ("constant", "step7"):
            raise ValueError("unknown schedule arm")
        self.x, self.y = x, y
        self.config = (architecture, seed, batch_size, arm)
        self.batches = [
            order[offset : offset + batch_size]
            for order in make_orders(seed)
            for offset in range(0, 107, batch_size)
        ][:TARGET]
        self.model = wine.WineClassifier(architecture, seed)
        self.model.train()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4
        )
        self.scheduler = (
            torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.5)
            if arm == "step7"
            else None
        )
        self.trace = []
        self.valid = True

    def snapshot(self):
        if not self.valid:
            raise RuntimeError("failed update: restore a previous complete snapshot")
        return copy.deepcopy(
            {
                "config": self.config,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
                "trace": self.trace,
                "batches": self.batches,
            }
        )

    @classmethod
    def restore(cls, x, y, snapshot):
        """Restore a trusted snapshot produced by this experiment on the same input."""
        state = copy.deepcopy(snapshot)
        run = cls(x, y, *state["config"])
        if state["batches"] != run.batches:
            raise ValueError("snapshot batch sequence differs")
        # Construct scheduler first: its initialization may overwrite optimizer LR.
        run.model.load_state_dict(state["model"])
        if run.scheduler is not None:
            run.scheduler.load_state_dict(state["scheduler"])
        run.optimizer.load_state_dict(state["optimizer"])
        run.trace = state["trace"]
        if not 0 <= len(run.trace) <= TARGET:
            raise ValueError("snapshot cursor outside experiment")
        return run

    def step(self):
        if not self.valid:
            raise RuntimeError("failed update: restore a previous complete snapshot")
        if len(self.trace) == TARGET:
            raise StopIteration
        self.valid = False  # A partial operation cannot be saved or continued.
        ids = self.batches[len(self.trace)]
        self.optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self.model(self.x[ids]), self.y[ids])
        if not torch.isfinite(loss):
            raise ValueError("non-finite loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError("missing or non-finite gradient")
        before = _flat(self.model).clone()
        lr_used = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        after = _flat(self.model)
        if not torch.isfinite(after).all():
            raise ValueError("non-finite parameters")
        momentum = torch.cat(
            [
                self.optimizer.state[p]["momentum_buffer"].reshape(-1)
                for p in self.model.parameters()
            ]
        )
        self.trace.append(
            {
                "update": len(self.trace) + 1,
                "row_positions": ids,
                "lr_used": lr_used,
                "lr_next": self.optimizer.param_groups[0]["lr"],
                "scheduler_last_epoch": None
                if self.scheduler is None
                else self.scheduler.last_epoch,
                "loss_before_update": loss.item(),
                "actual_update_l2": float(torch.linalg.vector_norm(after - before)),
                "parameters_after_sha256": tensor_digest(after),
                "momentum_after_sha256": tensor_digest(momentum),
            }
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.valid = True

    def report(self):
        if not self.valid or len(self.trace) != TARGET:
            raise RuntimeError("report requires completed experiment")
        self.model.eval()
        try:
            with torch.no_grad():
                ce = F.cross_entropy(self.model(self.x), self.y).item()
        finally:
            self.model.train()
        if not math.isfinite(ce):
            raise ValueError("non-finite endpoint loss")
        return {
            "arm": self.config[-1],
            "updates": len(self.trace),
            "sample_visits": sum(len(t["row_positions"]) for t in self.trace),
            "final_train_cross_entropy": ce,
            "trace": copy.deepcopy(self.trace),
        }


def run_experiment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x, y, preprocessing = prepare_train()
        configs = []
        work_updates = work_visits = 0
        for architecture in wine.ARCHITECTURES:
            for seed in wine.SEEDS:
                for batch_size in (16, 32):
                    reports, restored = [], []
                    states = {}
                    initials = []
                    for arm in ("constant", "step7"):
                        run = UpdateRun(x, y, architecture, seed, batch_size, arm)
                        initials.append(tensor_digest(_flat(run.model)))
                        for update in range(1, TARGET + 1):
                            run.step()
                            work_updates += 1
                            work_visits += len(run.trace[-1]["row_positions"])
                            if arm == "step7":
                                states[update] = run.snapshot()
                        reports.append(run.report())
                    for cut in CUTS:
                        resumed = UpdateRun.restore(x, y, states[cut])
                        boundaries = [equal_state(resumed.snapshot(), states[cut])]
                        for update in range(cut + 1, TARGET + 1):
                            resumed.step()
                            work_updates += 1
                            work_visits += len(resumed.trace[-1]["row_positions"])
                            boundaries.append(equal_state(resumed.snapshot(), states[update]))
                        restored.append(
                            {
                                "cut_after_update": cut,
                                "boundary_states_equal": boundaries,
                                "complete_report_equal": resumed.report() == reports[1],
                            }
                        )
                    fixed, scheduled = reports
                    checks = {
                        "identical_initial_parameters": initials[0] == initials[1],
                        "identical_batch_rows": [t["row_positions"] for t in fixed["trace"]]
                        == [t["row_positions"] for t in scheduled["trace"]],
                        "first_seven_parameter_states_equal": all(
                            a["parameters_after_sha256"] == b["parameters_after_sha256"]
                            for a, b in zip(fixed["trace"][:7], scheduled["trace"][:7], strict=True)
                        ),
                        "exact_lr_clock": all(
                            t["lr_used"] == 0.05 * 0.5 ** ((t["update"] - 1) // 7)
                            and t["lr_next"] == 0.05 * 0.5 ** (t["update"] // 7)
                            and t["scheduler_last_epoch"] == t["update"]
                            for t in scheduled["trace"]
                        ),
                        "constant_lr": all(t["lr_used"] == 0.05 for t in fixed["trace"]),
                        "budgets": all(
                            r["updates"] == 14
                            and r["sample_visits"] == (214 if batch_size == 16 else 385)
                            for r in reports
                        ),
                        "restored_complete_state": all(
                            all(r["boundary_states_equal"]) and r["complete_report_equal"]
                            for r in restored
                        ),
                    }
                    configs.append(
                        {
                            "architecture": architecture,
                            "seed": seed,
                            "batch_size": batch_size,
                            "runs": reports,
                            "restoration": restored,
                            "checks": checks,
                        }
                    )
        return {
            "schema_version": 1,
            "environment": {"python": platform.python_version(), "torch": torch.__version__},
            "preprocessing": preprocessing,
            "protocol": {
                "device": "cpu",
                "dtype": "float64",
                "threads": 1,
                "scored_partitions": ["train"],
                "comparison_runs": 24,
                "restored_suffix_runs": 36,
                "actual_optimizer_updates": work_updates,
                "actual_training_sample_visits": work_visits,
                "endpoint_evaluations": 60,
                "evaluation_sample_visits": 6420,
                "restoration": "trusted in-memory snapshots into fresh objects, same process",
            },
            "configurations": configs,
            "all_checks_passed": all(all(c["checks"].values()) for c in configs),
            "limits": [
                "No heldout scoring or schedule selection.",
                "Equal updates do not equal sample visits or compute.",
                "No file durability, new-process, arbitrary-module, AMP or distributed recovery.",
                "Assistant-run evidence does not establish learner mastery.",
            ],
        }
    finally:
        torch.set_num_threads(threads)
