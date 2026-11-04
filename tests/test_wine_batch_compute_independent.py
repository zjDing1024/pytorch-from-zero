"""Independent actual-call and clock-boundary audit of the prepared-batch timer."""

import statistics

import pytest
import torch

from pytorch_lab import wine
from pytorch_lab import wine_batch_budget as budget
from pytorch_lab import wine_batch_compute as probe


@pytest.mark.parametrize("arm", list(budget.ARMS))
def test_actual_clock_boundary_and_updates(monkeypatch, arm):
    x, y, _ = probe.prepare_train()
    events = []
    state = {"active": False, "steps": 0, "visits": 0, "eval": 0}
    forward = wine.WineClassifier.forward
    step = torch.optim.SGD.step
    digest = probe.tensor_digest

    def wall():
        events.append("wall")
        state["active"] = not state["active"]
        return 100 if state["active"] else 1100

    def cpu():
        assert state["active"]
        events.append("cpu")
        return 200 if events.count("cpu") == 1 else 900

    def wrapped_forward(self, xb):
        if torch.is_grad_enabled():
            assert state["active"]
            state["visits"] += len(xb)
        else:
            assert not state["active"]
            state["eval"] += 1
        return forward(self, xb)

    def wrapped_step(self, *args, **kwargs):
        assert state["active"]
        state["steps"] += 1
        return step(self, *args, **kwargs)

    def wrapped_digest(value):
        assert not state["active"]
        return digest(value)

    monkeypatch.setattr(probe.time, "perf_counter_ns", wall)
    monkeypatch.setattr(probe.time, "process_time_ns", cpu)
    monkeypatch.setattr(wine.WineClassifier, "forward", wrapped_forward)
    monkeypatch.setattr(torch.optim.SGD, "step", wrapped_step)
    monkeypatch.setattr(probe, "tensor_digest", wrapped_digest)
    _, row = probe.train_timed(x, y, "linear", 42, arm, budget.make_orders(42))
    assert events == ["wall", "cpu", "cpu", "wall"]
    assert row["wall_ns"] == 1000 and row["process_cpu_ns"] == 700
    assert state["steps"] == budget.ARMS[arm][1] == row["updates"]
    assert state["visits"] == budget.ARMS[arm][2] == row["sample_visits"]
    assert state["eval"] == 1 and not state["active"]


def test_actual_full_budget_isolation_and_summaries(monkeypatch):
    original_load = wine.load_wine
    x, y, _ = probe.prepare_train()
    captured = {"steps": 0, "visits": 0, "eval": 0}
    forward, step = wine.WineClassifier.forward, torch.optim.SGD.step

    def poisoned_load():
        all_x, all_y = original_load()
        import json
        from importlib.resources import files

        ids = json.loads(files("pytorch_lab").joinpath("data/wine_split.json").read_text())[
            "row_ids"
        ]["train"]
        heldout = [i for i in range(len(all_y)) if i not in ids]
        all_x[heldout] = float("nan")
        all_y[heldout] = -100
        return all_x, all_y

    def checked_forward(self, xb):
        assert all(any(torch.equal(row, train_row) for train_row in x) for row in xb)
        if torch.is_grad_enabled():
            captured["visits"] += len(xb)
        else:
            captured["eval"] += 1
        return forward(self, xb)

    def checked_step(self, *args, **kwargs):
        captured["steps"] += 1
        return step(self, *args, **kwargs)

    monkeypatch.setattr(wine, "load_wine", poisoned_load)
    monkeypatch.setattr(wine.WineClassifier, "forward", checked_forward)
    monkeypatch.setattr(torch.optim.SGD, "step", checked_step)
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    report = probe.run_experiment()
    assert captured == {"steps": 1080, "visits": 24390, "eval": 90}
    assert torch.equal(rng, torch.get_rng_state())
    assert threads == torch.get_num_threads()
    assert report["all_checks_passed"]
    for config in report["configurations"]:
        rows = config["observations"]
        assert len(rows) == 12
        for summary in config["summaries"]:
            measured = [r for r in rows if r["phase"] == "measured" and r["arm"] == summary["arm"]]
            assert sorted(r["position"] for r in measured) == [0, 1, 2]
            assert summary["wall_ns_median"] == statistics.median(r["wall_ns"] for r in measured)
            assert summary["process_cpu_ns_median"] == statistics.median(
                r["process_cpu_ns"] for r in measured
            )
            assert (
                len({r["final_parameters_sha256"] for r in rows if r["arm"] == summary["arm"]}) == 1
            )
