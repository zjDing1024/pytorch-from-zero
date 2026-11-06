"""Independent leakage, selection, and measured-work review for frozen Wine curves."""

import copy
import json
from importlib.resources import files

import pytest
import torch
from torch.nn import functional as F

from pytorch_lab import wine
from pytorch_lab import wine_learning_curves as curves


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def grid(value=1.0):
    return [
        {
            "architecture": architecture,
            "seed": seed,
            "arm": arm,
            "curve": [
                {"epoch": epoch, "train_ce": value, "validation_ce": value}
                for epoch in range(1, 21)
            ],
        }
        for architecture in ("linear", "mlp16")
        for seed in (42, 7, 123)
        for arm in ("constant", "step7")
    ]


def test_selection_uses_mean_not_luckiest_seed_or_training_loss():
    runs = grid(4.0)
    for run in runs:
        run["curve"][19]["train_ce"] = 0.0
        if run["architecture"] == "linear" and run["arm"] == "constant":
            run["curve"][1]["validation_ce"] = {42: 0.0, 7: 3.0, 123: 3.0}[run["seed"]]
        if run["architecture"] == "mlp16" and run["arm"] == "step7":
            run["curve"][4]["validation_ce"] = 1.5
    result = curves.select_endpoint(runs)
    assert len(result["all_candidates"]) == 16
    selected = result["selected"]
    assert (selected["architecture"], selected["arm"], selected["epoch"]) == ("mlp16", "step7", 5)
    assert selected["mean_validation_ce"] == 1.5
    assert curves.select_endpoint(list(reversed(runs))) == result


def test_exact_tie_priority_and_non_candidates_are_excluded():
    runs = grid()
    for run in runs:
        run["curve"][0]["validation_ce"] = 0.0
    chosen = curves.select_endpoint(runs)["selected"]
    assert (chosen["epoch"], chosen["architecture"], chosen["arm"]) == (2, "linear", "constant")
    for run in runs:
        if run["architecture"] == "mlp16" and run["arm"] == "step7":
            run["curve"][1]["validation_ce"] = 0.5
        if run["architecture"] == "linear" and run["arm"] == "constant":
            run["curve"][4]["validation_ce"] = 0.5
    assert curves.select_endpoint(runs)["selected"]["epoch"] == 2


@pytest.mark.parametrize("defect", ["missing", "duplicate", "curve", "nan", "inf"])
def test_failed_or_incomplete_runs_cannot_be_silently_dropped(defect):
    runs = grid()
    if defect == "missing":
        runs.pop()
    elif defect == "duplicate":
        runs[-1] = copy.deepcopy(runs[0])
    elif defect == "curve":
        runs[0]["curve"].pop(0)
    else:
        runs[0]["curve"][2]["validation_ce"] = float(defect)
    with pytest.raises(ValueError):
        curves.select_endpoint(runs)


def test_test_rows_and_validation_values_do_not_fit_scaler(monkeypatch):
    original_x, original_y = wine.load_wine()
    baseline = curves.prepare_data()
    manifest = json.loads(files("pytorch_lab").joinpath("data/wine_split.json").read_text())
    modified_x, modified_y = original_x.clone(), original_y.clone()
    heldout = manifest["row_ids"]["validation"] + manifest["row_ids"]["test"]
    modified_x[heldout] = modified_x[heldout] * -1000 + 50000
    modified_y[heldout] = (modified_y[heldout] + 1) % 3
    monkeypatch.setattr(wine, "load_wine", lambda: (modified_x, modified_y))
    changed = curves.prepare_data()
    assert torch.equal(baseline[0], changed[0])
    assert torch.equal(baseline[1], changed[1])
    assert baseline[4] == changed[4]
    assert not torch.equal(baseline[2], changed[2])
    modified_x[manifest["row_ids"]["test"]] = float("nan")
    modified_y[manifest["row_ids"]["test"]] = -999
    test_poisoned = curves.prepare_data()
    assert all(torch.equal(a, b) for a, b in zip(changed[:4], test_poisoned[:4], strict=True))


@pytest.mark.parametrize("mode", [False, True])
def test_evaluation_preserves_mode_parameters_gradients_and_rng(mode):
    x, y, _, _, _ = curves.prepare_data()
    model = wine.WineClassifier("mlp16", 42)
    model.train(mode)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    parameters = [p.detach().clone() for p in model.parameters()]
    gradients = [p.grad.clone() for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    actual = curves.evaluate(model, x, y)
    assert model.training == mode
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(p, v) for p, v in zip(model.parameters(), parameters, strict=True))
    assert all(torch.equal(p.grad, v) for p, v in zip(model.parameters(), gradients, strict=True))
    with torch.no_grad():
        assert actual == F.cross_entropy(model(x), y).item()


def test_validation_perturbation_cannot_change_training_trajectory():
    x, y, vx, vy, _ = curves.prepare_data()
    first = curves.train_run(x, y, vx, vy, "linear", 42, "constant")
    second = curves.train_run(x, y, vx * -2, (vy + 1) % 3, "linear", 42, "constant")
    assert first["order_sha256"] == second["order_sha256"]
    for a, b in zip(first["curve"], second["curve"], strict=True):
        assert a["parameters_sha256"] == b["parameters_sha256"]
        assert a["train_ce"] == b["train_ce"]
    assert first["curve"][-1]["validation_ce"] != second["curve"][-1]["validation_ce"]


@pytest.mark.parametrize("arm", ["constant", "step7"])
def test_uninstrumented_training_oracle_and_tail_budget(arm):
    x, y, vx, vy, _ = curves.prepare_data()
    observed = curves.train_run(x, y, vx, vy, "linear", 7, arm)
    model = wine.WineClassifier("linear", 7)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(7)
    # Independent arithmetic LR oracle: no scheduler or experiment evaluation calls.
    for epoch in range(1, 21):
        order = torch.randperm(107, generator=generator)
        for batch, ids in enumerate(order.split(16)):
            update = (epoch - 1) * 7 + batch
            optimizer.param_groups[0]["lr"] = 0.05 * (0.5 ** (update // 7) if arm == "step7" else 1)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[ids]), y[ids]).backward()
            optimizer.step()
        point = observed["curve"][epoch - 1]
        assert curves.digest_model(model) == point["parameters_sha256"]
        assert point["updates"] == epoch * 7
        assert point["sample_visits"] == epoch * 107
        assert point["lr_last_used"] == 0.05 * (0.5 ** (epoch - 1) if arm == "step7" else 1)
        assert point["lr_next"] == 0.05 * (0.5**epoch if arm == "step7" else 1)
    assert observed["endpoint_reload_checks"] == [True] * 4


def test_work_totals_are_actual_calls_and_sample_visits(monkeypatch):
    real_step, real_evaluate = torch.optim.SGD.step, curves.evaluate
    observed = {"updates": 0, "evaluation_calls": 0, "evaluation_visits": 0}

    def step(self, *args, **kwargs):
        observed["updates"] += 1
        return real_step(self, *args, **kwargs)

    def evaluate(model, x, y):
        observed["evaluation_calls"] += 1
        observed["evaluation_visits"] += len(y)
        return real_evaluate(model, x, y)

    monkeypatch.setattr(torch.optim.SGD, "step", step)
    monkeypatch.setattr(curves, "evaluate", evaluate)
    result = curves.run_experiment()
    work = result["work"]
    assert observed["updates"] == work["optimizer_updates"] == 1680
    assert sum(r["curve"][-1]["sample_visits"] for r in result["runs"]) == 25680
    assert work["training_sample_visits"] == 25680
    assert (
        observed["evaluation_calls"]
        == (work["curve_evaluation_calls"] + work["reload_evaluation_calls"])
        == 528
    )
    assert (
        observed["evaluation_visits"]
        == (work["curve_evaluation_sample_visits"] + work["reload_evaluation_sample_visits"])
        == 36048
    )
