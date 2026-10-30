"""New accumulation tests do not alter historical numerical diagnostics."""

import json
import subprocess
import sys

import pytest
import torch

from pytorch_lab.accumulation import train_accumulated_epoch
from pytorch_lab.accumulation_cli import _compare, _effective_batches, _RecordingSGD, run_experiment
from pytorch_lab.minibatch import AffineRegressor, train_one_epoch


def batches(n=23, size=5, k=2):
    x = torch.arange(n * 3, dtype=torch.float64).reshape(n, 3) / (n * 5)
    y = torch.arange(n * k, dtype=torch.float64).reshape(n, k) / (n * 3)
    return list(zip(x.split(size), y.split(size), strict=True))


@pytest.mark.parametrize(
    "n,size,steps", [(23, 5, 3), (18, 4, 3), (10, 4, 3), (11, 4, 2), (1, 5, 4)]
)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_parameters_and_momentum_match_every_effective_update(n, size, steps, k):
    micro = batches(n, size, k)
    a, b = AffineRegressor(3, k), AffineRegressor(3, k)
    oa, ob = _RecordingSGD(a), _RecordingSGD(b)
    for _ in range(3):
        report = train_accumulated_epoch(a, iter(micro), oa, steps)
        reference = train_one_epoch(b, _effective_batches(micro, steps), ob)
        assert report["optimizer_steps"] == reference["batches"]
        assert report["samples_seen"] == n
        assert report["target_elements"] == n * k
        assert report["online_mse"] == pytest.approx(reference["mse"], rel=1e-12, abs=1e-12)
    assert _compare(oa.trace, ob.trace)["all_close"]
    assert all(p.grad is None for p in a.parameters())


def test_update_boundary_and_partial_final_window():
    model = AffineRegressor(3, 2)
    opt = _RecordingSGD(model)
    report = train_accumulated_epoch(model, batches(), opt, 3)
    assert [(r["samples"], r["target_elements"], r["microbatches"]) for r in report["updates"]] == [
        (15, 30, 3),
        (8, 16, 2),
    ]
    assert len(opt.trace) == 2


def test_accumulation_one_matches_original_per_microbatch_loop():
    a, b = AffineRegressor(3, 2), AffineRegressor(3, 2)
    oa, ob = _RecordingSGD(a), _RecordingSGD(b)
    train_accumulated_epoch(a, batches(), oa, 1)
    train_one_epoch(b, batches(), ob)
    assert _compare(oa.trace, ob.trace)["all_close"]


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.0, "2", None])
def test_bad_accumulation_steps_rejected_without_updates(bad):
    m = AffineRegressor(3, 2)
    opt = _RecordingSGD(m)
    with pytest.raises(ValueError, match="positive integer"):
        train_accumulated_epoch(m, batches(), opt, bad)
    assert not opt.trace


def test_empty_iterable_rejected_and_stale_gradients_cleared():
    m = AffineRegressor(3, 2)
    opt = _RecordingSGD(m)
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    with pytest.raises(ValueError, match="no samples"):
        train_accumulated_epoch(m, iter([]), opt, 3)
    assert not opt.trace
    assert all(p.grad is None for p in m.parameters())


def test_stale_gradients_not_carried_into_next_epoch():
    a, b = AffineRegressor(3, 2), AffineRegressor(3, 2)
    oa, ob = _RecordingSGD(a), _RecordingSGD(b)
    for p in a.parameters():
        p.grad = torch.full_like(p, 999)
    train_accumulated_epoch(a, batches(), oa, 3)
    train_accumulated_epoch(b, batches(), ob, 3)
    assert _compare(oa.trace, ob.trace)["all_close"]


def test_iterator_failure_discards_only_pending_gradients():
    m = AffineRegressor(3, 2)
    opt = _RecordingSGD(m)

    def broken():
        yield from batches()[:4]  # One completed update plus one pending microbatch.
        raise RuntimeError("fixture iterator failed")

    with pytest.raises(RuntimeError, match="fixture iterator"):
        train_accumulated_epoch(m, broken(), opt, 3)
    assert len(opt.trace) == 1
    assert all(p.grad is None for p in m.parameters())
    for name, p in m.named_parameters():
        assert torch.equal(p, opt.trace[0]["parameters"][name])


@pytest.mark.parametrize(
    "kind", ["nan_x", "inf_y", "empty", "broadcast", "float32", "wrong_d", "not_tensor"]
)
def test_invalid_batch_clears_partial_window_without_update(kind):
    m = AffineRegressor(3, 2)
    opt = _RecordingSGD(m)
    x, y = (v.clone() for v in batches()[0])
    if kind == "nan_x":
        x[0, 0] = float("nan")
    elif kind == "inf_y":
        y[0, 0] = float("inf")
    elif kind == "empty":
        x, y = x[:0], y[:0]
    elif kind == "broadcast":
        y = y[:, :1]
    elif kind == "float32":
        y = y.float()
    elif kind == "wrong_d":
        x = x[:, :2]
    else:
        y = [[0]]
    with pytest.raises(ValueError):
        train_accumulated_epoch(m, [batches()[0], (x, y)], opt, 3)
    assert not opt.trace
    assert all(p.grad is None for p in m.parameters())


def test_optimizer_must_match_exact_model_parameters():
    m, other = AffineRegressor(3, 2), AffineRegressor(3, 2)
    with pytest.raises(ValueError, match="own each"):
        train_accumulated_epoch(m, batches(), _RecordingSGD(other), 3)
    with pytest.raises(ValueError, match="torch.optim.SGD"):
        train_accumulated_epoch(m, batches(), torch.optim.Adam(m.parameters()), 3)


def test_cpu_float64_affine_contract():
    m = AffineRegressor(3, 2, dtype=torch.float32)
    with pytest.raises(ValueError, match="CPU float64"):
        train_accumulated_epoch(m, batches(), _RecordingSGD(m), 3)
    generic = torch.nn.Linear(3, 2).double()
    with pytest.raises(ValueError, match="AffineRegressor"):
        train_accumulated_epoch(
            generic, batches(), torch.optim.SGD(generic.parameters(), lr=0.1), 3
        )


def test_loss_overflow_rejected_without_update():
    m = AffineRegressor(3, 2)
    opt = _RecordingSGD(m)
    x = torch.zeros(2, 3, dtype=torch.float64)
    y = torch.full((2, 2), 1e200, dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite"):
        train_accumulated_epoch(m, [(x, y)], opt, 2)
    assert not opt.trace
    assert all(p.grad is None for p in m.parameters())


def test_experiment_records_positive_and_negative_controls():
    r = run_experiment()
    assert r["all_checks_passed"]
    assert len(r["cases"]) == 9
    assert all(r["checks"].values())
    assert r["protocol"]["atol"] == 1e-12
    assert r["protocol"]["rtol"] == 1e-12
    json.dumps(r, allow_nan=False)


def test_cli_matches_stdout_and_does_not_overwrite(tmp_path):
    out = tmp_path / "results.json"
    command = [sys.executable, "-m", "pytorch_lab.accumulation_cli", "--output", str(out)]
    run = subprocess.run(command, check=True, text=True, capture_output=True, timeout=60)
    assert json.loads(run.stdout) == json.loads(out.read_text())
    before = out.read_bytes()
    retry = subprocess.run(command, check=False, text=True, capture_output=True, timeout=60)
    assert retry.returncode != 0
    assert out.read_bytes() == before


def test_thirteenth_installed_cli_keeps_prior_twelve_entries():
    import runpy
    from pathlib import Path

    commands = runpy.run_path(str(Path(__file__).parents[1] / "scripts/check_installed.py"))[
        "COMMANDS"
    ]
    assert [item[0] for item in commands] == [
        "baseline",
        "minibatch",
        "checkpoint",
        "scheduler",
        "wine",
        "runtime",
        "numerical",
        "fixed_input",
        "accumulation",
        "accumulation_recovery",
        "shuffle_stream",
        "joint_recovery",
        "dropout_rng",
    ]
    assert commands[-5] == ("accumulation", "pytorch_lab.accumulation_cli", [])
    assert commands[-4] == ("accumulation_recovery", "pytorch_lab.accumulation_recovery_cli", [])
    assert commands[-3] == ("shuffle_stream", "pytorch_lab.shuffle_stream_cli", [])
    assert commands[-2] == ("joint_recovery", "pytorch_lab.joint_recovery_cli", [])

    assert commands[-1] == ("dropout_rng", "pytorch_lab.dropout_rng_cli", [])
