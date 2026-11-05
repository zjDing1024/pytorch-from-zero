"""Update-clock contract, state ownership and failure boundaries."""

import copy

import pytest
import torch

from pytorch_lab.wine_observations import prepare_train
from pytorch_lab.wine_update_schedule import UpdateRun, equal_state, run_experiment


@pytest.fixture(scope="module")
def data():
    return prepare_train()[:2]


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("architecture", ["linear", "mlp16"])
def test_update_clock_and_restoration(data, batch_size, architecture):
    run = UpdateRun(*data, architecture, 42, batch_size, "step7")
    snapshots = {0: run.snapshot()}
    for update in range(1, 15):
        run.step()
        snapshots[update] = run.snapshot()
        assert run.trace[-1]["lr_used"] == 0.05 * 0.5 ** ((update - 1) // 7)
        assert run.trace[-1]["lr_next"] == 0.05 * 0.5 ** (update // 7)
    for cut in (0, 6, 7, 8, 14):
        restored = UpdateRun.restore(*data, snapshots[cut])
        assert equal_state(restored.snapshot(), snapshots[cut])
        for update in range(cut + 1, 15):
            restored.step()
            assert equal_state(restored.snapshot(), snapshots[update])
        assert restored.report() == run.report()
    with pytest.raises(StopIteration):
        run.step()


@pytest.mark.parametrize("batch_size", [True, 0, 15, 32.0])
def test_invalid_batch(data, batch_size):
    with pytest.raises(ValueError):
        UpdateRun(*data, "linear", 42, batch_size, "step7")


def test_snapshot_owns_data(data):
    run = UpdateRun(*data, "linear", 42, 16, "step7")
    run.step()
    snapshot = run.snapshot()
    untouched = copy.deepcopy(snapshot)
    restored = UpdateRun.restore(*data, snapshot)
    restored.step()
    run.step()
    assert equal_state(snapshot, untouched)
    snapshot["trace"][0]["lr_used"] = -1
    assert run.trace[0]["lr_used"] == restored.trace[0]["lr_used"] == 0.05


def test_incomplete_or_failed_updates_refuse_state(data, monkeypatch):
    run = UpdateRun(*data, "linear", 42, 16, "step7")
    with pytest.raises(RuntimeError, match="completed"):
        run.report()
    saved = run.snapshot()

    def broken():
        raise RuntimeError("injected scheduler failure")

    monkeypatch.setattr(run.scheduler, "step", broken)
    with pytest.raises(RuntimeError, match="injected"):
        run.step()
    for operation in (run.step, run.snapshot, run.report):
        with pytest.raises(RuntimeError):
            operation()
    restored = UpdateRun.restore(*data, saved)
    restored.step()
    assert restored.trace[0]["lr_used"] == 0.05


def test_resetting_scheduler_has_phase_sensitive_effect(data):
    # At cut 6 the next LR is correct but losing count delays the pending decay.
    run = UpdateRun(*data, "linear", 42, 16, "step7")
    for _ in range(6):
        run.step()
    restored = UpdateRun.restore(*data, run.snapshot())
    fresh = UpdateRun(*data, "linear", 42, 16, "step7")
    restored.scheduler.load_state_dict(fresh.scheduler.state_dict())
    run.step()
    restored.step()
    assert run.trace[-1]["lr_used"] == restored.trace[-1]["lr_used"]
    assert run.trace[-1]["parameters_after_sha256"] == restored.trace[-1]["parameters_after_sha256"]
    assert run.trace[-1]["lr_next"] != restored.trace[-1]["lr_next"]
    run.step()
    restored.step()
    assert run.trace[-1]["parameters_after_sha256"] != restored.trace[-1]["parameters_after_sha256"]


def test_complete_experiment_and_rng(data):
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    report = run_experiment()
    assert report["all_checks_passed"] is True
    assert len(report["configurations"]) == 12
    assert report["protocol"]["actual_optimizer_updates"] == 588
    assert report["protocol"]["endpoint_evaluations"] == 60
    assert torch.equal(rng, torch.get_rng_state())
    assert threads == torch.get_num_threads()


@pytest.mark.parametrize("left,right", [(True, 1), ([1], (1,)), ({"x": 1}, {"y": 1})])
def test_exact_state_types(left, right):
    assert not equal_state(left, right)


@pytest.mark.parametrize("fault", ["momentum", "extra_scheduler_step"])
def test_faulted_restore_changes_next_update(data, fault):
    run = UpdateRun(*data, "mlp16", 7, 32, "step7")
    for _ in range(6):
        run.step()
    restored = UpdateRun.restore(*data, run.snapshot())
    if fault == "momentum":
        restored.optimizer.state.clear()
    else:
        restored.scheduler.step()
    run.step()
    restored.step()
    assert run.trace[-1]["parameters_after_sha256"] != restored.trace[-1]["parameters_after_sha256"]


def test_period_boundary_reset_can_preserve_remaining_parameter_path(data):
    run = UpdateRun(*data, "linear", 42, 16, "step7")
    for _ in range(7):
        run.step()
    restored = UpdateRun.restore(*data, run.snapshot())
    fresh = UpdateRun(*data, "linear", 42, 16, "step7")
    restored.scheduler.load_state_dict(fresh.scheduler.state_dict())
    for _ in range(7):
        run.step()
        restored.step()
        assert (
            run.trace[-1]["parameters_after_sha256"]
            == restored.trace[-1]["parameters_after_sha256"]
        )
    assert not equal_state(run.snapshot(), restored.snapshot())
