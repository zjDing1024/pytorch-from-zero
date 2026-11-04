import pytest
import torch

from pytorch_lab import wine_batch_budget as b
from pytorch_lab import wine_batch_compute as c
from pytorch_lab.wine_observations import prepare_train


@pytest.mark.parametrize("architecture", ["linear", "mlp16"])
@pytest.mark.parametrize("arm", list(b.ARMS))
def test_reference(architecture, arm):
    torch.set_num_threads(1)
    x, y, _ = prepare_train()
    orders = b.make_orders(42)
    model, row = c.train_timed(x, y, architecture, 42, arm, orders)
    ref, info = b.train_run(x, y, architecture, 42, arm, orders)
    assert all(torch.equal(v, ref.state_dict()[k]) for k, v in model.state_dict().items())
    assert row["final_train_cross_entropy"] == info["final_train_cross_entropy"]
    assert row["budget_matches"]
    assert row["wall_ns"] > 0 and row["process_cpu_ns"] >= 0


@pytest.mark.parametrize("orders", [[], None, [[0] * 107] * 4, [[True] + list(range(1, 107))] * 4])
def test_bad_orders(orders):
    x, y, _ = prepare_train()
    with pytest.raises(ValueError):
        c.train_timed(x, y, "linear", 42, "batch16_14updates", orders)


def test_repeat_contract():
    report = c.run_experiment()
    assert report["all_checks_passed"]
    assert report["protocol"]["total_updates"] == 1080
    for config in report["configurations"]:
        rows = config["observations"]
        assert len(rows) == 12
        assert len([r for r in rows if r["phase"] == "measured"]) == 9
        for arm in b.ARMS:
            values = [r for r in rows if r["arm"] == arm]
            assert len({r["final_parameters_sha256"] for r in values}) == 1
