"""Independent review: safe-point grid, malformed snapshots and fresh processes.

Run with the repository installed in the current Python environment:
    python validation/verify_accumulation_recovery.py

Uses private fixture state only as a white-box test, not as a supported user API.
Outputs sanitized JSON to stdout. Does not mutate tracked repository files.
"""

import copy
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from pytorch_lab.accumulation import train_accumulated_epoch
from pytorch_lab.accumulation_recovery import (
    WindowConfig,
    WindowTrainer,
    load_window_checkpoint,
    restore_snapshot,
    save_window_checkpoint,
)
from pytorch_lab.checkpoint import MAX_CHECKPOINT_BYTES, CheckpointError, _content_digest
from pytorch_lab.minibatch import AffineRegressor


def seal(p):
    p["content_sha256"] = _content_digest({k: v for k, v in p.items() if k != "content_sha256"})
    return p


def run():
    torch.set_num_threads(1)
    cases = points = 0
    for n, micro, accum, seed in itertools.product((1, 11, 23), (1, 5, 30), (1, 3), (7, 42)):
        cfg = WindowConfig(samples=n, microbatch_size=micro, accumulation_steps=accum, seed=seed)
        continuous = WindowTrainer(cfg)
        continuous.train_to(2)
        target = continuous.snapshot()
        # Reference is yesterday's stateless accumulation function, not new replay logic.
        model = AffineRegressor(3, 2)
        opt = torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        for _ in range(2):
            d = continuous._data
            batches = (
                (d.features[i : i + micro], d.targets[i : i + micro]) for i in range(0, n, micro)
            )
            train_accumulated_epoch(model, batches, opt, accum)
        assert _content_digest(dict(model.state_dict())) == _content_digest(target["model"])
        assert _content_digest(opt.state_dict()) == _content_digest(target["optimizer"])
        part = WindowTrainer(cfg)
        while True:
            snap = part.snapshot()
            restored = restore_snapshot(snap, cfg)
            restored.train_to(2)
            assert restored.snapshot()["content_sha256"] == target["content_sha256"], (
                n,
                micro,
                accum,
                seed,
                part.epoch,
                part.cursor,
            )
            points += 1
            if part.epoch == 2:
                break
            part.step_microbatch()
        cases += 1
    trainer = WindowTrainer()
    trainer.step_microbatch()
    trainer.step_microbatch()
    original = trainer.snapshot()
    mutations = []
    for key, value in [
        ("epoch", True),
        ("cursor", True),
        ("updates", True),
        ("pending_batches", True),
        ("pending_samples", True),
        ("pending_elements", True),
        ("cursor", 6),
        ("cursor", 0),
        ("pending_elements", 10),
        ("pending_samples", 5),
        ("pending_batches", 1),
        ("updates", 1),
        ("epoch", 101),
        ("data_sha256", "0" * 64),
        ("torch_version", "incorrect"),
        ("order", "shuffled"),
        ("training_rng", "missing"),
    ]:
        p = copy.deepcopy(original)
        p[key] = value
        mutations.append((key, p))
    for name, mut in [
        ("missing_grad", lambda p: p["pending_gradients"].__setitem__("weight", None)),
        ("nonfinite_grad", lambda p: p["pending_gradients"]["weight"].fill_(float("nan"))),
        (
            "wrong_dtype",
            lambda p: p["pending_gradients"].__setitem__(
                "weight", p["pending_gradients"]["weight"].float()
            ),
        ),
        (
            "wrong_shape",
            lambda p: p["model"].__setitem__("bias", torch.zeros(1, dtype=torch.float64)),
        ),
        ("extra_field", lambda p: p.__setitem__("unexpected", 1)),
        (
            "bool_optimizer_id",
            lambda p: p["optimizer"]["param_groups"][0]["params"].__setitem__(0, False),
        ),
        (
            "wrong_learning_rate",
            lambda p: p["optimizer"]["param_groups"][0].__setitem__("lr", 0.04),
        ),
        ("config_bool", lambda p: p["config"].__setitem__("seed", True)),
    ]:
        p = copy.deepcopy(original)
        mut(p)
        mutations.append((name, p))
    rejected = 0
    for name, p in mutations:
        # Validation must reject non-finite tensors even with a recomputed checksum.
        try:
            seal(p)
            restore_snapshot(p)
        except (CheckpointError, ValueError, TypeError):
            rejected += 1
        else:
            raise AssertionError("accepted invalid state " + name)
    assert trainer.snapshot()["content_sha256"] == original["content_sha256"]
    # Valid shape + resealed zero gradients is not authenticated: demonstrate replay failure.
    p = copy.deepcopy(original)
    for g in p["pending_gradients"].values():
        g.zero_()
    lost = restore_snapshot(seal(p))
    lost.train_to(2)
    baseline = restore_snapshot(original)
    baseline.train_to(2)
    assert lost.snapshot()["content_sha256"] != baseline.snapshot()["content_sha256"]
    # Caller snapshot isolation.
    s = trainer.snapshot()
    s["model"]["weight"].fill_(99)
    s["pending_gradients"]["bias"].zero_()
    assert trainer.snapshot()["content_sha256"] == original["content_sha256"]
    # Failure invalidates continuation and snapshots, rather than exporting partial work.
    failed = WindowTrainer()
    failed.step_microbatch()
    failed._data.features[5, 0] = float("nan")
    try:
        failed.step_microbatch()
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite input accepted")
    for action in (failed.snapshot, failed.step_microbatch, lambda: failed.train_to(2)):
        try:
            action()
        except CheckpointError:
            pass
        else:
            raise AssertionError("failed state remained usable")
    # RNG is not touched by fixed-data construction, load, or training.
    rng = torch.get_rng_state().clone()
    r = WindowTrainer()
    r.train_to(2)
    restore_snapshot(r.snapshot())
    assert torch.equal(rng, torch.get_rng_state())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "own-checkpoint.pt"
        save_window_checkpoint(trainer, path)
        before = path.read_bytes()
        try:
            save_window_checkpoint(trainer, path)
        except FileExistsError:
            pass
        else:
            raise AssertionError("clobbered checkpoint")
        assert path.read_bytes() == before
        assert (
            load_window_checkpoint(path).snapshot()["content_sha256"] == original["content_sha256"]
        )
        file_rejections = 0
        bad_path = Path(tmp) / "invalid.pt"
        for content in (b"not a checkpoint", before[:32], b"0" * (MAX_CHECKPOINT_BYTES + 1)):
            bad_path.write_bytes(content)
            try:
                load_window_checkpoint(bad_path)
            except CheckpointError:
                file_rejections += 1
            else:
                raise AssertionError("invalid file accepted")
        try:
            load_window_checkpoint(path, WindowConfig(seed=7))
        except CheckpointError:
            file_rejections += 1
        else:
            raise AssertionError("expected config mismatch accepted")
        script = """import json,sys,torch
from pytorch_lab.accumulation_recovery import (
 WindowTrainer, save_window_checkpoint, load_window_checkpoint,
)
torch.set_num_threads(1)
mode,path=sys.argv[1:]
if mode=="save":
 t=WindowTrainer()
 for _ in range(4):t.step_microbatch()
 save_window_checkpoint(t,path)
 print(t.snapshot()["content_sha256"])
else:
 t=load_window_checkpoint(path) if mode=="resume" else WindowTrainer()
 t.train_to(3)
 print(t.snapshot()["content_sha256"])
"""
        fresh = Path(tmp) / "fresh.pt"

        def child(mode):
            return subprocess.run(
                [sys.executable, "-I", "-c", script, mode, str(fresh)],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            ).stdout.strip()

        child("save")
        assert child("resume") == child("continuous")
    return {
        "all_checks_passed": True,
        "configurations": cases,
        "safe_point_roundtrips": points,
        "malformed_snapshots_rejected": rejected,
        "invalid_files_or_expected_config_rejected": file_rejections,
        "reference": "prior stateless accumulation function",
        "exact_full_state_replay": True,
        "fresh_process_tail_window_replay": True,
        "zero_gradient_negative_control_detected": True,
        "snapshot_isolation": True,
        "failure_invalidation": True,
        "global_rng_unchanged": True,
        "no_clobber_save": True,
        "scope": "same CPU float64 fixed ordered affine fixture; trusted own checkpoints only",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
