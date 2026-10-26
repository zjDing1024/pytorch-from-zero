"""Bounded public saved-tensor observation and matched-input log-softmax VJP."""

import argparse
import base64
import importlib.util
import json
import math
import os
import struct
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN_SHA = "2e18232b25b56ca76227604e2d00de99781048db755f3b60d8899c5b443ed27f"
PLAN_ID = "wine-ce-saved-forward-v1"
MAX_BYTES = 8 * 1024 * 1024
MAX_ULPS = 8
ORDER = ["A", "B", "B", "A"]
spec = importlib.util.spec_from_file_location("oct13_reference", ROOT / "reference/oct13/probe.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def protocol():
    payload = (ROOT / "docs/plan.json").read_bytes()
    if reference.sha(payload) != PLAN_SHA:
        raise ValueError("preregistered plan changed")
    value = reference.parse(payload)
    for name, digest in value["inputs"]["reference_files_sha256"].items():
        if reference.sha((ROOT / "reference/oct13" / name).read_bytes()) != digest:
            raise ValueError("reference changed")
    reference.load_fixture()
    return value


def parse(payload):
    if len(payload) > MAX_BYTES:
        raise ValueError("report byte limit")
    try:
        value = json.loads(payload, object_pairs_hook=reference.reject_duplicates)
    except RecursionError as error:
        raise ValueError("JSON nesting limit") from error
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > 300000 or depth > 24:
            raise ValueError("JSON structure limit")
        if type(item) is dict:
            pending.extend((v, depth + 1) for v in item.values())
        elif type(item) is list:
            pending.extend((v, depth + 1) for v in item)
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("nonfinite JSON")
    return value


class SavedObserver:
    """Pack snapshots separately; the packed payload has no original Tensor reference."""

    def __init__(self):
        self.packs = []
        self.unpacks = []

    def pack(self, tensor):
        index = len(self.packs)
        if index >= 8:
            raise ValueError("unexpected saved-tensor count")
        self.packs.append(reference.record(tensor.detach().clone()))
        return index, tensor.detach()

    def unpack(self, payload):
        index, tensor = payload
        self.unpacks.append({"index": index, "tensor": reference.record(tensor)})
        return tensor

    def report(self):
        return {"packs": self.packs, "unpacks": self.unpacks}


def instrumented_step(fixture, observer=None):
    """Same original first-step sequence; only the CE forward is observed."""
    import torch
    from torch.nn import functional as functional

    inputs = {name: reference.tensor_from(value) for name, value in fixture["inputs"].items()}
    params = {name: inputs[name].clone().requires_grad_() for name in reference.PARAM_SHAPES}
    optimizer = torch.optim.SGD(list(params.values()), **reference.OPTIMIZER)
    empty_before = len(optimizer.state) == 0
    pre = functional.linear(inputs["x"], params["weight1"], params["bias1"])
    hidden = functional.relu(pre)
    logits = functional.linear(hidden, params["weight2"], params["bias2"])
    for tensor in (pre, hidden, logits):
        tensor.retain_grad()
    scope = (
        torch.autograd.graph.saved_tensors_hooks(observer.pack, observer.unpack)
        if observer is not None
        else nullcontext()
    )
    with scope:
        loss = functional.cross_entropy(logits, inputs["y"], reduction="mean")
    stages = {"hidden_pre": pre, "hidden_relu": hidden, "logits": logits, "loss": loss}
    loss.backward()
    stages.update(
        {
            "logits_gradient": logits.grad,
            "hidden_relu_gradient": hidden.grad,
            "hidden_pre_gradient": pre.grad,
            **{f"{name}_gradient": value.grad for name, value in params.items()},
        }
    )
    result = [{"name": name, **reference.record(value)} for name, value in stages.items()]
    optimizer.step()
    for name, tensor in params.items():
        result.append(
            {
                "name": f"{name}_momentum",
                **reference.record(optimizer.state[tensor]["momentum_buffer"]),
            }
        )
    result.extend(
        {"name": f"{name}_updated", **reference.record(tensor)} for name, tensor in params.items()
    )
    return result, {
        "optimizer_state_empty_before": empty_before,
        "four_momentum_buffers_after": len(optimizer.state) == 4,
        "inputs_unchanged": all(
            reference.record(v) == fixture["inputs"][k] for k, v in inputs.items()
        ),
    }


def tensor_part(record):
    return {key: value for key, value in record.items() if key != "name"}


def named(stages, name):
    return tensor_part(next(stage for stage in stages if stage["name"] == name))


def ce_decomposition(logits_record, labels):
    from torch.nn import functional

    logits = reference.tensor_from(logits_record).requires_grad_()
    log_probs = functional.log_softmax(logits, dim=1)
    log_probs.retain_grad()
    loss = functional.nll_loss(log_probs, labels, reduction="mean")
    loss.backward()
    return {
        "logits": reference.record(logits),
        "log_probs": reference.record(log_probs),
        "loss": reference.record(loss),
        "upstream": reference.record(log_probs.grad),
        "logits_gradient": reference.record(logits.grad),
    }


def fixed_upstream(labels):
    import torch

    # Each mean-reduced unweighted target gets -1/N, exactly representable here.
    value = torch.zeros((16, 3), dtype=torch.float64)
    for row, label in enumerate(labels.tolist()):
        value[row, label] = -0.0625
    return value


def matched_vjp(logits_record, upstream):
    import torch
    from torch.nn import functional

    logits = reference.tensor_from(logits_record).requires_grad_()
    log_probs = functional.log_softmax(logits, dim=1)
    gradient = torch.autograd.grad(log_probs, logits, grad_outputs=upstream)[0]
    return {
        "logits": reference.record(logits),
        "log_probs": reference.record(log_probs),
        "upstream": reference.record(upstream),
        "logits_gradient": reference.record(gradient),
    }


def local_checks(scientific):
    raw = scientific["raw_before"]["stages"]
    observer = scientific["saved_observer"]
    packs, unpacks = observer["packs"], observer["unpacks"]
    same = all(scientific[name]["stages"] == raw for name in ("observed", "raw_after"))
    return {
        "observer_noninterference": same,
        "all_original_step_checks": all(
            scientific[name]["checks"]
            == {
                "optimizer_state_empty_before": True,
                "four_momentum_buffers_after": True,
                "inputs_unchanged": True,
            }
            for name in ("raw_before", "observed", "raw_after")
        ),
        "packs_nonempty_all_consumed_once": bool(packs)
        and sorted(event["index"] for event in unpacks) == list(range(len(packs))),
        "unpacked_values_unchanged": all(
            event["tensor"] == packs[event["index"]] for event in unpacks
        ),
        "decomposition_input_exact": scientific["decomposed"]["logits"] == named(raw, "logits"),
        "matched_input_exact": scientific["matched"]["logits"] == named(raw, "logits"),
        "matched_upstream_exact": scientific["matched"]["upstream"] == scientific["fixed_upstream"],
    }


def observed_relations(scientific):
    """Relations are outcomes, never required assumptions for capture validity."""
    raw, decomp, matched = (
        scientific["raw_before"]["stages"],
        scientific["decomposed"],
        scientific["matched"],
    )
    return {
        "saved_pack_indices_equal_decomposed_log_probs": [
            i
            for i, value in enumerate(scientific["saved_observer"]["packs"])
            if value == decomp["log_probs"]
        ],
        "decomposed_loss_matches_original": decomp["loss"] == named(raw, "loss"),
        "decomposed_gradient_matches_original": decomp["logits_gradient"]
        == named(raw, "logits_gradient"),
        "decomposed_upstream_matches_fixed": decomp["upstream"] == scientific["fixed_upstream"],
        "matched_log_probs_equal_decomposed": matched["log_probs"] == decomp["log_probs"],
        "matched_gradient_matches_original": matched["logits_gradient"]
        == named(raw, "logits_gradient"),
    }


def worker():
    protocol()
    reference.configure_torch()
    fixture, manifest = reference.load_fixture()
    scientific = {
        "plan_id": PLAN_ID,
        "plan_sha256": PLAN_SHA,
        "fixture_sha256": manifest["fixture_sha256"],
        "input_hashes": manifest["input_hashes"],
    }
    before, checks = reference.step(fixture)
    scientific["raw_before"] = {"stages": before, "checks": checks}
    observer = SavedObserver()
    stages, checks = instrumented_step(fixture, observer)
    scientific["observed"] = {"stages": stages, "checks": checks}
    stages, checks = reference.step(fixture)
    scientific["raw_after"] = {"stages": stages, "checks": checks}
    scientific["saved_observer"] = observer.report()
    labels = reference.tensor_from(fixture["inputs"]["y"])
    logits = named(before, "logits")
    scientific["decomposed"] = ce_decomposition(logits, labels)
    upstream = fixed_upstream(labels)
    scientific["fixed_upstream"] = reference.record(upstream)
    scientific["matched"] = matched_vjp(logits, upstream)
    return {
        "scientific": scientific,
        "scientific_sha256": reference.sha(reference.canonical(scientific)),
        "runtime": reference.runtime(),
        "source_sha256": reference.sha(Path(__file__).read_bytes()),
        "checks": local_checks(scientific),
        "relations": observed_relations(scientific),
    }


def validate_sample(sample):
    protocol()
    parse(reference.canonical(sample))
    if type(sample) is not dict or set(sample) != {
        "scientific",
        "scientific_sha256",
        "runtime",
        "source_sha256",
        "checks",
        "relations",
    }:
        raise ValueError("sample schema")
    if sample["source_sha256"] != reference.sha(Path(__file__).read_bytes()):
        raise ValueError("worker source changed")
    reference.validate_runtime(sample["runtime"])
    science = sample["scientific"]
    if type(science) is not dict or set(science) != {
        "plan_id",
        "plan_sha256",
        "fixture_sha256",
        "input_hashes",
        "raw_before",
        "observed",
        "raw_after",
        "saved_observer",
        "decomposed",
        "fixed_upstream",
        "matched",
    }:
        raise ValueError("scientific schema")
    _, manifest = reference.load_fixture()
    expected = {
        "plan_id": PLAN_ID,
        "plan_sha256": PLAN_SHA,
        "fixture_sha256": manifest["fixture_sha256"],
        "input_hashes": manifest["input_hashes"],
    }
    for key, value in expected.items():
        if science[key] != value:
            raise ValueError("scientific protocol/input mismatch")
    for name in ("raw_before", "observed", "raw_after"):
        item = science[name]
        if type(item) is not dict or set(item) != {"stages", "checks"}:
            raise ValueError("first-step schema")
        original = {
            "plan_id": reference.PLAN_ID,
            "plan_sha256": reference.PLAN_SHA,
            "fixture_sha256": manifest["fixture_sha256"],
            "input_hashes": manifest["input_hashes"],
            "stages": item["stages"],
        }
        reference.validate_sample(
            {
                "scientific": original,
                "scientific_sha256": reference.sha(reference.canonical(original)),
                "runtime": sample["runtime"],
                "checks": item["checks"],
            }
        )
    observer = science["saved_observer"]
    if type(observer) is not dict or set(observer) != {"packs", "unpacks"}:
        raise ValueError("observer schema")
    packs, unpacks = observer["packs"], observer["unpacks"]
    if (
        type(packs) is not list
        or not 1 <= len(packs) <= 8
        or type(unpacks) is not list
        or len(unpacks) > 16
    ):
        raise ValueError("observer count")
    for record in packs:
        if type(record) is not dict or record.get("dtype") not in ("float64", "int64"):
            raise ValueError("saved dtype")
        shape = record.get("shape")
        if shape not in ([16, 3], [16], []):
            raise ValueError("saved shape")
        if record["dtype"] == "int64" and shape != [16]:
            raise ValueError("saved integer shape")
        reference.validate_record(record, shape, record["dtype"])
    for event in unpacks:
        if type(event) is not dict or set(event) != {"index", "tensor"}:
            raise ValueError("unpack schema")
        index = event["index"]
        if type(index) is not int or not 0 <= index < len(packs):
            raise ValueError("unpack index")
        reference.validate_record(event["tensor"], packs[index]["shape"], packs[index]["dtype"])
    for name in ("decomposed", "matched"):
        shapes = {
            "logits": [16, 3],
            "log_probs": [16, 3],
            "upstream": [16, 3],
            "logits_gradient": [16, 3],
        }
        if name == "decomposed":
            shapes["loss"] = []
        if type(science[name]) is not dict or set(science[name]) != set(shapes):
            raise ValueError("decomposition schema")
        for key, shape in shapes.items():
            reference.validate_record(science[name][key], shape)
    reference.validate_record(science["fixed_upstream"], [16, 3])
    fixture, _ = reference.load_fixture()
    expected_upstream = [0.0] * 48
    for row, label in enumerate(fixture["inputs"]["y"]["values"]):
        expected_upstream[3 * row + label] = -0.0625
    if science["fixed_upstream"]["values"] != expected_upstream or science["fixed_upstream"][
        "sha256_be"
    ] != reference.sha(struct.pack(">48d", *expected_upstream)):
        raise ValueError("fixed upstream changed")
    if sample["scientific_sha256"] != reference.sha(reference.canonical(science)):
        raise ValueError("scientific hash")
    if reference.canonical(sample["checks"]) != reference.canonical(local_checks(science)) or any(
        type(v) is not bool for v in sample["checks"].values()
    ):
        raise ValueError("inconsistent checks")
    if reference.canonical(sample["relations"]) != reference.canonical(observed_relations(science)):
        raise ValueError("inconsistent relations")


def records(scientific):
    values = {}
    for mode in ("raw_before", "observed", "raw_after"):
        values.update({f"{mode}/{s['name']}": tensor_part(s) for s in scientific[mode]["stages"]})
    values.update(
        {f"saved_pack/{i}": v for i, v in enumerate(scientific["saved_observer"]["packs"])}
    )
    for mode in ("decomposed", "matched"):
        values.update({f"{mode}/{k}": v for k, v in scientific[mode].items()})
    values["fixed_upstream"] = scientific["fixed_upstream"]
    return values


def compare_record(left, right):
    if left["shape"] != right["shape"] or left["dtype"] != right["dtype"]:
        raise ValueError("comparison structure differs")
    changed = left["sha256_be"] != right["sha256_be"]
    pairs = list(zip(left["values"], right["values"], strict=True))
    result = {"count": len(pairs), "strict_equal": not changed}
    if left["dtype"] == "int64":
        return {
            **result,
            "changed_values": sum(x != y for x, y in pairs),
            "exact_integer_equal": not changed,
        }
    distances = [reference.ulps(x, y) for x, y in pairs]
    return {
        **result,
        "changed_values": sum(struct.pack(">d", x) != struct.pack(">d", y) for x, y in pairs),
        "max_ulps": max(distances),
        "over_budget": sum(d > MAX_ULPS for d in distances),
        "max_abs_diff": max(abs(x - y) for x, y in pairs),
    }


def compare(left, right):
    a, b = records(left["scientific"]), records(right["scientific"])
    if list(a) != list(b):
        return {"comparable": False, "reason": "record schema differs"}
    try:
        result = {name: compare_record(a[name], b[name]) for name in a}
    except ValueError:
        return {"comparable": False, "reason": "record shape/dtype differs"}
    return {
        "comparable": True,
        "strict_equal": left["scientific_sha256"] == right["scientific_sha256"],
        "within_8_ulps_and_exact_integers": all(
            v.get("over_budget", 0) == 0 and v.get("exact_integer_equal", True)
            for v in result.values()
        ),
        "records": result,
    }


def summarize(samples):
    checks = {
        "all_four_workers_completed": len(samples) == 4,
        "local_controls_valid": bool(samples) and all(all(s["checks"].values()) for s in samples),
        "requested_capability_matches": all(
            s["runtime"]["environment"]["ATEN_CPU_CAPABILITY"]
            == (None if label == "A" else "default")
            for s, label in zip(samples, ORDER, strict=False)
        ),
        "nonintervention_controls_equal": bool(samples)
        and all(
            reference.normalized_runtime(s["runtime"])
            == reference.normalized_runtime(samples[0]["runtime"])
            and s["source_sha256"] == samples[0]["source_sha256"]
            for s in samples
        ),
        "effective_intervention": len(samples) == 4
        and samples[0]["runtime"]["capability"] == samples[3]["runtime"]["capability"] != "DEFAULT"
        and samples[1]["runtime"]["capability"] == samples[2]["runtime"]["capability"] == "DEFAULT",
        "A_strict_repeat": len(samples) == 4
        and samples[0]["scientific_sha256"] == samples[3]["scientific_sha256"],
        "B_strict_repeat": len(samples) >= 3
        and samples[1]["scientific_sha256"] == samples[2]["scientific_sha256"],
    }
    return {
        "checks": checks,
        "capture_valid": all(checks.values()),
        "comparisons_to_A1": [compare(samples[0], s) for s in samples[1:]] if samples else [],
    }


def byte_evidence(payload, limit=2 * 1024 * 1024):
    """Preserve bounded exact bytes; record full-stream hash/length even if truncated."""
    if payload is None:
        payload = b""
    if type(payload) is str:
        payload = payload.encode("utf-8")
    if type(payload) is not bytes:
        raise TypeError("worker evidence must be bytes")
    prefix = payload[:limit]
    return {
        "bytes": len(payload),
        "sha256": reference.sha(payload),
        "retained_bytes": len(prefix),
        "truncated": len(prefix) != len(payload),
        "base64": base64.b64encode(prefix).decode("ascii"),
    }


def capture(timeout=60, runner=subprocess.run):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("timeout must be finite and in (0,60]")
    protocol()
    samples, failure = [], None
    base = dict(os.environ, **{name: "1" for name in reference.THREAD_ENV})
    base.pop("ATEN_CPU_CAPABILITY", None)
    start, started = time.monotonic(), datetime.now(UTC).isoformat()
    for label in ORDER:
        done = None
        env = dict(base)
        if label == "B":
            env["ATEN_CPU_CAPABILITY"] = "default"
        try:
            done = runner(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
                env=env,
                capture_output=True,
                check=True,
                timeout=timeout,
            )
            sample = parse(done.stdout)
            validate_sample(sample)
            samples.append(sample)
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
            OverflowError,
        ) as error:
            failure = {
                "condition": label,
                "completed_workers": len(samples),
                "category": type(error).__name__,
                "returncode": getattr(error, "returncode", getattr(done, "returncode", None)),
                "stdout": byte_evidence(getattr(error, "stdout", getattr(done, "stdout", None))),
                "stderr": byte_evidence(getattr(error, "stderr", getattr(done, "stderr", None))),
            }
            break
    return {
        "plan_id": PLAN_ID,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - start,
        "order": ORDER,
        "samples": samples,
        "failure": failure,
        **summarize(samples),
        "meaning": "Capture validity is separate from strict and 8-ULP outcomes.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    try:
        if args.worker:
            print(json.dumps(worker(), allow_nan=False))
            return 0
        if (
            args.output is None
            or args.output.exists()
            or args.output.is_symlink()
            or not args.output.parent.is_dir()
        ):
            raise ValueError("provide new output path in existing directory")
        value = capture(args.timeout)
        reference.write_new(args.output, value)
        print(
            json.dumps({"capture_valid": value["capture_valid"], "workers": len(value["samples"])})
        )
        return 0 if value["capture_valid"] else 1
    except (ValueError, OSError, KeyError, OverflowError, TypeError) as error:
        print(f"probe failed: {type(error).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
