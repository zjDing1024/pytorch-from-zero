"""Original, bounded frozen Wine first-step investigation; no package mutation."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN_SHA = "5b79ea625d6693b4203a18bddd424cb01e04f3e8e1f14459d5bbefe9afcad951"
PLAN_ID = "wine-first-step-dispatch-v1"
FROZEN_SHA = "4db70a0c4f889e1a0adb5d3ed8c4c17e0eb36871b4c85e82885672e77fcfddf7"
MAX_BYTES = 2 * 1024 * 1024
MAX_ULPS = 8
PARAM_SHAPES = {"weight1": [16, 13], "bias1": [16], "weight2": [3, 16], "bias2": [3]}
INPUT_SHAPES = {"x": [16, 13], "y": [16], **PARAM_SHAPES}
STAGE_SHAPES = {
    "hidden_pre": [16, 16],
    "hidden_relu": [16, 16],
    "logits": [16, 3],
    "loss": [],
    "logits_gradient": [16, 3],
    "hidden_relu_gradient": [16, 16],
    "hidden_pre_gradient": [16, 16],
    **{f"{name}_gradient": shape for name, shape in PARAM_SHAPES.items()},
    **{f"{name}_momentum": shape for name, shape in PARAM_SHAPES.items()},
    **{f"{name}_updated": shape for name, shape in PARAM_SHAPES.items()},
}
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
OPTIMIZER = {
    "lr": 0.05,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "dampening": 0.0,
    "nesterov": False,
    "maximize": False,
    "foreach": False,
    "fused": False,
}


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse(payload):
    if len(payload) > MAX_BYTES:
        raise ValueError("report exceeds byte limit")
    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except RecursionError as error:
        raise ValueError("JSON nesting exceeds parser limit") from error
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > 50000 or depth > 20:
            raise ValueError("JSON structure limit")
        if type(item) is dict:
            pending.extend((x, depth + 1) for x in item.values())
        elif type(item) is list:
            pending.extend((x, depth + 1) for x in item)
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("nonfinite JSON")
    return value


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def record(tensor):
    values = tensor.detach().reshape(-1).tolist()
    dtype = str(tensor.dtype).removeprefix("torch.")
    code = {"float64": "d", "int64": "q"}[dtype]
    if any(not math.isfinite(x) for x in values):
        raise ValueError("nonfinite tensor")
    return {
        "shape": list(tensor.shape),
        "dtype": dtype,
        "values": values,
        "sha256_be": sha(struct.pack(f">{len(values)}{code}", *values)),
    }


def validate_record(value, shape, dtype="float64"):
    if type(value) is not dict or set(value) != {
        "shape",
        "dtype",
        "values",
        "sha256_be",
    }:
        raise ValueError("tensor schema")
    if value["shape"] != shape or any(type(x) is not int for x in value["shape"]):
        raise ValueError("tensor shape")
    if value["dtype"] != dtype or type(value["values"]) is not list:
        raise ValueError("tensor dtype/values")
    values = value["values"]
    if len(values) != math.prod(shape):
        raise ValueError("tensor count")
    typ, code = (float, "d") if dtype == "float64" else (int, "q")
    if any(type(x) is not typ or not math.isfinite(x) for x in values):
        raise ValueError("tensor scalar type/finite")
    if value["sha256_be"] != sha(struct.pack(f">{len(values)}{code}", *values)):
        raise ValueError("tensor hash")


def tensor_from(value):
    import torch

    return torch.tensor(value["values"], dtype=getattr(torch, value["dtype"])).reshape(
        value["shape"]
    )


def plan():
    payload = (ROOT / "docs/plan.json").read_bytes()
    if sha(payload) != PLAN_SHA:
        raise ValueError("preregistered plan changed")
    result = parse(payload)
    if result["stage_order"] != list(STAGE_SHAPES):
        raise ValueError("stage order differs from plan")
    return result


def configure_torch():
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def runtime():
    import torch

    inventory = []
    for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
        item = {}
        for line in block.splitlines():
            key, sep, val = line.partition(":")
            key = key.strip()
            if sep and key in {
                "vendor_id",
                "model name",
                "cpu family",
                "model",
                "flags",
            }:
                item[key] = sorted(val.split()) if key == "flags" else val.strip()
        if item and item not in inventory:
            inventory.append(item)
    config = "\n".join(
        x for x in torch.__config__.show().splitlines() if "CPU capability usage:" not in x
    )
    root = Path(torch.__file__).parent
    package_hashes = {}
    for name in (
        "__init__.py",
        "optim/sgd.py",
        "lib/libtorch_cpu.so",
        "lib/libtorch_python.so",
    ):
        digest = hashlib.sha256()
        with (root / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        package_hashes[name] = digest.hexdigest()
    allowed = {
        "ATEN_CPU_CAPABILITY": {"default", "avx2", "avx512"},
        "MKL_CBWR": {"AUTO", "COMPATIBLE", "SSE4_2", "AVX", "AVX2", "AVX512"},
        "MKL_ENABLE_INSTRUCTIONS": {"SSE4_2", "AVX", "AVX2", "AVX512"},
        "ONEDNN_MAX_CPU_ISA": {"DEFAULT", "SSE41", "AVX", "AVX2", "AVX512_CORE"},
        **{name: {"1"} for name in THREAD_ENV},
    }
    controls = {}
    for name, accepted in allowed.items():
        raw = os.environ.get(name)
        controls[name] = raw if raw is None or raw in accepted else "redacted-unrecognized"
    return {
        "cpu_models_and_features": inventory,
        "affinity": sorted(os.sched_getaffinity(0)),
        "python": platform.python_version(),
        "kernel": platform.release(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "torch_build_sha256": sha(config.encode()),
        "torch_files_sha256": package_hashes,
        "capability": torch.backends.cpu.get_cpu_capability(),
        "environment": controls,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "source_sha256": sha(Path(__file__).read_bytes()),
    }


def freeze():
    """Freeze input only: this function performs no forward/loss/backward/update."""
    spec = plan()
    torch = configure_torch()
    if os.environ.get("ATEN_CPU_CAPABILITY") is not None:
        raise ValueError("freeze must use A with capability unset")
    data = (ROOT / "data/wine.csv").read_bytes()
    split_bytes = (ROOT / "data/wine_split.json").read_bytes()
    recipe = spec["fixture_recipe"]
    if sha(data) != recipe["raw_data_sha256"] or sha(split_bytes) != recipe["split_sha256"]:
        raise ValueError("dataset/split bytes changed")
    rows = list(csv.reader(io.StringIO(data.decode())))[1:]
    ids = parse(split_bytes)["row_ids"]["train"]
    if len(ids) != 107 or len(set(ids)) != 107 or any(type(x) is not int for x in ids):
        raise ValueError("train split IDs")
    x = torch.tensor([[float(v) for v in rows[i][:13]] for i in ids], dtype=torch.float64)
    y = torch.tensor([int(rows[i][13]) for i in ids], dtype=torch.int64)
    mean, scale = x.mean(dim=0), x.std(dim=0, correction=0)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    normalized = (x - mean) / scale
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(normalized, y, torch.tensor(ids)),
        batch_size=16,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(42),
    )
    batch_x, batch_y, batch_ids = next(iter(loader))
    generator = torch.Generator().manual_seed(42)
    tensors = {"x": batch_x, "y": batch_y}
    for name, shape in PARAM_SHAPES.items():
        if name.startswith("weight"):
            bound = 1 / math.sqrt(shape[1])
            tensors[name] = torch.empty(shape, dtype=torch.float64).uniform_(
                -bound, bound, generator=generator
            )
        else:
            tensors[name] = torch.zeros(shape, dtype=torch.float64)
    fixture = {
        "schema_version": 1,
        "plan_id": PLAN_ID,
        "plan_sha256": PLAN_SHA,
        "train_row_ids": ids,
        "batch_row_ids": batch_ids.tolist(),
        "preprocessing": {"mean": record(mean), "scale": record(scale)},
        "inputs": {name: record(value) for name, value in tensors.items()},
    }
    write_new(ROOT / "data/frozen_fixture.json", fixture)
    manifest = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "plan_sha256": PLAN_SHA,
        "fixture_sha256": sha((ROOT / "data/frozen_fixture.json").read_bytes()),
        "source_sha256_at_freeze": sha(Path(__file__).read_bytes()),
        "runtime": runtime(),
        "input_hashes": {name: value["sha256_be"] for name, value in fixture["inputs"].items()},
        "model_evaluations_before_freeze": 0,
    }
    write_new(ROOT / "docs/freeze.json", manifest)
    return manifest


def load_fixture():
    plan()
    payload = (ROOT / "data/frozen_fixture.json").read_bytes()
    manifest = parse((ROOT / "docs/freeze.json").read_bytes())
    if (
        manifest["plan_sha256"] != PLAN_SHA
        or sha(payload) != FROZEN_SHA
        or manifest["fixture_sha256"] != FROZEN_SHA
    ):
        raise ValueError("frozen bytes changed")
    value = parse(payload)
    if value["schema_version"] != 1 or value["plan_id"] != PLAN_ID:
        raise ValueError("fixture protocol")
    if value["plan_sha256"] != PLAN_SHA or set(value["inputs"]) != set(INPUT_SHAPES):
        raise ValueError("fixture schema")
    for name, shape in INPUT_SHAPES.items():
        validate_record(value["inputs"][name], shape, "int64" if name == "y" else "float64")
        if value["inputs"][name]["sha256_be"] != manifest["input_hashes"][name]:
            raise ValueError("frozen tensor hash changed")
    if any(x not in (0, 1, 2) for x in value["inputs"]["y"]["values"]):
        raise ValueError("class labels")
    if len(value["batch_row_ids"]) != 16 or len(set(value["batch_row_ids"])) != 16:
        raise ValueError("batch IDs")
    if not set(value["batch_row_ids"]) <= set(value["train_row_ids"]):
        raise ValueError("nontrain row")
    return value, manifest


def step(fixture):
    import torch
    from torch.nn import functional

    inputs = {name: tensor_from(value) for name, value in fixture["inputs"].items()}
    params = {name: inputs[name].clone().requires_grad_() for name in PARAM_SHAPES}
    optimizer = torch.optim.SGD(list(params.values()), **OPTIMIZER)
    empty_before = len(optimizer.state) == 0
    hidden_pre = functional.linear(inputs["x"], params["weight1"], params["bias1"])
    hidden_relu = functional.relu(hidden_pre)
    logits = functional.linear(hidden_relu, params["weight2"], params["bias2"])
    for value in (hidden_pre, hidden_relu, logits):
        value.retain_grad()
    loss = functional.cross_entropy(logits, inputs["y"], reduction="mean")
    stages = {
        "hidden_pre": hidden_pre,
        "hidden_relu": hidden_relu,
        "logits": logits,
        "loss": loss,
    }
    loss.backward()
    stages.update(
        {
            "logits_gradient": logits.grad,
            "hidden_relu_gradient": hidden_relu.grad,
            "hidden_pre_gradient": hidden_pre.grad,
            **{f"{name}_gradient": value.grad for name, value in params.items()},
        }
    )
    result = [{"name": name, **record(value)} for name, value in stages.items()]
    optimizer.step()
    for name, value in params.items():
        result.append(
            {
                "name": f"{name}_momentum",
                **record(optimizer.state[value]["momentum_buffer"]),
            }
        )
    for name, value in params.items():
        result.append({"name": f"{name}_updated", **record(value)})
    return result, {
        "optimizer_state_empty_before": empty_before,
        "four_momentum_buffers_after": len(optimizer.state) == 4,
        "inputs_unchanged": all(record(t) == fixture["inputs"][n] for n, t in inputs.items()),
    }


def worker():
    configure_torch()
    fixture, manifest = load_fixture()
    stages, checks = step(fixture)
    scientific = {
        "plan_id": PLAN_ID,
        "plan_sha256": PLAN_SHA,
        "fixture_sha256": manifest["fixture_sha256"],
        "input_hashes": manifest["input_hashes"],
        "stages": stages,
    }
    return {
        "scientific": scientific,
        "scientific_sha256": sha(canonical(scientific)),
        "runtime": runtime(),
        "checks": checks,
    }


def validate_sample(sample):
    parse(canonical(sample))
    if type(sample) is not dict or set(sample) != {
        "scientific",
        "scientific_sha256",
        "runtime",
        "checks",
    }:
        raise ValueError("sample schema")
    _, manifest = load_fixture()
    scientific = sample["scientific"]
    if type(scientific) is not dict or set(scientific) != {
        "plan_id",
        "plan_sha256",
        "fixture_sha256",
        "input_hashes",
        "stages",
    }:
        raise ValueError("scientific schema")
    for key, expected in {
        "plan_id": PLAN_ID,
        "plan_sha256": PLAN_SHA,
        "fixture_sha256": manifest["fixture_sha256"],
        "input_hashes": manifest["input_hashes"],
    }.items():
        if scientific[key] != expected:
            raise ValueError("scientific protocol/inputs changed")
    stages = scientific["stages"]
    if (
        type(stages) is not list
        or any(type(x) is not dict or "name" not in x for x in stages)
        or [x["name"] for x in stages] != list(STAGE_SHAPES)
    ):
        raise ValueError("stage order")
    for stage in stages:
        validate_record(
            {k: v for k, v in stage.items() if k != "name"}, STAGE_SHAPES[stage["name"]]
        )
    if sha(canonical(scientific)) != sample["scientific_sha256"]:
        raise ValueError("scientific hash")
    if sample["checks"] != {
        "optimizer_state_empty_before": True,
        "four_momentum_buffers_after": True,
        "inputs_unchanged": True,
    } or any(type(x) is not bool for x in sample["checks"].values()):
        raise ValueError("step checks")
    validate_runtime(sample["runtime"])


def validate_runtime(details):
    fields = {
        "cpu_models_and_features",
        "affinity",
        "python",
        "kernel",
        "torch",
        "torch_git",
        "torch_build_sha256",
        "torch_files_sha256",
        "capability",
        "environment",
        "threads",
        "interop_threads",
        "deterministic_algorithms",
        "source_sha256",
    }
    if type(details) is not dict or set(details) != fields:
        raise ValueError("runtime schema")
    for key in ("threads", "interop_threads"):
        if type(details[key]) is not int or details[key] != 1:
            raise ValueError("runtime thread control")
    if details["deterministic_algorithms"] is not True:
        raise ValueError("runtime deterministic control")
    for key in ("python", "kernel", "torch"):
        if type(details[key]) is not str or not 1 <= len(details[key]) <= 200:
            raise ValueError("runtime version string")
    if details["capability"] not in ("DEFAULT", "AVX2", "AVX512"):
        raise ValueError("unsupported effective capability")
    for key, length in (("torch_git", 40), ("torch_build_sha256", 64), ("source_sha256", 64)):
        if type(details[key]) is not str or not re.fullmatch(f"[0-9a-f]{{{length}}}", details[key]):
            raise ValueError("runtime hash")
    if details["source_sha256"] != sha(Path(__file__).read_bytes()):
        raise ValueError("worker source changed")
    hashes = details["torch_files_sha256"]
    if (
        type(hashes) is not dict
        or set(hashes)
        != {"__init__.py", "optim/sgd.py", "lib/libtorch_cpu.so", "lib/libtorch_python.so"}
        or any(type(v) is not str or not re.fullmatch("[0-9a-f]{64}", v) for v in hashes.values())
    ):
        raise ValueError("runtime package hashes")
    affinity = details["affinity"]
    if (
        type(affinity) is not list
        or not affinity
        or any(type(x) is not int or not 0 <= x < 1_000_000 for x in affinity)
        or sorted(set(affinity)) != affinity
    ):
        raise ValueError("runtime affinity")
    cpus = details["cpu_models_and_features"]
    if type(cpus) is not list or not cpus or len(cpus) > 256:
        raise ValueError("runtime CPUs")
    for cpu in cpus:
        if type(cpu) is not dict or set(cpu) != {
            "vendor_id",
            "model name",
            "cpu family",
            "model",
            "flags",
        }:
            raise ValueError("runtime CPU schema")
        for key in ("vendor_id", "model name", "cpu family", "model"):
            if type(cpu[key]) is not str or not 1 <= len(cpu[key]) <= 256:
                raise ValueError("runtime CPU scalar")
        flags = cpu["flags"]
        if (
            type(flags) is not list
            or not flags
            or any(type(v) is not str or not re.fullmatch("[A-Za-z0-9_]+", v) for v in flags)
            or sorted(set(flags)) != flags
        ):
            raise ValueError("runtime CPU flags")
    environment = details["environment"]
    allowed = {
        "ATEN_CPU_CAPABILITY": {None, "default", "avx2", "avx512"},
        "MKL_CBWR": {None, "AUTO", "COMPATIBLE", "SSE4_2", "AVX", "AVX2", "AVX512"},
        "MKL_ENABLE_INSTRUCTIONS": {None, "SSE4_2", "AVX", "AVX2", "AVX512"},
        "ONEDNN_MAX_CPU_ISA": {None, "DEFAULT", "SSE41", "AVX", "AVX2", "AVX512_CORE"},
        **{name: {"1"} for name in THREAD_ENV},
    }
    if type(environment) is not dict or set(environment) != set(allowed):
        raise ValueError("runtime environment schema")
    for name, accepted in allowed.items():
        if type(environment[name]) not in (str, type(None)) or environment[name] not in accepted:
            raise ValueError("runtime unrecognized environment control")


def ulps(left, right):
    if type(left) is not float or type(right) is not float:
        raise ValueError("binary64 scalars required")
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("finite scalars required")

    def key(value):
        bits = struct.unpack(">Q", struct.pack(">d", value))[0]
        magnitude = bits & ((1 << 63) - 1)
        return (1 << 63) - magnitude if bits >> 63 else (1 << 63) + magnitude

    return abs(key(left) - key(right))


def compare(left, right):
    validate_sample(left)
    validate_sample(right)
    output = []
    for a, b in zip(left["scientific"]["stages"], right["scientific"]["stages"], strict=True):
        distances = [ulps(x, y) for x, y in zip(a["values"], b["values"], strict=True)]
        output.append(
            {
                "name": a["name"],
                "count": len(distances),
                "strict_equal": a["sha256_be"] == b["sha256_be"],
                "changed_values": sum(
                    struct.pack(">d", x) != struct.pack(">d", y)
                    for x, y in zip(a["values"], b["values"], strict=True)
                ),
                "max_ulps": max(distances),
                "over_budget": sum(x > MAX_ULPS for x in distances),
                "max_abs_diff": max(
                    abs(x - y) for x, y in zip(a["values"], b["values"], strict=True)
                ),
            }
        )
    return {
        "strict_equal": left["scientific_sha256"] == right["scientific_sha256"],
        "within_8_ulps": all(x["over_budget"] == 0 for x in output),
        "stages": output,
        "first_unequal_forward_boundary": next(
            (x["name"] for x in output[:4] if not x["strict_equal"]), None
        ),
        "first_unequal_backward_boundary": next(
            (x["name"] for x in output[4:7] if not x["strict_equal"]), None
        ),
    }


def normalized_runtime(details):
    value = parse(canonical(details))
    value.pop("capability")
    value["environment"].pop("ATEN_CPU_CAPABILITY")
    return value


def summarize(samples, order):
    checks = {
        "all_four_workers_completed": len(samples) == len(order),
        "requested_capability_matches": all(
            s["runtime"]["environment"]["ATEN_CPU_CAPABILITY"]
            == (None if label == "A" else "default")
            for s, label in zip(samples, order, strict=False)
        ),
        "nonintervention_controls_equal": bool(samples)
        and all(
            normalized_runtime(s["runtime"]) == normalized_runtime(samples[0]["runtime"])
            for s in samples
        ),
        "effective_intervention": len(samples) == 4
        and samples[0]["runtime"]["capability"] == samples[3]["runtime"]["capability"]
        and samples[1]["runtime"]["capability"] == samples[2]["runtime"]["capability"] == "DEFAULT"
        and samples[0]["runtime"]["capability"] != "DEFAULT",
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


def capture(timeout=60, runner=subprocess.run):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("timeout must be finite and in (0,60]")
    load_fixture()
    order, samples, failure = ["A", "B", "B", "A"], [], None
    base = dict(os.environ, **{name: "1" for name in THREAD_ENV})
    base.pop("ATEN_CPU_CAPABILITY", None)
    started = datetime.now(UTC).isoformat()
    start = time.monotonic()
    for label in order:
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
            }
            break
    return {
        "plan_id": PLAN_ID,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - start,
        "order": order,
        "samples": samples,
        "failure": failure,
        **summarize(samples, order),
        "meaning": "capture_valid is control integrity only; numerical outcomes are independent",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    try:
        if args.freeze:
            value = freeze()
        elif args.worker:
            value = worker()
        else:
            if args.output is None or args.output.exists() or args.output.is_symlink():
                raise ValueError("provide a new output path")
            if not args.output.parent.is_dir():
                raise ValueError("output parent must exist")
            value = capture(args.timeout)
            write_new(args.output, value)
            print(
                json.dumps(
                    {
                        "capture_valid": value["capture_valid"],
                        "samples": len(value["samples"]),
                    }
                )
            )
            return 0 if value["capture_valid"] else 1
        print(json.dumps(value, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, OverflowError, TypeError) as error:
        print(f"probe failed: {type(error).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
