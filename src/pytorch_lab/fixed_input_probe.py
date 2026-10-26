"""Bounded fixed-logits dispatch diagnostics; no training or ground-truth claim."""

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

from pytorch_lab.numerical_probe import THREAD_ENV, _without_dispatch, runtime_details
from pytorch_lab.runtime_probe import package_file_hashes, result_digest
from pytorch_lab.scientific_compare import (
    MAX_ULPS,
    _reject_constant,
    _scan_json,
    _unique_object,
    ulp_distance,
)

PLAN_ID = "fixed-logits-dispatch-v1"
FIXTURE_SHA256 = "ea3b9d24cb7466c19fced9c85413f53137238fb36e5d4268d20e81686d299f94"
STAGES = {
    "identity": ([16, 3], "float64"),
    "row_max": ([16, 1], "float64"),
    "shifted": ([16, 3], "float64"),
    "exp": ([16, 3], "float64"),
    "denominator": ([16, 1], "float64"),
    "log_denominator": ([16, 1], "float64"),
    "decomposed_softmax": ([16, 3], "float64"),
    "decomposed_log_softmax": ([16, 3], "float64"),
    "softmax": ([16, 3], "float64"),
    "log_softmax": ([16, 3], "float64"),
    "cross_entropy_none": ([16], "float64"),
    "cross_entropy_mean": ([], "float64"),
    "logits_gradient": ([16, 3], "float64"),
    "predictions": ([16], "int64"),
}
EXPLICIT_CHAIN = ("identity", "row_max", "shifted", "exp", "denominator", "decomposed_softmax")
CONTROL_NAMES = (
    "cloned_input_bits_identical",
    "equal_logit_softmax_symmetric",
    "equal_logit_predictions_first_class",
)


def load_fixture() -> dict:
    """Check literal bytes before parsing; the fixture never depends on torch dispatch."""
    payload = files("pytorch_lab").joinpath("data/fixed_logits_v1.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != FIXTURE_SHA256:
        raise ValueError("fixed input bytes changed")
    return json.loads(payload)


def _bits(values: list, dtype: str) -> str:
    # Explicit network byte order; no NumPy, tensor pickle, native byte order or padding.
    code = "d" if dtype == "float64" else "q"
    return hashlib.sha256(b"".join(struct.pack(">" + code, v) for v in values)).hexdigest()


def _record(name: str, shape: list, dtype: str, values: list) -> dict:
    return {
        "name": name,
        "shape": shape,
        "dtype": dtype,
        "values": values,
        "sha256": _bits(values, dtype),
    }


def _tensor_record(name: str, tensor) -> dict:
    return _record(
        name,
        list(tensor.shape),
        str(tensor.dtype).removeprefix("torch."),
        tensor.detach().reshape(-1).tolist(),
    )


def expected_inputs() -> dict:
    fixture = load_fixture()
    return {
        "logits": _record("logits", [16, 3], "float64", sum(fixture["logits"], [])),
        "labels": _record("labels", [16], "int64", fixture["labels"]),
    }


def evaluate_fixture() -> dict:
    """Record alternate explicit intermediates and independent torch API outputs."""
    import torch
    from torch.nn import functional as F

    fixture = load_fixture()
    logits = torch.tensor(fixture["logits"], dtype=torch.float64)
    labels = torch.tensor(fixture["labels"], dtype=torch.int64)
    if not logits.is_contiguous() or logits.stride() != (3, 1):
        raise ValueError("unexpected fixed input layout")
    identity = logits.clone()
    maximum = logits.max(dim=1, keepdim=True).values
    shifted = logits - maximum
    exponential = shifted.exp()
    denominator = exponential.sum(dim=1, keepdim=True)
    log_denominator = denominator.log()
    probabilities = logits.softmax(dim=1)
    log_probabilities = logits.log_softmax(dim=1)
    differentiable = logits.clone().requires_grad_(True)
    loss = F.cross_entropy(differentiable, labels)
    (gradient,) = torch.autograd.grad(loss, differentiable)
    tensors = (
        identity,
        maximum,
        shifted,
        exponential,
        denominator,
        log_denominator,
        exponential / denominator,
        shifted - log_denominator,
        probabilities,
        log_probabilities,
        F.cross_entropy(logits, labels, reduction="none"),
        loss,
        gradient,
        logits.argmax(dim=1),
    )
    stages = [_tensor_record(name, t) for name, t in zip(STAGES, tensors, strict=True)]
    inputs = {
        "logits": _tensor_record("logits", logits),
        "labels": _tensor_record("labels", labels),
    }
    controls = {
        "cloned_input_bits_identical": stages[0]["sha256"] == inputs["logits"]["sha256"],
        "equal_logit_softmax_symmetric": all(
            len({_bits([v], "float64") for v in probabilities[i].tolist()}) == 1 for i in (1, 9)
        ),
        "equal_logit_predictions_first_class": all(int(tensors[-1][i]) == 0 for i in (1, 9)),
    }
    result = {
        "schema_version": 1,
        "plan_id": PLAN_ID,
        "fixture_sha256": FIXTURE_SHA256,
        "inputs": inputs,
        "stages": stages,
        "controls": controls,
    }
    validate_scientific(result)
    return result


def _validate_tensor(value: dict, name: str, shape: list, dtype: str) -> None:
    if type(value) is not dict or set(value) != {"name", "shape", "dtype", "values", "sha256"}:
        raise ValueError("invalid tensor record")
    if (
        value["name"] != name
        or value["dtype"] != dtype
        or type(value["shape"]) is not list
        or value["shape"] != shape
        or any(type(x) is not int for x in value["shape"])
        or type(value["values"]) is not list
        or len(value["values"]) != math.prod(shape)
    ):
        raise ValueError("unexpected tensor name, dtype, shape or size")
    kind = float if dtype == "float64" else int
    for v in value["values"]:
        if type(v) is not kind or (kind is float and (not math.isfinite(v) or abs(v) > 1e100)):
            raise ValueError("invalid bounded finite tensor value")
        if kind is int and v not in (0, 1, 2):
            raise ValueError("invalid class index")
    if value["sha256"] != _bits(value["values"], dtype):
        raise ValueError("tensor bytes do not match digest")


def validate_scientific(value: dict) -> None:
    """Accept only this fixed fixture/schema; validate evidence, not every metric's truth."""
    if type(value) is not dict or set(value) != {
        "schema_version",
        "plan_id",
        "fixture_sha256",
        "inputs",
        "stages",
        "controls",
    }:
        raise ValueError("unexpected fixed-input schema")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["plan_id"] != PLAN_ID
        or value["fixture_sha256"] != FIXTURE_SHA256
        or type(value["inputs"]) is not dict
        or set(value["inputs"]) != {"logits", "labels"}
        or type(value["stages"]) is not list
        or len(value["stages"]) != len(STAGES)
        or type(value["controls"]) is not dict
        or set(value["controls"]) != set(CONTROL_NAMES)
        or any(type(v) is not bool for v in value["controls"].values())
    ):
        raise ValueError("invalid fixed-input identity or controls")
    for name, expected in expected_inputs().items():
        actual = value["inputs"][name]
        _validate_tensor(actual, name, expected["shape"], expected["dtype"])
        if actual["sha256"] != expected["sha256"]:
            raise ValueError("fixed input tensor changed")
    for item, (name, (shape, dtype)) in zip(value["stages"], STAGES.items(), strict=True):
        _validate_tensor(item, name, shape, dtype)
    if value["stages"][0]["sha256"] != value["inputs"]["logits"]["sha256"]:
        raise ValueError("clone identity changed")
    probabilities = value["stages"][list(STAGES).index("softmax")]["values"]
    predictions = value["stages"][-1]["values"]
    recomputed = {
        "cloned_input_bits_identical": True,
        "equal_logit_softmax_symmetric": all(
            len({_bits([v], "float64") for v in probabilities[3 * i : 3 * i + 3]}) == 1
            for i in (1, 9)
        ),
        "equal_logit_predictions_first_class": all(predictions[i] == 0 for i in (1, 9)),
    }
    if value["controls"] != recomputed or not all(recomputed.values()):
        raise ValueError("fixed-input controls failed")


def compare_scientific(left: dict, right: dict) -> dict:
    """Separate bitwise equality from the unchanged eight-step diagnostic budget."""
    validate_scientific(left)
    validate_scientific(right)
    stages = []
    for a, b in zip(left["stages"], right["stages"], strict=True):
        distances = [
            ulp_distance(x, y) if a["dtype"] == "float64" else int(x != y)
            for x, y in zip(a["values"], b["values"], strict=True)
        ]
        allowed = MAX_ULPS if a["dtype"] == "float64" else 0
        mismatches = [
            i
            for i, (x, y) in enumerate(zip(a["values"], b["values"], strict=True))
            if _bits([x], a["dtype"]) != _bits([y], a["dtype"])
        ]
        failures = [i for i, distance in enumerate(distances) if distance > allowed]
        stages.append(
            {
                "name": a["name"],
                "exact_passed": a["sha256"] == b["sha256"],
                "diagnostic_passed": not failures,
                "bit_difference_count": len(mismatches),
                "ulp_failure_count": len(failures) if a["dtype"] == "float64" else 0,
                "exact_integer_failure_count": len(failures) if a["dtype"] == "int64" else 0,
                "max_ulp_distance": max(distances) if a["dtype"] == "float64" else None,
                "max_absolute_difference": max(
                    abs(x - y) for x, y in zip(a["values"], b["values"], strict=True)
                ),
                "examples": [
                    {
                        "flat_index": i,
                        "left": a["values"][i],
                        "right": b["values"][i],
                        "ulp_distance": distances[i] if a["dtype"] == "float64" else None,
                    }
                    for i in mismatches[:8]
                ],
            }
        )
    first = next(
        (s["name"] for s in stages if s["name"] in EXPLICIT_CHAIN and not s["exact_passed"]), None
    )
    return {
        "policy": {
            "name": PLAN_ID,
            "max_ulps": MAX_ULPS,
            "absolute_tolerance": 0.0,
            "scope": "Float stage outputs only; inputs, structure and predictions exact.",
        },
        "exact_passed": result_digest(left) == result_digest(right),
        "diagnostic_passed": all(s["diagnostic_passed"] for s in stages),
        "left_scientific_sha256": result_digest(left),
        "right_scientific_sha256": result_digest(right),
        "first_unequal_explicit_chain_boundary": first,
        "boundary_scope": (
            "Recorded alternate softmax chain only, not fused kernel internals "
            "or historical CI attribution."
        ),
        "stages": stages,
    }


def capture_worker() -> dict:
    if sys.platform != "linux":
        raise RuntimeError("Linux only")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.version.cuda is not None:
        raise RuntimeError("CPU-only torch required")
    scientific = evaluate_fixture()
    return {
        "scientific": scientific,
        "scientific_sha256": result_digest(scientific),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "libc": list(platform.libc_ver()),
            "device": "cpu",
            "dtype": "float64",
            "torch_cuda_build": torch.version.cuda,
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "python_isolated_mode": bool(sys.flags.isolated),
        },
        "cpu_runtime": runtime_details(),
        "package_files_sha256": package_file_hashes(),
    }


def _validate_sample(sample: dict, condition: str) -> None:
    _scan_json(sample)
    if type(sample) is not dict or set(sample) != {
        "scientific",
        "scientific_sha256",
        "environment",
        "cpu_runtime",
        "package_files_sha256",
    }:
        raise ValueError("invalid worker envelope")
    env = sample["environment"]
    if (
        type(env) is not dict
        or set(env)
        != {
            "python",
            "torch",
            "libc",
            "device",
            "dtype",
            "torch_cuda_build",
            "intraop_threads",
            "interop_threads",
            "python_isolated_mode",
        }
        or any(type(env[key]) is not str or not env[key] for key in ("python", "torch"))
    ):
        raise ValueError("invalid runtime metadata")
    if (
        type(env["libc"]) is not list
        or len(env["libc"]) != 2
        or any(type(v) is not str for v in env["libc"])
    ):
        raise ValueError("invalid libc metadata")
    cpu = sample["cpu_runtime"]
    if type(cpu) is not dict or set(cpu) != {
        "native_build_versions",
        "cpu_models_and_features",
        "cpu_affinity",
        "kernel_release",
        "byteorder",
        "float_info",
        "torch_git_version",
        "torch_cpu_capability",
        "torch_build_config_sha256",
        "mkl_available",
        "mkldnn_available",
        "mkldnn_enabled",
        "openmp_available",
        "deterministic_algorithms",
        "environment_controls",
    }:
        raise ValueError("invalid CPU metadata")
    if cpu["torch_cpu_capability"] not in {"DEFAULT", "AVX2", "AVX512", "VSX", "Z VECTOR"}:
        raise ValueError("invalid observed capability")
    if type(cpu["environment_controls"]) is not dict:
        raise ValueError("invalid environment controls")
    if type(sample["package_files_sha256"]) is not dict:
        raise ValueError("invalid package provenance")
    validate_scientific(sample["scientific"])
    if result_digest(sample["scientific"]) != sample["scientific_sha256"]:
        raise ValueError("scientific digest mismatch")
    env = sample["environment"]
    if any(
        env.get(k) != v or type(env.get(k)) is not type(v)
        for k, v in {
            "device": "cpu",
            "dtype": "float64",
            "torch_cuda_build": None,
            "intraop_threads": 1,
            "interop_threads": 1,
            "python_isolated_mode": True,
        }.items()
    ):
        raise ValueError("unexpected worker environment")
    if sample["package_files_sha256"] != package_file_hashes():
        raise ValueError("worker package files differ from parent")
    controls = sample["cpu_runtime"]["environment_controls"]
    if controls["ATEN_CPU_CAPABILITY"] != (None if condition == "A" else "default"):
        raise ValueError("dispatch request mismatch")
    if any(controls[name] != "1" for name in THREAD_ENV):
        raise ValueError("thread request mismatch")


def validate_config(repeats: int, timeout: float) -> None:
    if type(repeats) is not int or not 2 <= repeats <= 4:
        raise ValueError("repeats must be an integer in [2, 4]")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError("timeout must be finite and in (0, 300]")


def run_capture(repeats: int = 2, timeout: float = 60, dispatch_study: bool = False) -> dict:
    validate_config(repeats, timeout)
    if type(dispatch_study) is not bool:
        raise ValueError("dispatch_study must be bool")
    if sys.platform != "linux":
        raise RuntimeError("Linux only")
    order = ["A", "B", "B", "A"] if dispatch_study else ["A"] * repeats
    base_env = dict(os.environ, **{name: "1" for name in THREAD_ENV})
    base_env.pop("ATEN_CPU_CAPABILITY", None)
    samples, error = [], None
    for label in order:
        env = dict(base_env)
        if label == "B":
            env["ATEN_CPU_CAPABILITY"] = "default"
        try:
            run = subprocess.run(
                [sys.executable, "-I", "-m", "pytorch_lab.fixed_input_probe", "--worker"],
                env=env,
                text=True,
                capture_output=True,
                check=True,
                timeout=timeout,
            )
            if len(run.stdout) > 1024 * 1024:
                raise ValueError("worker report exceeds limit")
            sample = json.loads(
                run.stdout, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
            _validate_sample(sample, label)
            sample["condition"] = label
            sample["stderr_nonempty"] = bool(run.stderr.strip())
            samples.append(sample)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
            subprocess.SubprocessError,
            RecursionError,
        ) as exc:
            error = {
                "sample_index": len(samples),
                "condition": label,
                "error_type": type(exc).__name__,
                "detail": (
                    "Worker failed; completed evidence retained, arbitrary child output omitted."
                ),
            }
            break
    complete = len(samples) == len(order) and error is None
    controlled = bool(samples) and all(
        s["environment"] == samples[0]["environment"]
        and s["package_files_sha256"] == samples[0]["package_files_sha256"]
        and _without_dispatch(s["cpu_runtime"]) == _without_dispatch(samples[0]["cpu_runtime"])
        for s in samples
    )
    repeated = complete and all(
        len({s["scientific_sha256"] for s in samples if s["condition"] == label}) == 1
        and all(
            s["cpu_runtime"] == next(t["cpu_runtime"] for t in samples if t["condition"] == label)
            for s in samples
            if s["condition"] == label
        )
        for label in set(order)
    )
    intervention = (
        None
        if not dispatch_study
        else complete
        and (
            samples[0]["cpu_runtime"]["torch_cpu_capability"]
            != samples[1]["cpu_runtime"]["torch_cpu_capability"]
            and samples[1]["cpu_runtime"]["torch_cpu_capability"] == "DEFAULT"
        )
    )
    checks = {
        "all_planned_captures_completed": complete,
        "nonintervention_controls_identical": controlled,
        "within_condition_strict_repeats": repeated,
        "dispatch_intervention_observed": intervention,
    }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "plan_id": PLAN_ID,
        "protocol": {
            "order": order,
            "timeout_seconds_per_child": timeout,
            "training_updates": 0,
            "fresh_isolated_processes": True,
            "fixture_sha256": FIXTURE_SHA256,
        },
        "samples": samples,
        "capture_error": error,
        "capture_checks": checks,
        "comparisons_to_first_sample": [
            compare_scientific(samples[0]["scientific"], s["scientific"]) for s in samples[1:]
        ],
        "all_checks_passed": complete and controlled and repeated and intervention is not False,
        "all_checks_scope": (
            "Capture and control validity only; strict-output and 8-ULP results remain separate."
        ),
        "limits": [
            "Synthetic fixed logits; no training, Wine test-set or parameter-path localization.",
            "The explicit decomposition is neither ground truth nor a fused-kernel trace.",
            "No historical CI cause, executed microkernel, general error bound or speed claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--dispatch-study", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        validate_config(args.repeats, args.timeout)
        if args.worker and args.dispatch_study:
            raise ValueError("worker does not orchestrate studies")
        if args.output is not None and (args.output.exists() or args.output.is_symlink()):
            raise FileExistsError("output exists")
        report = (
            capture_worker()
            if args.worker
            else run_capture(args.repeats, args.timeout, args.dispatch_study)
        )
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as target:
                target.write(encoded)
        print(encoded, end="")
        if not args.worker and not report["all_checks_passed"]:
            parser.exit(
                1, "Fixed-input capture/control checks failed; partial evidence retained.\n"
            )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError):
        parser.exit(2, "Fixed-input probe rejected: invalid configuration or unavailable output.\n")


if __name__ == "__main__":
    main()
