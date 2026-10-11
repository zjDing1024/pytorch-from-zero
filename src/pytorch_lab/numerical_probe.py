"""Retain full Wine outputs and inspect one predeclared ATen-dispatch contrast.

The original runtime CLI and its strict hash gate are unchanged. This companion
adds evidence; numeric regression and experiment-control status are separate.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from pytorch_lab.runtime_probe import _validate_measurement, _worker, result_digest, validate_config

CPU_FIELDS = {"vendor_id", "cpu family", "model", "model name", "stepping", "flags"}
ENUM_ENV = {
    "ATEN_CPU_CAPABILITY": {"default", "avx2", "avx512"},
    "MKL_CBWR": {"AUTO", "COMPATIBLE", "SSE4_2", "AVX", "AVX2", "AVX512"},
    "MKL_ENABLE_INSTRUCTIONS": {"SSE4_2", "AVX", "AVX2", "AVX512"},
    "ONEDNN_MAX_CPU_ISA": {"DEFAULT", "SSE41", "AVX", "AVX2", "AVX512_CORE"},
}
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def cpu_inventory(text: str) -> list[dict]:
    """Only public hardware characteristics, never serials or processor IDs."""
    records = []
    for block in text.split("\n\n"):
        record = {}
        for line in block.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in CPU_FIELDS:
                value = value.strip()
                record[key.strip()] = sorted(value.split()) if key.strip() == "flags" else value
        if record and record not in records:
            records.append(record)
    return sorted(records, key=lambda x: json.dumps(x, sort_keys=True))


def environment_controls() -> dict:
    """Never publish arbitrary environment values or a full environment dump."""
    values = {}
    for name, allowed in ENUM_ENV.items():
        raw = os.environ.get(name)
        values[name] = raw if raw is None or raw in allowed else "unrecognized (redacted)"
    for name in (*THREAD_ENV, "PYTHONHASHSEED"):
        raw = os.environ.get(name)
        valid = raw is None or re.fullmatch(r"[0-9]{1,10}", raw) is not None
        values[name] = raw if valid else "unrecognized (redacted)"
    return values


def runtime_details() -> dict:
    import torch

    try:
        cpus = cpu_inventory(Path("/proc/cpuinfo").read_text())
    except OSError:
        cpus = []
    # Hash build configuration rather than copying its embedded compiler paths.
    config = torch.__config__.show()
    build_lines = [line for line in config.splitlines() if "CPU capability usage:" not in line]
    native_versions = {}
    for name, expression in {
        "gcc": r"GCC ([0-9.]+)",
        "cxx_standard": r"C\+\+ Version: ([0-9]+)",
        "mkl": r"Math Kernel Library Version ([0-9.]+)",
        "mkldnn": r"MKL-DNN v([0-9.]+)",
        "openmp": r"OpenMP ([0-9]+)",
    }.items():
        match = re.search(expression, config)
        native_versions[name] = match.group(1) if match else None
    return {
        "native_build_versions": native_versions,
        "cpu_models_and_features": cpus,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "kernel_release": platform.release(),
        "byteorder": sys.byteorder,
        "float_info": {k: getattr(sys.float_info, k) for k in ("radix", "mant_dig", "epsilon")},
        "torch_git_version": torch.version.git_version,
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_build_config_sha256": hashlib.sha256("\n".join(build_lines).encode()).hexdigest(),
        "mkl_available": torch.backends.mkl.is_available(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "openmp_available": torch.backends.openmp.is_available(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment_controls": environment_controls(),
    }


def capture_worker(epochs: int) -> dict:
    sample = _worker(epochs, retain_scientific=True)
    sample["cpu_runtime"] = runtime_details()
    return sample


def _without_dispatch(details: dict) -> dict:
    value = json.loads(json.dumps(details))
    value.pop("torch_cpu_capability", None)
    value["environment_controls"].pop("ATEN_CPU_CAPABILITY", None)
    return value


def run_capture(
    epochs: int = 120, repeats: int = 2, timeout: float = 300, dispatch_study: bool = False
) -> dict:
    from pytorch_lab.scientific_compare import compare_results

    validate_config(epochs, repeats, timeout)
    if sys.platform != "linux":
        raise RuntimeError("Linux only")
    labels = ["A", "B", "B", "A"] if dispatch_study else ["A"] * repeats
    base_env = dict(os.environ, **{name: "1" for name in THREAD_ENV})
    # A is explicitly unset; don't inherit a hidden dispatch intervention.
    base_env.pop("ATEN_CPU_CAPABILITY", None)
    samples = []
    capture_error = None
    for label in labels:
        env = dict(base_env)
        if label == "B":
            env["ATEN_CPU_CAPABILITY"] = "default"
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-m",
                    "pytorch_lab.numerical_probe",
                    "--worker",
                    "--epochs",
                    str(epochs),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
                env=env,
            )
            sample = json.loads(completed.stdout)
            _validate_measurement(sample, epochs)
            requested = None if label == "A" else "default"
            if sample["cpu_runtime"]["environment_controls"]["ATEN_CPU_CAPABILITY"] != requested:
                raise ValueError("dispatch request metadata mismatch")
            if result_digest(sample["wine_result"]) != sample["scientific_result_sha256"]:
                raise ValueError("retained output digest mismatch")
            # Schema validation is also performed for the very first retained output.
            compare_results(sample["wine_result"], sample["wine_result"])
            sample["condition"] = label
            sample["child_end_to_end_wall_seconds"] = time.perf_counter() - start
            sample["stderr_nonempty"] = bool(completed.stderr.strip())
            samples.append(sample)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as error:
            capture_error = {
                "sample_index": len(samples),
                "condition": label,
                "error_type": type(error).__name__,
                "detail": "Worker failed; completed captures retained, arbitrary output omitted.",
            }
            break
    comparisons = [
        compare_results(samples[0]["wine_result"], s["wine_result"]) for s in samples[1:]
    ]
    complete = len(samples) == len(labels) and capture_error is None
    controlled = bool(samples) and all(
        s["environment"] == samples[0]["environment"]
        and s["package_files_sha256"] == samples[0]["package_files_sha256"]
        and _without_dispatch(s["cpu_runtime"]) == _without_dispatch(samples[0]["cpu_runtime"])
        for s in samples
    )
    same_within_condition = all(
        len({s["scientific_result_sha256"] for s in samples if s["condition"] == label}) == 1
        for label in set(labels)
    )
    same_runtime_within_condition = all(
        all(
            s["cpu_runtime"] == next(t["cpu_runtime"] for t in samples if t["condition"] == label)
            for s in samples
            if s["condition"] == label
        )
        for label in set(labels)
    )
    dispatch_changed = None
    if dispatch_study:
        dispatch_changed = len(samples) >= 2 and (
            samples[0]["cpu_runtime"]["torch_cpu_capability"]
            != samples[1]["cpu_runtime"]["torch_cpu_capability"]
            and samples[1]["cpu_runtime"]["torch_cpu_capability"] == "DEFAULT"
        )
    checks = {
        "all_planned_captures_completed": complete,
        "nonintervention_controls_identical": controlled,
        "strict_scientific_hash_equal_within_each_condition": same_within_condition,
        "runtime_identical_within_each_condition": same_runtime_within_condition,
        "dispatch_intervention_observed": dispatch_changed,
    }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "plan_id": "wine-float64-lastbits-v1",
        "protocol": {
            "epochs": epochs,
            "order": labels,
            "fresh_isolated_processes": True,
            "A": "ATEN_CPU_CAPABILITY unset",
            "B": "ATEN_CPU_CAPABILITY=default",
            "timeout_seconds_per_child": timeout,
        },
        "samples": samples,
        "capture_error": capture_error,
        "comparisons_to_first_sample": comparisons,
        "capture_checks": checks,
        "all_checks_passed": (
            complete
            and controlled
            and same_within_condition
            and same_runtime_within_condition
            and dispatch_changed is not False
        ),
        "all_checks_scope": "Capture/control integrity and exact within-condition repeats only. "
        "Cross-condition numeric/hash outcomes are separately reported, not "
        "silently converted into success.",
        "limits": [
            "A dispatch contrast is not an attribution for earlier CI runs.",
            "ATen dispatch does not control every MKL or oneDNN code path.",
            "Shared-host timing is descriptive; no speed or general error-bound claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--dispatch-study", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        validate_config(args.epochs, args.repeats, args.timeout)
        if args.output is not None and (args.output.exists() or args.output.is_symlink()):
            raise FileExistsError("output exists")
        if args.worker and args.dispatch_study:
            raise ValueError("worker does not orchestrate studies")
        result = (
            capture_worker(args.epochs)
            if args.worker
            else run_capture(args.epochs, args.repeats, args.timeout, args.dispatch_study)
        )
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as f:
                f.write(encoded)
        print(encoded, end="")
        if not args.worker and result["capture_error"] is not None:
            parser.exit(2, "Worker failed; partial capture evidence retained.\n")
        if not args.worker and not result["all_checks_passed"]:
            parser.exit(1, "Capture control checks failed; evidence retained.\n")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError):
        parser.exit(2, "Numerical capture failed; check CPU installation/configuration/timeout.\n")


if __name__ == "__main__":
    main()
