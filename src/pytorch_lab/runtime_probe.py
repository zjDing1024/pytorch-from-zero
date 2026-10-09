"""Measure a fixed CPU Wine workload in fresh, isolated Python processes.

Linux only: ru_maxrss is KiB and is a process lifetime high-water mark.
This is a resource observation, not a portable performance benchmark.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

RUNTIME_DISTRIBUTIONS = (
    "torch",
    "filelock",
    "typing_extensions",
    "setuptools",
    "sympy",
    "networkx",
    "jinja2",
    "fsspec",
    "mpmath",
    "markupsafe",
)


def validate_config(epochs: int, repeats: int, timeout: float) -> None:
    if type(epochs) is not int or not 1 <= epochs <= 120:
        raise ValueError("epochs must be an integer in [1, 120]")
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError("repeats must be an integer in [1, 10]")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be finite and in (0, 3600]")
    if not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError("timeout must be finite and in (0, 3600]")


def result_digest(result: dict) -> str:
    """Hash scientific output, excluding only timestamp and runtime metadata."""
    scientific = {k: v for k, v in result.items() if k not in {"created_at_utc", "environment"}}
    payload = json.dumps(scientific, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def package_file_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and (p.suffix in {".py", ".csv", ".json"} or p.name.startswith("LICENSE."))
    }


def _worker(epochs: int) -> dict:
    if sys.platform != "linux":
        raise RuntimeError("resource measurement currently supports Linux only")
    import resource

    import torch

    from pytorch_lab.wine import WineConfig, run_wine_experiment

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.version.cuda is not None:
        raise RuntimeError("this protocol requires a CPU-only torch build")
    # Imports and thread setup are outside this interval; parent wall time includes both.
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    result = run_wine_experiment(WineConfig(epochs=epochs))
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if not result["all_checks_passed"]:
        raise RuntimeError("Wine protocol checks failed")
    return {
        "workload_wall_seconds": wall_seconds,
        "workload_process_cpu_seconds": cpu_seconds,
        "process_high_water_rss_kib": rss_kib,
        "process_high_water_rss_mib": rss_kib / 1024,
        "scientific_result_sha256": result_digest(result),
        "workload": {
            "name": "fixed Wine classification protocol",
            "epochs_per_fit": epochs,
            "fits": 6,
            "updates_total": sum(
                r["updates"] for m in result["models"].values() for r in m["runs"]
            ),
            "samples_seen_total": sum(
                r["samples_seen"] for m in result["models"].values() for r in m["runs"]
            ),
            "all_checks_passed": result["all_checks_passed"],
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "device": "cpu",
            "dtype": "float64",
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "logical_cpus": os.cpu_count(),
            "affinity_cpus": len(os.sched_getaffinity(0)),
            "python_isolated_mode": bool(sys.flags.isolated),
            "distributions": {n: importlib.metadata.version(n) for n in RUNTIME_DISTRIBUTIONS},
        },
        "package_files_sha256": package_file_hashes(),
    }


def _validate_measurement(value: dict, epochs: int) -> None:
    """Reject malformed evidence before using it in a successful report."""
    for key in (
        "workload_wall_seconds",
        "workload_process_cpu_seconds",
        "process_high_water_rss_kib",
    ):
        x = value[key]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
            raise ValueError(f"invalid measurement: {key}")
    if value["process_high_water_rss_mib"] != value["process_high_water_rss_kib"] / 1024:
        raise ValueError("invalid RSS unit conversion")
    workload = value["workload"]
    if (
        workload["epochs_per_fit"] != epochs
        or workload["fits"] != 6
        or workload["updates_total"] != 6 * 7 * epochs
        or workload["samples_seen_total"] != 6 * 107 * epochs
        or workload["all_checks_passed"] is not True
    ):
        raise ValueError("unexpected workload or failed checks")
    environment = value["environment"]
    if (
        environment["device"] != "cpu"
        or environment["torch_cuda_build"] is not None
        or environment["intraop_threads"] != 1
        or environment["interop_threads"] != 1
        or environment["python_isolated_mode"] is not True
    ):
        raise ValueError("unexpected runtime controls")
    digest = value["scientific_result_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("invalid result digest")
    if not value["package_files_sha256"]:
        raise ValueError("missing installed-package provenance")


def run_probe(epochs: int = 120, repeats: int = 3, timeout: float = 300) -> dict:
    validate_config(epochs, repeats, timeout)
    if sys.platform != "linux":
        raise RuntimeError("resource measurement currently supports Linux only")
    samples = []
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for _ in range(repeats):
        start = time.perf_counter()
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "pytorch_lab.runtime_probe",
                "--worker",
                "--epochs",
                str(epochs),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
            env=env,
        )
        child_wall = time.perf_counter() - start
        value = json.loads(completed.stdout)
        _validate_measurement(value, epochs)
        value["child_end_to_end_wall_seconds"] = child_wall
        # Warnings may contain machine paths; record presence, not arbitrary stderr text.
        value["stderr_nonempty"] = bool(completed.stderr.strip())
        samples.append(value)
    same_output = len({x["scientific_result_sha256"] for x in samples}) == 1
    same_runtime = all(x["environment"] == samples[0]["environment"] for x in samples)
    same_package = all(
        x["package_files_sha256"] == samples[0]["package_files_sha256"] for x in samples
    )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "epochs": epochs,
            "repeats": repeats,
            "timeout_seconds_per_child": timeout,
            "fresh_process_per_sample": True,
            "warmup_runs": 0,
            "measurement_scope": (
                "Wine load, preprocessing, six fits and evaluation; "
                "excludes initial imports/thread setup"
            ),
            "end_to_end_scope": (
                "child start through exit; imports and JSON serialization included"
            ),
            "rss_scope": ("Linux process lifetime high-water RSS; includes imports, not a delta"),
            "comparison": "same fixed workload and code; no hardware or speedup claim",
        },
        "samples": samples,
        "summary": {
            key: {
                "min": min(x[key] for x in samples),
                "median": statistics.median(x[key] for x in samples),
                "max": max(x[key] for x in samples),
            }
            for key in (
                "workload_wall_seconds",
                "workload_process_cpu_seconds",
                "process_high_water_rss_mib",
                "child_end_to_end_wall_seconds",
            )
        },
        "checks": {
            "scientific_results_identical": same_output,
            "runtime_identical": same_runtime,
            "package_files_identical": same_package,
        },
        "all_checks_passed": same_output and same_runtime and same_package,
        "limits": [
            "Fresh interpreter does not imply cold filesystem or CPU caches.",
            "Shared-host scheduling and resource limits can affect observations.",
            "RSS includes interpreter, libraries and allocator; not tensor or container memory.",
            "No GPU, energy, FLOP, significant speedup or cross-platform identity claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--output", type=Path, help="New JSON file; existing paths are never overwritten"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        validate_config(args.epochs, args.repeats, args.timeout)
        if args.output is not None and (args.output.exists() or args.output.is_symlink()):
            raise FileExistsError("output already exists; choose a new path")
        if args.worker:
            result = _worker(args.epochs)
        else:
            result = run_probe(args.epochs, args.repeats, args.timeout)
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                output.write(encoded)
        print(encoded, end="")
        if not args.worker and not result["all_checks_passed"]:
            parser.exit(1, "Runtime reproducibility checks failed.\n")
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError):
        # No arbitrary child output or local paths are copied into public evidence.
        parser.exit(
            2, "Runtime probe failed; check configuration, CPU install, platform and timeout.\n"
        )


if __name__ == "__main__":
    main()
