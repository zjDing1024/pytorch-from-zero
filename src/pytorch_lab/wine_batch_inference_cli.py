"""File-to-file partial batch inference, or a bounded CPU latency experiment."""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import torch

from pytorch_lab.wine_batch_inference import WineBatchPredictor
from pytorch_lab.wine_model_bundle import predict, strict_json, train_bundle, write_new

BATCH_SIZES = (1, 16, 107, 1024)
REPEATS = 3
WARMUPS = 1


def serve(bundle_path, request_path, output_path):
    start = perf_counter()
    bundle = strict_json(bundle_path)
    parsed_bundle = perf_counter()
    engine = WineBatchPredictor(bundle)
    loaded = perf_counter()
    request = strict_json(request_path)
    parsed_request = perf_counter()
    response = engine.predict(request)
    predicted = perf_counter()
    # This diagnostic ends before encoding/writing itself, and says so in the key.
    response["file_timings_seconds"] = {
        "bundle_read_parse": parsed_bundle - start,
        "bundle_validate_load": loaded - parsed_bundle,
        "request_read_parse": parsed_request - loaded,
        "prediction": predicted - parsed_request,
        "before_output_encoding": predicted - start,
    }
    write_new(output_path, response)
    return response


def _distribution(values):
    return {
        "samples": values,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def verify_and_measure():
    # Fixture generation is outside every latency boundary, and its work is disclosed.
    bundle, source, _, work = train_bundle()
    engine = WineBatchPredictor(bundle)
    runs, checks = [], []
    with tempfile.TemporaryDirectory(prefix="wine-batch-") as directory:
        root = Path(directory)
        write_new(root / "bundle.json", bundle)
        for size in BATCH_SIZES:
            request = {
                "features": source["features"],
                "rows": [source["rows"][i % 107] for i in range(size)],
            }
            reference = predict(bundle, request)
            request_path = root / f"request-{size}.json"
            write_new(request_path, request)
            warm, cold, stages = [], [], []
            for repeat in range(WARMUPS + REPEATS):
                response = engine.predict(request)
                output = root / f"response-{size}-{repeat}.json"
                start = perf_counter()
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytorch_lab.wine_batch_inference_cli",
                        "--bundle",
                        str(root / "bundle.json"),
                        "--request",
                        str(request_path),
                        "--output",
                        str(output),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
                elapsed = perf_counter() - start
                fresh = strict_json(output)
                for candidate in (response, fresh):
                    actual = [item["prediction"]["logits"] for item in candidate["items"]]
                    checks.append(
                        bool(
                            torch.allclose(
                                torch.tensor(actual, dtype=torch.float64),
                                torch.tensor(reference["logits"], dtype=torch.float64),
                                rtol=1e-12,
                                atol=1e-12,
                            )
                        )
                    )
                    checks.append(
                        [item["prediction"]["class_index"] for item in candidate["items"]]
                        == reference["class_indices"]
                    )
                if repeat >= WARMUPS:
                    warm.append(response["timings_seconds"])
                    cold.append(elapsed)
                    stages.append(fresh["file_timings_seconds"])
            runs.append(
                {
                    "rows": size,
                    "input_bytes": request_path.stat().st_size,
                    "output_bytes": output.stat().st_size,
                    "warm_seconds": {
                        key: _distribution([row[key] for row in warm]) for key in warm[0]
                    },
                    "fresh_process_file_to_file_seconds": _distribution(cold),
                    "fresh_process_inside_seconds": stages,
                }
            )
    mixed = engine.predict(
        {
            "features": source["features"],
            "rows": [source["rows"][0], [True] * 13, [], source["rows"][1]],
        }
    )
    checks.extend(
        [
            mixed["summary"] == {"requested": 4, "succeeded": 2, "failed": 2},
            [item["index"] for item in mixed["items"]] == [0, 1, 2, 3],
            mixed["items"][1]["error"]["code"] == "invalid_number",
            mixed["items"][2]["error"]["code"] == "invalid_shape",
        ]
    )
    return {
        "protocol": "wine-batch-inference-v1",
        "fixture_training_work": work,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.system(),
            "machine": platform.machine(),
            "cpu_threads": torch.get_num_threads(),
            "dtype": "float64",
        },
        "warmups": WARMUPS,
        "repeats": REPEATS,
        "runs": runs,
        "mixed_example": mixed,
        "checks_passed": sum(checks),
        "checks_total": len(checks),
        "all_checks_passed": all(checks),
        "scope": (
            "CPU local latency, not throughput/load/SLA. Fresh-process wall time includes "
            "interpreter, imports, read/parse, bundle load, inference, encode/write and exit; "
            "no fsync durability guarantee. Warm timings exclude model load and JSON IO. "
            "No new accuracy scoring or monetary extrapolation."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    if bool(args.bundle) != bool(args.request):
        parser.error("bundle and request must be supplied together")
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if args.bundle:
            serve(args.bundle, args.request, args.output)
            return 0
        report = verify_and_measure()
        write_new(args.output, report)
        print(json.dumps(report, allow_nan=False))
        return 0 if report["all_checks_passed"] else 1
    except (OSError, ValueError, subprocess.SubprocessError):
        parser.exit(2, "Batch inference failed; no success report claimed.\n")
    finally:
        torch.set_num_threads(old)


if __name__ == "__main__":
    raise SystemExit(main())
