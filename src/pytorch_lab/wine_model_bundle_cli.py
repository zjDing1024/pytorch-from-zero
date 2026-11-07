"""Export and independently reload a strict JSON Wine inference artifact."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from pytorch_lab.wine_model_bundle import predict, strict_json, train_bundle, write_new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    if bool(args.bundle) != bool(args.request):
        parser.error("bundle and request must be supplied together")
    if args.bundle:
        result = predict(strict_json(args.bundle), strict_json(args.request))
        write_new(args.output, result)
        return 0
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        bundle, request, expected, work = train_bundle()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_new(root / "model.json", bundle)
            write_new(root / "request.json", request)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytorch_lab.wine_model_bundle_cli",
                    "--bundle",
                    str(root / "model.json"),
                    "--request",
                    str(root / "request.json"),
                    "--output",
                    str(root / "prediction.json"),
                ],
                check=True,
                timeout=60,
            )
            result = strict_json(root / "prediction.json")
        reloaded = predict(bundle, request)
        checks = {
            "fresh_process_logits_equal": result["logits"] == expected,
            "all_predictions_equal": result == reloaded,
        }
        report = {
            "protocol": "wine-model-bundle-v1",
            "work": work,
            "prediction_rows": len(expected),
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "bundle": bundle,
            "predictions": result,
        }
        write_new(args.output, report)
        print(json.dumps(report, allow_nan=False))
        return 0 if report["all_checks_passed"] else 1
    finally:
        torch.set_num_threads(threads)


if __name__ == "__main__":
    raise SystemExit(main())
