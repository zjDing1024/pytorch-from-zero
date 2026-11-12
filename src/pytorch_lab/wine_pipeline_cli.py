"""Run the frozen train -> validation -> export -> fresh-process prediction path."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from pytorch_lab import wine
from pytorch_lab.wine_model_bundle import load_payload, strict_json, train_bundle, write_new


def run_pipeline(output: Path) -> dict:
    """One training run; validation is descriptive, never an export/selection gate."""
    output = Path(output)
    artifacts = output.with_name(output.stem + "-artifacts")
    if output.exists() or output.is_symlink():
        raise FileExistsError("report already exists")
    # Exclusive directory creation also refuses old/partial runs and dangling links.
    artifacts.mkdir(parents=True, exist_ok=False)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        bundle, _, _, work = train_bundle()
        training = {"config": bundle["payload"]["config"], "work": work}
        write_new(artifacts / "training.json", training)

        # Score only the existing validation split, with the very same trained state.
        # The CSV parser verifies all source bytes; this is not a data-access claim.
        x, y = wine.load_wine()
        split = wine.build_split(y)
        wine.audit_split(x, y, split)
        ids = split["validation"]
        model, scaler = load_payload(bundle)
        with torch.no_grad():
            logits = model(scaler.transform(x[ids]))
        evaluation = {
            "split": "reused_validation",
            "model_sha256": bundle["sha256"],
            "metrics": wine.metrics(logits, y[ids], ids),
            "test_rows_scored": 0,
            "used_for_selection": False,
        }
        write_new(artifacts / "evaluation.json", evaluation)

        # Export is the existing bounded inference JSON, not a training checkpoint.
        write_new(artifacts / "model.json", bundle)
        request = {"features": list(wine.FEATURE_NAMES), "rows": x[ids].tolist()}
        write_new(artifacts / "request.json", request)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytorch_lab.wine_batch_inference_cli",
                "--bundle",
                str((artifacts / "model.json").resolve()),
                "--request",
                str((artifacts / "request.json").resolve()),
                "--output",
                str((artifacts / "predictions.json").resolve()),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        predictions = strict_json(artifacts / "predictions.json")
        items = predictions["items"]
        actual = torch.tensor([item["prediction"]["logits"] for item in items], dtype=torch.float64)
        checks = {
            "same_exported_model": strict_json(artifacts / "model.json") == bundle,
            "all_rows_predicted": predictions["summary"]
            == {"requested": len(ids), "succeeded": len(ids), "failed": 0},
            "row_order_preserved": [item["index"] for item in items] == list(range(len(ids))),
            "fresh_process_logits_match": actual.shape == logits.shape
            and bool(torch.allclose(actual, logits, rtol=1e-12, atol=1e-12)),
            "fresh_process_classes_match": [item["prediction"]["class_index"] for item in items]
            == evaluation["metrics"]["predictions"],
        }
        report = {
            "protocol": "wine-end-to-end-v1",
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.system(),
                "machine": platform.machine(),
                "cpu_threads": torch.get_num_threads(),
                "dtype": "float64",
            },
            "training": training,
            "evaluation": evaluation,
            "artifacts": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(artifacts.iterdir())
            },
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "scope": (
                "Integration of existing fixed training and inference components; "
                "reused validation is descriptive, not fresh generalization evidence. "
                "No tuning, test scoring, service deployment or performance claim. "
                "Checks validate artifact/prediction agreement, not accuracy thresholds."
            ),
        }
        write_new(output, report)
        return report
    finally:
        torch.set_num_threads(old_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_pipeline(args.output)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        parser.exit(2, "Pipeline failed; partial artifacts may remain. Choose a new output path.\n")
    print(json.dumps(report, allow_nan=False))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
