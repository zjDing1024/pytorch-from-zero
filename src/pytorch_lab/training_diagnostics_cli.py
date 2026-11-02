"""Run the fixed-batch training observation experiment and retain JSON evidence."""

import argparse
import json
from pathlib import Path

from pytorch_lab.training_diagnostics import run_experiment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new path")
    result = run_experiment()
    # Encode before opening the target, so invalid numerical evidence leaves no file.
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    print(encoded, end="")
    return 0 if result.get("all_checks_passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
