"""Write bounded fixed-state Wine gradient sampling evidence without replacing existing evidence."""

import argparse
import json
from pathlib import Path

from pytorch_lab.wine_gradient_sampling import run_experiment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new path")
    report = run_experiment()
    # Validate the complete serialization before creating directories or the evidence file.
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(encoded, end="")
    return 0 if report.get("all_checks_passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
