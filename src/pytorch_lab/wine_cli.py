"""Run the pinned offline Wine classification protocol and preserve original evidence."""

import argparse
import json
from pathlib import Path

from pytorch_lab.checkpoint import _atomic_write
from pytorch_lab.wine import WineConfig, run_wine_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120, help="120 for the recorded protocol")
    parser.add_argument(
        "--output", type=Path, help="New JSON path; existing files are never overwritten"
    )
    args = parser.parse_args()
    try:
        if args.output is not None and (args.output.exists() or args.output.is_symlink()):
            raise FileExistsError(f"Output already exists: {args.output}; choose a new path")
        result = run_wine_experiment(WineConfig(epochs=args.epochs))
        encoded = (
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode()
        if args.output is not None:
            _atomic_write(args.output, lambda output: output.write(encoded))
        print(encoded.decode(), end="")
        if not result["all_checks_passed"]:
            parser.exit(1, "One or more data-protocol checks failed.\n")
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"Wine experiment failed: {error}\n")


if __name__ == "__main__":
    main()
