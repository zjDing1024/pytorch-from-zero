"""Run the lab and optionally persist its JSON evidence."""

import argparse
import json
from pathlib import Path

from pytorch_lab.experiments import run_all
from pytorch_lab.regression import TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument(
        "--output", type=Path, help="New JSON file; existing files are not overwritten"
    )
    args = parser.parse_args()
    try:
        result = run_all(TrainingConfig(args.seed, args.steps, args.learning_rate))
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"Experiment failed: {error}\n")
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(encoded)
        except FileExistsError:
            parser.exit(2, f"Output already exists: {args.output}. Choose a new path.\n")
    print(encoded, end="")
    if not result["all_checks_passed"]:
        parser.exit(1, "One or more experiment checks failed.\n")


if __name__ == "__main__":
    main()
