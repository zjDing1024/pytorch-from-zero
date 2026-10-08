"""Train, resume, or verify epoch-boundary recovery for the CPU affine lab."""

import argparse
import json
import os
from pathlib import Path

from pytorch_lab.checkpoint import (
    EpochTrainer,
    RecoveryConfig,
    _atomic_write,
    load_checkpoint,
    save_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Start a new fixed CPU/float64 synthetic run")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--batch-size", type=int, default=20)
    train.add_argument("--learning-rate", type=float, default=0.05)
    train.add_argument("--momentum", type=float, default=0.8)
    train.add_argument("--noise-std", type=float, default=0.05)
    train.add_argument("--epochs", type=int, default=40, help="Total completed epochs")
    train.add_argument("--checkpoint", type=Path, required=True, help="New checkpoint path")
    resume = commands.add_parser("resume", help="Resume an owned/trusted checkpoint")
    resume.add_argument("--checkpoint", type=Path, required=True, help="Existing trusted file")
    resume.add_argument("--epochs", type=int, required=True, help="Total epochs, not additional")
    resume.add_argument("--save-checkpoint", type=Path, help="New output path; never overwrite")
    verify = commands.add_parser("verify", help="Compare three fresh Python processes")
    verify.add_argument("--seed", type=int, default=42)
    verify.add_argument("--epochs", type=int, default=40)
    verify.add_argument("--split-epoch", type=int, default=7)
    for command in (train, resume, verify):
        command.add_argument("--output", type=Path, help="Optional new JSON evidence path")
    args = parser.parse_args()
    try:
        # Fast refusal is a courtesy, not the concurrency guarantee: atomic link below is.
        destinations = [args.output]
        if args.command == "train":
            destinations.append(args.checkpoint)
        elif args.command == "resume":
            destinations.append(args.save_checkpoint)
        resolved = [path.absolute() for path in destinations if path is not None]
        if len(resolved) != len(set(resolved)):
            raise ValueError("checkpoint and JSON output paths must differ")
        for path in destinations:
            if path is not None and (path.exists() or path.is_symlink()):
                raise FileExistsError(f"Output already exists: {path}; choose a new path")
        if args.command == "verify":
            from pytorch_lab.checkpoint_experiment import run_checkpoint_experiment

            result = run_checkpoint_experiment(args.seed, args.epochs, args.split_epoch)
        else:
            if args.command == "train":
                config = RecoveryConfig(
                    seed=args.seed,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    momentum=args.momentum,
                    noise_std=args.noise_std,
                )
                trainer = EpochTrainer(config)
            else:
                trainer = load_checkpoint(args.checkpoint)
            trainer.train_to(args.epochs)
            destination = args.checkpoint if args.command == "train" else args.save_checkpoint
            if destination is not None:
                save_checkpoint(trainer, destination)
            result = {"process_id": os.getpid(), "training": trainer.report()}
        encoded = (
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode()
        if args.output is not None:
            _atomic_write(args.output, lambda output: output.write(encoded))
        print(encoded.decode(), end="")
        if result.get("all_checks_passed") is False:
            parser.exit(1, "One or more recovery checks failed.\n")
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"Recovery failed: {error}\n")


if __name__ == "__main__":
    main()
