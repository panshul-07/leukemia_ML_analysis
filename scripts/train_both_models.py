from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train both ALL-IDB1 models on the prepared real dataset."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hybrid-backbone", type=str, default="vgg11_bn")
    return parser.parse_args()


def append_common_args(command: list[str], args: argparse.Namespace) -> list[str]:
    if args.data_dir is not None:
        command.extend(["--data-dir", str(args.data_dir)])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.num_workers is not None:
        command.extend(["--num-workers", str(args.num_workers)])
    if args.device is not None:
        command.extend(["--device", args.device])
    return command


def run_training(command: list[str]) -> None:
    print("\nRunning:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    runs = [
        [
            sys.executable,
            "train.py",
            "--config",
            str(args.config),
            "--model",
            "resnet18_osl",
            "--preprocessing-profile",
            "paper",
            "--report-dir",
            "results/reports/resnet18_osl",
            "--checkpoint-dir",
            "results/checkpoints/resnet18_osl",
        ],
        [
            sys.executable,
            "train.py",
            "--config",
            str(args.config),
            "--model",
            "hybrid_cnn_transformer",
            "--cnn-backbone-name",
            args.hybrid_backbone,
            "--preprocessing-profile",
            "hybrid",
            "--report-dir",
            "results/reports/hybrid_cnn_transformer",
            "--checkpoint-dir",
            "results/checkpoints/hybrid_cnn_transformer",
        ],
    ]

    for command in runs:
        run_training(append_common_args(command, args))


if __name__ == "__main__":
    main()
