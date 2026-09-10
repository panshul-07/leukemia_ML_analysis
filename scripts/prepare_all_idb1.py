from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leukemia_osl.preprocess import (
    build_manifest,
    extract_rar,
    stratified_split,
    summarize_records,
    write_imagefolder_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ALL-IDB1 as ImageFolder splits.")
    parser.add_argument("--archive", type=Path, help="Path to ALL_IDB1.rar.")
    parser.add_argument(
        "--rar-password",
        type=str,
        default=None,
        help="Password for the RAR archive.",
    )
    parser.add_argument(
        "--rar-password-env",
        type=str,
        default=None,
        help="Environment variable containing the RAR password.",
    )
    parser.add_argument("--source", type=Path, help="Already extracted ALL-IDB1 directory.")
    parser.add_argument("--raw-output", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-output", type=Path, default=Path("data/processed/all_idb1"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rar_password = args.rar_password
    if rar_password is None and args.rar_password_env:
        rar_password = os.environ.get(args.rar_password_env)

    if args.archive:
        print(f"Extracting {args.archive} into {args.raw_output}...")
        extract_rar(args.archive, args.raw_output, password=rar_password)

    source = args.source or args.raw_output
    records = build_manifest(source)
    counts = summarize_records(records)
    print(f"Detected {len(records)} labeled images under {source}.")
    print(f"Class counts: {counts}")
    expected_counts = {"healthy": 59, "leukemia": 49}
    if counts != expected_counts:
        print(
            "Warning: paper ALL-IDB1 counts are {'healthy': 59, 'leukemia': 49}. "
            "If this is meant to be full ALL-IDB1, verify the archive password/layout."
        )

    splits = stratified_split(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    for split_name, split_records in splits.items():
        print(f"{split_name}: {len(split_records)} images {summarize_records(split_records)}")

    if args.dry_run:
        print("Dry run complete; no files copied.")
        return

    write_imagefolder_split(splits, args.processed_output, overwrite=args.overwrite)
    print(f"Prepared dataset written to {args.processed_output}")


if __name__ == "__main__":
    main()
