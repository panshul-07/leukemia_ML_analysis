"""Utilities for extracting and preparing the ALL-IDB1 dataset."""

from __future__ import annotations

import json
import hashlib
import random
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DEFAULT_CLASS_NAMES = ("healthy", "leukemia")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str


def extract_rar(archive_path: Path, output_dir: Path, password: str | None = None) -> Path:
    """Extract a RAR archive with unar or SevenZip and return the output directory."""

    archive_path = archive_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ALL-IDB1 uses an older encrypted RAR method that current SevenZip can
    # inspect but cannot extract. unar supports the archive correctly.
    if shutil.which("unar"):
        cmd = ["unar", "-quiet", "-force-overwrite", "-o", str(output_dir)]
        if password is not None:
            cmd.extend(["-p", password])
        cmd.append(str(archive_path))
    elif shutil.which("7zz"):
        cmd = ["7zz", "x", "-y", f"-o{output_dir}"]
        if password is not None:
            cmd.append(f"-p{password}")
        cmd.append(str(archive_path))
    else:
        raise RuntimeError(
            "A RAR extractor is required. Install unar with `brew install unar`."
        )

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to extract the RAR archive. Verify its password and integrity."
        ) from None

    return output_dir


def infer_all_idb1_label(image_path: Path) -> str:
    """Infer ALL-IDB1 class from official filename suffixes or class folder names.

    Official ALL-IDB1 filenames usually end with `_0` for healthy and `_1` for ALL.
    Directory-name fallback keeps the script usable for already organized copies.
    """

    stem = image_path.stem.lower()
    suffix_match = re.search(r"[_-]([01])$", stem)
    if suffix_match:
        return "healthy" if suffix_match.group(1) == "0" else "leukemia"

    parent_names = [part.name.lower() for part in image_path.parents[:3]]
    healthy_tokens = {"healthy", "normal", "hem", "non-leukemia", "non_leukemia", "negative"}
    leukemia_tokens = {"leukemia", "leukaemia", "all", "sick", "positive", "blast"}

    for parent in parent_names:
        normalized = parent.replace(" ", "_").replace("-", "_")
        if normalized in healthy_tokens or "healthy" in normalized or "normal" in normalized:
            return "healthy"
        if normalized in leukemia_tokens or normalized.startswith("leuk"):
            return "leukemia"

    raise ValueError(
        f"Could not infer class for {image_path}. Expected official ALL-IDB1 suffix "
        "like `*_0.jpg` or `*_1.jpg`, or class folders named healthy/leukemia."
    )


def discover_images(source_dir: Path) -> list[Path]:
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Dataset source does not exist: {source_dir}")

    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_manifest(source_dir: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    skipped: list[str] = []

    for image_path in discover_images(source_dir):
        try:
            records.append(ImageRecord(path=image_path, label=infer_all_idb1_label(image_path)))
        except ValueError as exc:
            skipped.append(str(exc))

    if not records:
        detail = "\n".join(skipped[:8])
        raise ValueError(f"No labeled ALL-IDB1 images were found under {source_dir}.\n{detail}")

    return records


def summarize_records(records: Iterable[ImageRecord]) -> dict[str, int]:
    return dict(sorted(Counter(record.label for record in records).items()))


def content_group_ids(records: Iterable[ImageRecord]) -> list[str]:
    """Return stable content hashes for leakage-safe grouped splitting."""

    return [hashlib.sha256(record.path.read_bytes()).hexdigest() for record in records]


def acquisition_group_ids(
    records: Iterable[ImageRecord],
    max_gap_seconds: int = 300,
) -> list[str]:
    """Group nearby captures without using labels.

    ALL-IDB1 does not provide patient or slide identifiers. Its EXIF metadata does
    expose camera and capture time, so consecutive filenames are treated as one
    acquisition burst until the camera changes, time goes backwards, or the gap
    exceeds ``max_gap_seconds``. Exact duplicate files are always unioned into the
    same group. The returned IDs align with the input record order.
    """

    record_list = list(records)
    if not record_list:
        return []
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive.")

    def image_number(record: ImageRecord) -> tuple[int, str]:
        match = re.search(r"(\d+)", record.path.stem)
        return (int(match.group(1)) if match else 10**9, record.path.name)

    sorted_indices = sorted(range(len(record_list)), key=lambda index: image_number(record_list[index]))
    raw_groups = [-1] * len(record_list)
    previous_camera: tuple[str, str] | None = None
    previous_timestamp: datetime | None = None
    group_index = -1

    for record_index in sorted_indices:
        record = record_list[record_index]
        with Image.open(record.path) as image:
            exif = image.getexif()
            make = str(exif.get(271, "")).strip()
            model = str(exif.get(272, "")).strip()
            timestamp_text = str(exif.get(306, "")).strip()

        if not make or not model or not timestamp_text:
            raise ValueError(f"Missing camera/time EXIF metadata: {record.path}")
        timestamp = datetime.strptime(timestamp_text, "%Y:%m:%d %H:%M:%S")
        camera = (make, model)
        gap_seconds = (
            None
            if previous_timestamp is None
            else (timestamp - previous_timestamp).total_seconds()
        )
        starts_new_group = (
            previous_camera is None
            or camera != previous_camera
            or gap_seconds is None
            or gap_seconds <= 0
            or gap_seconds > max_gap_seconds
        )
        if starts_new_group:
            group_index += 1
        raw_groups[record_index] = group_index
        previous_camera = camera
        previous_timestamp = timestamp

    parent = list(range(group_index + 1))

    def find(group: int) -> int:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    digest_to_group: dict[str, int] = {}
    for record, group in zip(record_list, raw_groups, strict=True):
        digest = hashlib.sha256(record.path.read_bytes()).hexdigest()
        if digest in digest_to_group:
            union(group, digest_to_group[digest])
        else:
            digest_to_group[digest] = group

    root_to_compact: dict[int, int] = {}
    compact_groups: list[str] = []
    for group in raw_groups:
        root = find(group)
        compact = root_to_compact.setdefault(root, len(root_to_compact))
        compact_groups.append(f"acquisition_{compact:02d}")
    return compact_groups


def find_duplicate_content(records: Iterable[ImageRecord]) -> list[list[Path]]:
    """Return groups of source files whose image bytes are exactly identical."""

    digest_groups: dict[str, list[Path]] = {}
    for record in records:
        digest = hashlib.sha256(record.path.read_bytes()).hexdigest()
        digest_groups.setdefault(digest, []).append(record.path)

    return [
        sorted(paths)
        for paths in digest_groups.values()
        if len(paths) > 1
    ]


def validate_prepared_all_idb1(
    processed_dir: Path,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, dict[str, int]]:
    """Validate labels, class totals, and split isolation for prepared ALL-IDB1."""

    processed_dir = processed_dir.expanduser().resolve()
    expected_counts = expected_counts or {"healthy": 59, "leukemia": 49}
    split_counts: dict[str, dict[str, int]] = {}
    observed_totals: Counter[str] = Counter()
    content_hash_to_split: dict[str, str] = {}

    for split_name in ("train", "val", "test"):
        split_dir = processed_dir / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing ALL-IDB1 split directory: {split_dir}")

        class_counts: dict[str, int] = {}
        for class_name in DEFAULT_CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")

            image_paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not image_paths:
                raise ValueError(f"No images found in {class_dir}")

            class_counts[class_name] = len(image_paths)
            observed_totals[class_name] += len(image_paths)
            for image_path in image_paths:
                inferred_label = infer_all_idb1_label(image_path)
                if inferred_label != class_name:
                    raise ValueError(
                        f"Label mismatch: {image_path.name} implies {inferred_label}, "
                        f"but is stored under {class_name}."
                    )

                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                previous_split = content_hash_to_split.get(digest)
                if previous_split is not None:
                    if previous_split != split_name:
                        raise ValueError(
                            f"Data leakage: identical image content occurs in "
                            f"{previous_split} and {split_name}."
                        )
                else:
                    content_hash_to_split[digest] = split_name

        split_counts[split_name] = class_counts

    observed = dict(sorted(observed_totals.items()))
    if observed != expected_counts:
        raise ValueError(
            f"Prepared data counts are {observed}; expected full ALL-IDB1 counts "
            f"are {expected_counts}."
        )
    return split_counts


def stratified_split(
    records: list[ImageRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[ImageRecord]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train, val, and test ratios must sum to 1.0")

    rng = random.Random(seed)
    by_label: dict[str, list[ImageRecord]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)

    splits = {"train": [], "val": [], "test": []}
    for label_records in by_label.values():
        shuffled = list(label_records)
        rng.shuffle(shuffled)
        n_records = len(shuffled)

        n_train = int(n_records * train_ratio)
        n_val = int(n_records * val_ratio)

        if val_ratio > 0 and n_val == 0 and n_records - n_train > 1:
            n_val = 1
        if test_ratio > 0 and n_records - n_train - n_val == 0 and n_train > 1:
            n_train -= 1

        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train : n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val :])

    for split_records in splits.values():
        rng.shuffle(split_records)

    return splits


def write_imagefolder_split(
    splits: dict[str, list[ImageRecord]],
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    for split_name, records in splits.items():
        for record in records:
            class_dir = output_dir / split_name / record.label
            class_dir.mkdir(parents=True, exist_ok=True)
            destination = class_dir / record.path.name
            if destination.exists():
                destination = class_dir / (
                    f"{record.path.stem}_{abs(hash(record.path))}{record.path.suffix}"
                )
            shutil.copy2(record.path, destination)

    manifest = {
        "class_names": list(DEFAULT_CLASS_NAMES),
        "splits": {
            split_name: [asdict(record) | {"path": str(record.path)} for record in records]
            for split_name, records in splits.items()
        },
        "counts": {
            split_name: summarize_records(records) for split_name, records in splits.items()
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
