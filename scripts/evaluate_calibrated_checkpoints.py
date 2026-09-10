from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score

from leukemia_osl.data import make_record_loaders
from leukemia_osl.model import create_model
from leukemia_osl.preprocess import (
    acquisition_group_ids,
    build_manifest,
    content_group_ids,
    validate_prepared_all_idb1,
)
from leukemia_osl.results import compute_result_metrics, save_result_bundle
from train import evaluate, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved grouped checkpoints with a decision threshold selected "
            "from each fold's validation split only."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed/all_idb1")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-result-dir", type=Path, default=None)
    parser.add_argument("--result-name", type=str, required=True)
    parser.add_argument("--grouping", choices=["acquisition", "content"], default="acquisition")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-time-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def choose_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def threshold_predictions(probabilities: np.ndarray, threshold: float) -> list[int]:
    return (probabilities[:, 1] >= threshold).astype(int).tolist()


def best_validation_threshold(labels: list[int], probabilities: np.ndarray) -> tuple[float, float, float]:
    labels_array = np.asarray(labels, dtype=int)
    positive_scores = probabilities[:, 1]
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], positive_scores)))
    best_score = (-1.0, -1.0, -1.0, 0.5)
    for threshold in candidates:
        predictions = np.asarray(threshold_predictions(probabilities, float(threshold)))
        balanced = balanced_accuracy_score(labels_array, predictions)
        accuracy = float((predictions == labels_array).mean())
        f1_macro = f1_score(labels_array, predictions, average="macro", zero_division=0)
        score = (balanced, accuracy, f1_macro, -float(threshold))
        if score > best_score:
            best_score = score
    balanced, accuracy, _, negative_threshold = best_score
    return -negative_threshold, balanced, accuracy


def load_run_config(path: Path | None) -> dict:
    if path is None:
        return {}
    config_path = path / "run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def records_for_groups(records, groups: np.ndarray, selected_groups: list[str]):
    selected = set(selected_groups)
    return [record for record, group in zip(records, groups, strict=True) if group in selected]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    source_result_dir = args.source_result_dir.expanduser().resolve() if args.source_result_dir else None
    result_dir = PROJECT_ROOT / "results" / args.result_name
    result_dir.mkdir(parents=True, exist_ok=True)

    run_config = load_run_config(source_result_dir)
    profile = args.profile or run_config.get("preprocessing_profile") or run_config.get("profile") or "hybrid"
    image_size = args.image_size or int(run_config.get("image_size", 224))
    batch_size = args.batch_size or int(run_config.get("batch_size", 16))

    validate_prepared_all_idb1(args.data_dir)
    records = build_manifest(args.data_dir)
    groups = (
        np.asarray(acquisition_group_ids(records))
        if args.grouping == "acquisition"
        else np.asarray(content_group_ids(records))
    )

    criterion = nn.CrossEntropyLoss()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    all_paths: list[str] = []
    all_groups: list[str] = []
    all_folds: list[int] = []
    all_thresholds: list[float] = []
    all_inference_times: list[float] = []
    fold_rows: list[dict] = []
    weighted_test_loss = 0.0
    total_images = 0

    checkpoint_paths = sorted(checkpoint_dir.glob("fold_*_best.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No fold checkpoints found in {checkpoint_dir}")

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        fold = int(checkpoint.get("fold", checkpoint_path.stem.split("_")[1]))
        class_names = list(checkpoint["class_names"])
        model_config = dict(checkpoint["model_config"])
        model_config["pretrained_backbone"] = False
        model = create_model(num_classes=len(class_names), **model_config).to(device)
        assert model.uses_pretrained_weights is False
        model.load_state_dict(checkpoint["model_state_dict"])

        train_records = records_for_groups(records, groups, checkpoint["train_groups"])
        val_records = records_for_groups(records, groups, checkpoint["val_groups"])
        test_records = records_for_groups(records, groups, checkpoint["test_groups"])
        loaders, _ = make_record_loaders(
            train_records,
            val_records,
            test_records,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=0,
            profile=profile,
            mean=tuple(checkpoint["normalization_mean"]),
            std=tuple(checkpoint["normalization_std"]),
        )

        _, _, val_labels, _, val_probabilities, _, _ = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            test_time_augmentation=args.test_time_augmentation,
        )
        threshold, val_balanced, val_accuracy = best_validation_threshold(
            val_labels,
            val_probabilities,
        )
        (
            test_loss,
            _,
            test_labels,
            _,
            test_probabilities,
            test_embeddings,
            inference_times,
        ) = evaluate(
            model,
            loaders["test"],
            criterion,
            device,
            collect_embeddings=True,
            measure_inference=True,
            test_time_augmentation=args.test_time_augmentation,
        )
        test_predictions = threshold_predictions(test_probabilities, threshold)
        fold_metrics = compute_result_metrics(
            test_labels,
            test_predictions,
            test_probabilities,
            class_names,
            test_loss,
            inference_times,
            bootstrap_samples=0,
            seed=args.seed + fold,
        )
        fold_metrics.update(
            {
                "Fold": fold,
                "Evaluation/Test_Groups": ",".join(checkpoint["test_groups"]),
                "Decision/Threshold": threshold,
                "Decision/Val_Balanced_Accuracy": val_balanced,
                "Decision/Val_Accuracy": val_accuracy,
            }
        )
        fold_rows.append(fold_metrics)
        print(
            f"Fold {fold:02d}: test_accuracy={fold_metrics['Final/Accuracy']:.4f}, "
            f"threshold={threshold:.4f}, val_balanced={val_balanced:.4f}",
            flush=True,
        )

        all_labels.extend(test_labels)
        all_predictions.extend(test_predictions)
        all_probabilities.append(test_probabilities)
        if test_embeddings is not None:
            all_embeddings.append(test_embeddings)
        all_paths.extend(str(record.path) for record in test_records)
        fold_groups = [group for group in groups if group in set(checkpoint["test_groups"])]
        all_groups.extend(fold_groups)
        all_folds.extend([fold] * len(test_records))
        all_thresholds.extend([threshold] * len(test_records))
        all_inference_times.extend(inference_times)
        weighted_test_loss += test_loss * len(test_labels)
        total_images += len(test_labels)

    probabilities = np.concatenate(all_probabilities, axis=0)
    embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else None
    final_metrics = compute_result_metrics(
        all_labels,
        all_predictions,
        probabilities,
        class_names,
        weighted_test_loss / total_images,
        all_inference_times,
        seed=args.seed,
        bootstrap_groups=all_groups,
    )
    final_metrics["Evaluation/Folds"] = len(checkpoint_paths)
    final_metrics["Training/Pretrained_Weights"] = 0
    final_metrics["Decision/Mean_Threshold"] = float(np.mean(all_thresholds))
    final_metrics["Decision/Policy"] = "validation_balanced_accuracy"

    if source_result_dir and (source_result_dir / "training_history.csv").exists():
        history = pd.read_csv(source_result_dir / "training_history.csv").to_dict("records")
    else:
        history = [
            {
                "epoch": 0,
                "train_loss": np.nan,
                "val_loss": np.nan,
                "train_accuracy": np.nan,
                "val_accuracy": np.nan,
            }
        ]

    saved_paths = save_result_bundle(
        history,
        final_metrics,
        all_labels,
        all_predictions,
        probabilities,
        class_names,
        result_dir,
        image_paths=all_paths,
        embeddings=embeddings,
    )
    pd.DataFrame(fold_rows).sort_values("Fold").to_csv(
        result_dir / "fold_metrics.csv",
        index=False,
    )
    predictions = pd.read_csv(result_dir / "predictions.csv")
    predictions.insert(1, f"{args.grouping}_group", all_groups)
    predictions.insert(1, "fold", all_folds)
    predictions["decision_threshold"] = all_thresholds
    predictions.to_csv(result_dir / "predictions.csv", index=False)
    (result_dir / "run_config.json").write_text(
        json.dumps(
            {
                "checkpoint_dir": str(checkpoint_dir),
                "source_result_dir": str(source_result_dir) if source_result_dir else None,
                "profile": profile,
                "grouping": args.grouping,
                "image_size": image_size,
                "batch_size": batch_size,
                "threshold_policy": "validation_balanced_accuracy",
                "test_time_augmentation": args.test_time_augmentation,
                "seed": args.seed,
                "pretrained_weights": False,
                "source": "saved grouped checkpoints evaluated on prepared real ALL-IDB1 files",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(final_metrics, indent=2))
    print("Saved:")
    for path in saved_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
