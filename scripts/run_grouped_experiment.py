from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from leukemia_osl.data import estimate_record_channel_stats, make_record_loaders
from leukemia_osl.model import create_model
from leukemia_osl.preprocess import (
    acquisition_group_ids,
    build_manifest,
    content_group_ids,
    summarize_records,
    validate_prepared_all_idb1,
)
from leukemia_osl.results import compute_result_metrics, save_result_bundle
from train import evaluate, set_seed, train_one_epoch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped ALL-IDB1 tuning experiments.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed/all_idb1")
    parser.add_argument("--result-name", type=str, required=True)
    parser.add_argument("--checkpoint-name", type=str, required=True)
    parser.add_argument(
        "--model-name",
        choices=["hybrid_cnn_transformer", "hybrid_cnn_gnn"],
        default="hybrid_cnn_transformer",
    )
    parser.add_argument("--grouping", choices=["acquisition", "content"], default="acquisition")
    parser.add_argument(
        "--profile",
        choices=[
            "paper",
            "hybrid",
            "hybrid_microscopy",
            "hybrid_segmented_pretrained",
            "gnn_microscopy_enhanced",
        ],
        default="hybrid",
    )
    parser.add_argument("--backbone", type=str, default="vgg11_bn")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.35)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--gnn-dropout", type=float, default=0.20)
    parser.add_argument("--graph-neighbors", type=int, choices=[4, 8], default=8)
    parser.add_argument("--dropout", type=float, default=0.60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--threshold-policy", choices=["argmax", "val_balanced"], default="argmax")
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


def best_validation_threshold(
    labels: list[int],
    probabilities: np.ndarray,
) -> tuple[float, float, float]:
    labels_array = np.asarray(labels, dtype=int)
    candidates = np.unique(
        np.concatenate(([0.0, 0.5, 1.0], probabilities[:, 1]))
    )
    best_score = (-1.0, -1.0, -1.0, 0.5)
    for threshold in candidates:
        predictions = np.asarray(threshold_predictions(probabilities, float(threshold)))
        balanced = balanced_accuracy_score(labels_array, predictions)
        f1_macro = f1_score(labels_array, predictions, average="macro", zero_division=0)
        accuracy = float((predictions == labels_array).mean())
        score = (balanced, accuracy, f1_macro, -float(threshold))
        if score > best_score:
            best_score = score
    balanced, accuracy, f1_macro, negative_threshold = best_score
    return -negative_threshold, balanced, accuracy


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Using processed data: {args.data_dir}")
    split_counts = validate_prepared_all_idb1(args.data_dir)
    print(f"Validated split counts: {split_counts}")

    records = build_manifest(args.data_dir)
    print(f"Records: {len(records)} {summarize_records(records)}")
    labels_for_split = np.array([record.label for record in records])
    groups = (
        np.array(acquisition_group_ids(records))
        if args.grouping == "acquisition"
        else np.array(content_group_ids(records))
    )
    print(f"Grouping: {args.grouping} ({len(set(groups))} groups)")

    model_config = {
        "name": args.model_name,
        "pretrained_backbone": False,
        "freeze_backbone": False,
        "cnn_backbone_name": args.backbone,
        "embed_dim": args.embed_dim,
        "transformer_heads": args.heads,
        "transformer_layers": args.layers,
        "transformer_dropout": args.transformer_dropout,
        "attention_dropout": 0.0,
        "gnn_layers": args.gnn_layers,
        "gnn_dropout": args.gnn_dropout,
        "graph_neighbors": args.graph_neighbors,
        "dropout": args.dropout,
    }
    result_dir = PROJECT_ROOT / "results" / args.result_name
    checkpoint_dir = PROJECT_ROOT / "results" / "checkpoints" / args.checkpoint_name
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    all_paths: list[str] = []
    all_test_groups: list[str] = []
    all_inference_times: list[float] = []
    all_history: list[dict] = []
    fold_metric_rows: list[dict] = []
    weighted_test_loss = 0.0
    total_test_images = 0

    for fold, (train_val_indices, test_indices) in enumerate(
        splitter.split(np.arange(len(records)), labels_for_split, groups),
        start=1,
    ):
        inner_splitter = StratifiedGroupKFold(
            n_splits=args.inner_splits,
            shuffle=True,
            random_state=args.seed + fold,
        )
        train_relative, val_relative = next(
            inner_splitter.split(
                train_val_indices,
                labels_for_split[train_val_indices],
                groups=groups[train_val_indices],
            )
        )
        train_indices = train_val_indices[train_relative]
        val_indices = train_val_indices[val_relative]
        train_records = [records[index] for index in train_indices]
        val_records = [records[index] for index in val_indices]
        test_records = [records[index] for index in test_indices]

        mean, std = estimate_record_channel_stats(
            train_records,
            image_size=args.image_size,
            batch_size=max(1, min(args.batch_size, 8)),
            num_workers=0,
            profile=args.profile,
        )
        loaders, class_names = make_record_loaders(
            train_records,
            val_records,
            test_records,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=0,
            profile=args.profile,
            mean=mean,
            std=std,
        )

        model = create_model(num_classes=len(class_names), **model_config).to(device)
        assert model.uses_pretrained_weights is False
        class_to_idx = {name: index for index, name in enumerate(class_names)}
        train_targets = np.array([class_to_idx[record.label] for record in train_records])
        class_counts = np.bincount(train_targets, minlength=len(class_names))
        class_weights = class_counts.sum() / (len(class_names) * class_counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
            label_smoothing=args.label_smoothing,
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.lr_patience,
            min_lr=1e-6,
        )

        best_val_loss = float("inf")
        best_epoch = 0
        best_train_accuracy = 0.0
        best_val_accuracy = 0.0
        stale_epochs = 0
        best_path = checkpoint_dir / f"fold_{fold:02d}_best.pth"
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train_one_epoch(
                model,
                loaders["train"],
                criterion,
                optimizer,
                device,
                max_grad_norm=args.max_grad_norm,
            )
            val_loss, val_accuracy, _, _, _, _, _ = evaluate(
                model,
                loaders["val"],
                criterion,
                device,
                test_time_augmentation=args.test_time_augmentation,
            )
            scheduler.step(val_loss)
            all_history.append(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                best_train_accuracy = train_accuracy
                best_val_accuracy = val_accuracy
                stale_epochs = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_names": class_names,
                        "fold": fold,
                        "model_config": model_config,
                        "normalization_mean": mean,
                        "normalization_std": std,
                        "train_groups": sorted(set(groups[train_indices].tolist())),
                        "val_groups": sorted(set(groups[val_indices].tolist())),
                        "test_groups": sorted(set(groups[test_indices].tolist())),
                        "pretrained_weights": False,
                        "initialization_source": model.initialization_source,
                    },
                    best_path,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    break

        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

        threshold = 0.5
        if args.threshold_policy == "val_balanced":
            _, _, val_labels, _, val_probabilities, _, _ = evaluate(
                model,
                loaders["val"],
                criterion,
                device,
                test_time_augmentation=args.test_time_augmentation,
            )
            threshold, threshold_balanced, threshold_accuracy = best_validation_threshold(
                val_labels,
                val_probabilities,
            )
        else:
            threshold_balanced = float("nan")
            threshold_accuracy = float("nan")

        (
            test_loss,
            test_accuracy,
            fold_labels,
            raw_fold_predictions,
            fold_probabilities,
            fold_embeddings,
            fold_inference_times,
        ) = evaluate(
            model,
            loaders["test"],
            criterion,
            device,
            collect_embeddings=True,
            measure_inference=True,
            test_time_augmentation=args.test_time_augmentation,
        )
        fold_predictions = (
            threshold_predictions(fold_probabilities, threshold)
            if args.threshold_policy == "val_balanced"
            else raw_fold_predictions
        )
        fold_accuracy = float(
            np.mean(np.asarray(fold_predictions) == np.asarray(fold_labels))
        )

        fold_metrics = compute_result_metrics(
            fold_labels,
            fold_predictions,
            fold_probabilities,
            class_names,
            test_loss,
            fold_inference_times,
            bootstrap_samples=0,
            seed=args.seed + fold,
        )
        fold_metrics.update(
            {
                "Fold": fold,
                "Evaluation/Test_Groups": len(set(groups[test_indices].tolist())),
                "Training/Best_Epoch": best_epoch,
                "Training/Train_Accuracy_At_Best": best_train_accuracy,
                "Training/Val_Accuracy_At_Best": best_val_accuracy,
                "Training/Generalization_Gap": best_train_accuracy - best_val_accuracy,
                "Decision/Threshold": threshold,
                "Decision/Val_Balanced_Accuracy": threshold_balanced,
                "Decision/Val_Accuracy": threshold_accuracy,
            }
        )
        fold_metric_rows.append(fold_metrics)
        print(
            f"Fold {fold:02d}/{args.folds}: test_accuracy={fold_accuracy:.4f}, "
            f"best_epoch={best_epoch}, threshold={threshold:.4f}, "
            f"gap={best_train_accuracy - best_val_accuracy:.4f}",
            flush=True,
        )

        all_labels.extend(fold_labels)
        all_predictions.extend(fold_predictions)
        all_probabilities.append(fold_probabilities)
        if fold_embeddings is not None:
            all_embeddings.append(fold_embeddings)
        all_paths.extend(str(record.path) for record in test_records)
        all_test_groups.extend(groups[test_indices].tolist())
        all_inference_times.extend(fold_inference_times)
        weighted_test_loss += test_loss * len(fold_labels)
        total_test_images += len(fold_labels)

    probabilities = np.concatenate(all_probabilities, axis=0)
    embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else None
    final_metrics = compute_result_metrics(
        all_labels,
        all_predictions,
        probabilities,
        class_names,
        weighted_test_loss / total_test_images,
        all_inference_times,
        seed=args.seed,
        bootstrap_groups=all_test_groups,
    )
    final_metrics["Evaluation/Folds"] = args.folds
    final_metrics["Training/Pretrained_Weights"] = 0
    final_metrics["Training/Mean_Generalization_Gap"] = float(
        np.mean([row["Training/Generalization_Gap"] for row in fold_metric_rows])
    )
    final_metrics["Decision/Mean_Threshold"] = float(
        np.mean([row["Decision/Threshold"] for row in fold_metric_rows])
    )

    saved_paths = save_result_bundle(
        all_history,
        final_metrics,
        all_labels,
        all_predictions,
        probabilities,
        class_names,
        result_dir,
        image_paths=all_paths,
        embeddings=embeddings,
    )
    pd.DataFrame(fold_metric_rows).to_csv(result_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(
        {
            "image_path": all_paths,
            f"{args.grouping}_group": all_test_groups,
        }
    ).to_csv(result_dir / "evaluation_groups.csv", index=False)
    (result_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model": model_config,
                "profile": args.profile,
                "grouping": args.grouping,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "threshold_policy": args.threshold_policy,
                "test_time_augmentation": args.test_time_augmentation,
                "seed": args.seed,
                "source": "prepared real ALL-IDB1 files copied from manifest paths",
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
