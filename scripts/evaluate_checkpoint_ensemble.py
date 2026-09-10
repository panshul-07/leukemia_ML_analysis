from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

from leukemia_osl.data import make_record_loaders
from leukemia_osl.model import create_model
from leukemia_osl.preprocess import (
    DEFAULT_CLASS_NAMES,
    acquisition_group_ids,
    build_manifest,
    content_group_ids,
    validate_prepared_all_idb1,
)
from leukemia_osl.results import compute_result_metrics, save_result_bundle
from train import evaluate, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EnsembleMember:
    name: str
    checkpoint_dir: Path
    source_result_dir: Path | None
    profile: str
    image_size: int
    batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an ensemble of saved grouped checkpoints. Blend weights and "
            "decision threshold are selected on each fold's validation groups only."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed/all_idb1")
    parser.add_argument("--result-name", type=str, required=True)
    parser.add_argument("--grouping", choices=["acquisition", "content"], default="acquisition")
    parser.add_argument(
        "--member",
        action="append",
        required=True,
        help=(
            "Ensemble member as name|checkpoint_dir|source_result_dir. The source "
            "result dir supplies profile, image size, and batch size from run_config.json."
        ),
    )
    parser.add_argument("--weight-step", type=float, default=0.1)
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


def load_run_config(path: Path | None) -> dict:
    if path is None:
        return {}
    config_path = path / "run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_member(text: str) -> EnsembleMember:
    parts = text.split("|")
    if len(parts) != 3:
        raise ValueError("--member must be formatted as name|checkpoint_dir|source_result_dir")
    name, checkpoint_dir_text, source_result_dir_text = parts
    source_result_dir = Path(source_result_dir_text).expanduser().resolve()
    run_config = load_run_config(source_result_dir)
    profile = run_config.get("preprocessing_profile") or run_config.get("profile") or "hybrid"
    image_size = int(run_config.get("image_size", 224))
    batch_size = int(run_config.get("batch_size", 16))
    return EnsembleMember(
        name=name,
        checkpoint_dir=Path(checkpoint_dir_text).expanduser().resolve(),
        source_result_dir=source_result_dir,
        profile=profile,
        image_size=image_size,
        batch_size=batch_size,
    )


def records_for_groups(records, groups: np.ndarray, selected_groups: list[str]):
    selected = set(selected_groups)
    return [record for record, group in zip(records, groups, strict=True) if group in selected]


def threshold_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities[:, 1] >= threshold).astype(int)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    row_sums = probabilities.sum(axis=1, keepdims=True)
    return np.divide(
        probabilities,
        row_sums,
        out=np.full_like(probabilities, 1.0 / probabilities.shape[1]),
        where=row_sums > 0,
    )


def best_validation_threshold(
    labels: list[int],
    probabilities: np.ndarray,
) -> tuple[float, float, float, float]:
    labels_array = np.asarray(labels, dtype=int)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities[:, 1])))
    best_score = (-1.0, -1.0, -1.0, 0.5)
    for threshold in candidates:
        predictions = threshold_predictions(probabilities, float(threshold))
        balanced = balanced_accuracy_score(labels_array, predictions)
        accuracy = float((predictions == labels_array).mean())
        f1_macro = f1_score(labels_array, predictions, average="macro", zero_division=0)
        score = (balanced, accuracy, f1_macro, -float(threshold))
        if score > best_score:
            best_score = score
    balanced, accuracy, f1_macro, negative_threshold = best_score
    return -negative_threshold, balanced, accuracy, f1_macro


def weight_candidates(member_count: int, step: float) -> list[np.ndarray]:
    if member_count < 1:
        raise ValueError("At least one member is required.")
    if member_count == 1:
        return [np.ones(1, dtype=np.float64)]
    units = int(round(1.0 / step))
    if units < 1 or not np.isclose(units * step, 1.0):
        raise ValueError("--weight-step must divide 1.0 exactly, for example 0.1 or 0.2")

    candidates: list[np.ndarray] = []

    def build(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            candidates.append(np.asarray([*prefix, remaining], dtype=np.float64) / units)
            return
        for value in range(remaining + 1):
            build([*prefix, value], remaining - value, slots - 1)

    build([], units, member_count)
    return [candidate for candidate in candidates if np.any(candidate > 0)]


def infer_backbone_from_state_dict(checkpoint: dict, model_config: dict) -> dict:
    if model_config.get("name") != "hybrid_cnn_transformer":
        return model_config
    if "cnn_backbone_name" in model_config:
        return model_config
    projection_weight = checkpoint["model_state_dict"].get("projection.weight")
    if projection_weight is None:
        return model_config
    input_channels = int(projection_weight.shape[1])
    inferred = {
        512: "vgg11_bn",
        1280: "efficientnet_b0",
        960: "mobilenet_v3_large",
    }.get(input_channels)
    if inferred is not None:
        model_config = dict(model_config)
        model_config["cnn_backbone_name"] = inferred
    return model_config


def evaluate_member_on_fold(
    member: EnsembleMember,
    fold: int,
    records,
    groups: np.ndarray,
    device: torch.device,
    criterion,
    test_time_augmentation: bool,
) -> tuple[dict, list[int], np.ndarray, list[int], np.ndarray, np.ndarray | None, list[float], list[str]]:
    checkpoint_path = member.checkpoint_dir / f"fold_{fold:02d}_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {member.name}: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = list(checkpoint["class_names"])
    model_config = infer_backbone_from_state_dict(checkpoint, dict(checkpoint["model_config"]))
    model_config["pretrained_backbone"] = False
    model = create_model(num_classes=len(class_names), **model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_records = records_for_groups(records, groups, checkpoint["train_groups"])
    val_records = records_for_groups(records, groups, checkpoint["val_groups"])
    test_records = records_for_groups(records, groups, checkpoint["test_groups"])
    loaders, _ = make_record_loaders(
        train_records,
        val_records,
        test_records,
        image_size=member.image_size,
        batch_size=member.batch_size,
        num_workers=0,
        profile=member.profile,
        mean=tuple(checkpoint["normalization_mean"]),
        std=tuple(checkpoint["normalization_std"]),
    )
    _, _, val_labels, _, val_probabilities, _, _ = evaluate(
        model,
        loaders["val"],
        criterion,
        device,
        test_time_augmentation=test_time_augmentation,
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
        test_time_augmentation=test_time_augmentation,
    )
    test_paths = [str(record.path) for record in test_records]
    return (
        checkpoint,
        val_labels,
        val_probabilities,
        test_labels,
        test_probabilities,
        test_embeddings,
        inference_times,
        test_paths,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    criterion = nn.CrossEntropyLoss()
    members = [parse_member(text) for text in args.member]
    weights_to_try = weight_candidates(len(members), args.weight_step)
    result_dir = PROJECT_ROOT / "results" / args.result_name
    result_dir.mkdir(parents=True, exist_ok=True)

    validate_prepared_all_idb1(args.data_dir)
    records = build_manifest(args.data_dir)
    groups = (
        np.asarray(acquisition_group_ids(records))
        if args.grouping == "acquisition"
        else np.asarray(content_group_ids(records))
    )

    canonical_paths = sorted(members[0].checkpoint_dir.glob("fold_*_best.pth"))
    if not canonical_paths:
        raise FileNotFoundError(f"No fold checkpoints found in {members[0].checkpoint_dir}")
    folds = [int(path.stem.split("_")[1]) for path in canonical_paths]

    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    all_paths: list[str] = []
    all_groups: list[str] = []
    all_inference_times: list[float] = []
    history: list[dict] = []
    fold_rows: list[dict] = []
    class_names = list(DEFAULT_CLASS_NAMES)
    weighted_test_loss = 0.0
    total_images = 0

    for fold in folds:
        member_outputs = [
            evaluate_member_on_fold(
                member,
                fold,
                records,
                groups,
                device,
                criterion,
                args.test_time_augmentation,
            )
            for member in members
        ]
        canonical_checkpoint = member_outputs[0][0]
        for member, output in zip(members[1:], member_outputs[1:], strict=True):
            checkpoint = output[0]
            for key in ("train_groups", "val_groups", "test_groups"):
                if sorted(checkpoint[key]) != sorted(canonical_checkpoint[key]):
                    raise ValueError(
                        f"{member.name} fold {fold} has different {key}; refusing "
                        "to ensemble checkpoints with mismatched holdout groups."
                    )

        val_labels = member_outputs[0][1]
        test_labels = member_outputs[0][3]
        for output in member_outputs[1:]:
            if output[1] != val_labels or output[3] != test_labels:
                raise ValueError(f"Fold {fold} labels do not align across ensemble members.")

        val_probabilities_by_member = [output[2] for output in member_outputs]
        test_probabilities_by_member = [output[4] for output in member_outputs]

        best_score = (-1.0, -1.0, -1.0, -float("inf"))
        best_threshold = 0.5
        best_weights = weights_to_try[0]
        for weights in weights_to_try:
            val_probabilities = sum(
                weight * probabilities
                for weight, probabilities in zip(weights, val_probabilities_by_member, strict=True)
            )
            val_probabilities = normalize_probabilities(val_probabilities)
            threshold, balanced, accuracy, macro_f1 = best_validation_threshold(
                val_labels,
                val_probabilities,
            )
            validation_loss = log_loss(val_labels, val_probabilities, labels=[0, 1])
            score = (balanced, accuracy, macro_f1, -float(validation_loss))
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_weights = weights

        test_probabilities = sum(
            weight * probabilities
            for weight, probabilities in zip(best_weights, test_probabilities_by_member, strict=True)
        )
        test_probabilities = normalize_probabilities(test_probabilities)
        test_predictions = threshold_predictions(test_probabilities, best_threshold).tolist()
        test_loss = float(
            sum(
                weight
                * nn.CrossEntropyLoss()(
                    torch.from_numpy(np.log(np.clip(probabilities, 1e-8, 1.0))).float(),
                    torch.tensor(test_labels, dtype=torch.long),
                ).item()
                for weight, probabilities in zip(best_weights, test_probabilities_by_member, strict=True)
            )
        )

        inference_times = [
            sum(times) for times in zip(*[output[6] for output in member_outputs], strict=True)
        ]
        embeddings = np.concatenate(
            [
                output[5]
                for output in member_outputs
                if output[5] is not None
            ],
            axis=1,
        )
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
        fold_metric = {
            **fold_metrics,
            "Fold": fold,
            "Evaluation/Test_Groups": ",".join(canonical_checkpoint["test_groups"]),
            "Decision/Threshold": best_threshold,
            "Decision/Val_Balanced_Accuracy": best_score[0],
            "Decision/Val_Accuracy": best_score[1],
            "Decision/Val_F1_Macro": best_score[2],
            "Decision/Val_Log_Loss": -best_score[3],
        }
        for member, weight in zip(members, best_weights, strict=True):
            fold_metric[f"Weight/{member.name}"] = float(weight)
        fold_rows.append(fold_metric)
        history.append(
            {
                "fold": fold,
                "epoch": 1,
                "train_loss": np.nan,
                "train_accuracy": np.nan,
                "val_loss": np.nan,
                "val_accuracy": best_score[1],
                "learning_rate": np.nan,
            }
        )
        print(
            f"Fold {fold:02d}: test_accuracy={fold_metrics['Final/Accuracy']:.4f}, "
            f"threshold={best_threshold:.4f}, "
            f"weights={dict(zip([member.name for member in members], best_weights.round(3)))}",
            flush=True,
        )

        test_group_set = set(canonical_checkpoint["test_groups"])
        test_groups = [group for group in groups if group in test_group_set]
        all_labels.extend(test_labels)
        all_predictions.extend(test_predictions)
        all_probabilities.append(test_probabilities)
        all_embeddings.append(embeddings)
        all_paths.extend(member_outputs[0][7])
        all_groups.extend(test_groups)
        all_inference_times.extend(inference_times)
        weighted_test_loss += test_loss * len(test_labels)
        total_images += len(test_labels)

    probabilities = np.concatenate(all_probabilities, axis=0)
    embeddings = np.concatenate(all_embeddings, axis=0)
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
    final_metrics["Evaluation/Folds"] = len(folds)
    final_metrics["Training/Pretrained_Weights"] = 0
    final_metrics["Decision/Policy"] = "validation_grid_weights_plus_validation_threshold"

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
    pd.DataFrame(fold_rows).sort_values("Fold").to_csv(result_dir / "fold_metrics.csv", index=False)
    predictions = pd.read_csv(result_dir / "predictions.csv")
    predictions.insert(1, f"{args.grouping}_group", all_groups)
    predictions.to_csv(result_dir / "predictions.csv", index=False)
    (result_dir / "run_config.json").write_text(
        json.dumps(
            {
                "members": [
                    {
                        "name": member.name,
                        "checkpoint_dir": str(member.checkpoint_dir),
                        "source_result_dir": str(member.source_result_dir)
                        if member.source_result_dir
                        else None,
                        "profile": member.profile,
                        "image_size": member.image_size,
                        "batch_size": member.batch_size,
                    }
                    for member in members
                ],
                "weight_step": args.weight_step,
                "grouping": args.grouping,
                "test_time_augmentation": args.test_time_augmentation,
                "seed": args.seed,
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
