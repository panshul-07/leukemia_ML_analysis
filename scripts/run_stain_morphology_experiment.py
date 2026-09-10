from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC

from leukemia_osl.preprocess import (
    DEFAULT_CLASS_NAMES,
    acquisition_group_ids,
    build_manifest,
    content_group_ids,
    summarize_records,
    validate_prepared_all_idb1,
)
from leukemia_osl.results import compute_result_metrics, save_result_bundle
from leukemia_osl.stain_features import (
    FEATURE_VERSION,
    extract_stain_morphology_features,
    feature_dicts_to_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate label-blind ALL-IDB1 stain/morphology preprocessing."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed/all_idb1")
    parser.add_argument("--result-name", type=str, default="stain_morphology_grouped")
    parser.add_argument("--grouping", choices=["acquisition", "content"], default="acquisition")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def _candidate_pipelines(seed: int, feature_count: int) -> dict[str, Pipeline]:
    k_values = sorted({min(feature_count, value) for value in (36, 60, 96, 140, feature_count)})
    candidates: dict[str, Pipeline] = {}

    for scaler_name, scaler in (
        ("standard", StandardScaler()),
        ("robust", RobustScaler(quantile_range=(10.0, 90.0))),
    ):
        for k in k_values:
            for c_value in (0.3, 1.0, 3.0, 10.0, 30.0):
                candidates[f"svm_rbf_{scaler_name}_k{k}_c{c_value:g}"] = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", scaler),
                        ("select", SelectKBest(score_func=f_classif, k=k)),
                        (
                            "classifier",
                            SVC(
                                kernel="rbf",
                                C=c_value,
                                gamma="scale",
                                class_weight="balanced",
                                probability=True,
                                random_state=seed,
                            ),
                        ),
                    ]
                )
            for c_value in (0.3, 1.0, 3.0, 10.0):
                candidates[f"logreg_{scaler_name}_k{k}_c{c_value:g}"] = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", scaler),
                        ("select", SelectKBest(score_func=f_classif, k=k)),
                        (
                            "classifier",
                            LogisticRegression(
                                C=c_value,
                                class_weight="balanced",
                                max_iter=5000,
                                random_state=seed,
                            ),
                        ),
                    ]
                )

    for k in k_values:
        candidates[f"gaussian_nb_quantile_k{k}"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "quantile",
                    QuantileTransformer(
                        n_quantiles=32,
                        output_distribution="normal",
                        random_state=seed,
                    ),
                ),
                ("select", SelectKBest(score_func=f_classif, k=k)),
                ("classifier", GaussianNB(var_smoothing=1e-8)),
            ]
        )

    for max_depth in (None, 4, 7):
        suffix = "none" if max_depth is None else str(max_depth)
        candidates[f"extra_trees_depth{suffix}"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=600,
                        max_depth=max_depth,
                        min_samples_leaf=1,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        candidates[f"random_forest_depth{suffix}"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=600,
                        max_depth=max_depth,
                        min_samples_leaf=1,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    return candidates


def _positive_probabilities(model: Pipeline, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classifier = model.named_steps["classifier"]
    class_positions = {int(label): index for index, label in enumerate(classifier.classes_)}
    full = np.zeros((len(features), len(DEFAULT_CLASS_NAMES)), dtype=np.float64)
    for class_index in range(len(DEFAULT_CLASS_NAMES)):
        if class_index in class_positions:
            full[:, class_index] = probabilities[:, class_positions[class_index]]
    row_sums = full.sum(axis=1, keepdims=True)
    full = np.divide(full, row_sums, out=np.full_like(full, 1.0 / len(DEFAULT_CLASS_NAMES)), where=row_sums > 0)
    return full


def _threshold_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities[:, 1] >= threshold).astype(int)


def _best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, float]:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities[:, 1])))
    best = (-1.0, -1.0, -1.0, 0.5)
    for threshold in candidates:
        predictions = _threshold_predictions(probabilities, float(threshold))
        balanced = balanced_accuracy_score(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        score = (balanced, accuracy, macro_f1, -float(threshold))
        if score > best:
            best = score
    return -best[3], best[0], best[1]


def _safe_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        return float(log_loss(labels, probabilities, labels=[0, 1]))
    except ValueError:
        return float("nan")


def _group_level_counts(groups: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> Counter:
    labels_by_group: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        labels_by_group[str(groups[index])].append(int(labels[index]))
    return Counter(Counter(values).most_common(1)[0][0] for values in labels_by_group.values())


def _inner_train_val_split(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    inner_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    group_label_counts = _group_level_counts(groups, labels, indices)
    feasible_splits = min(inner_splits, len(set(groups[indices])), min(group_label_counts.values()))
    if feasible_splits >= 2:
        splitter = StratifiedGroupKFold(
            n_splits=feasible_splits,
            shuffle=True,
            random_state=seed,
        )
        train_relative, val_relative = next(
            splitter.split(indices, labels[indices], groups=groups[indices])
        )
        return indices[train_relative], indices[val_relative]

    rng = np.random.default_rng(seed)
    unique_groups = np.array(sorted(set(groups[indices])))
    rng.shuffle(unique_groups)
    val_group_count = max(1, int(round(len(unique_groups) * 0.20)))
    val_groups = set(unique_groups[:val_group_count].tolist())
    val_mask = np.array([group in val_groups for group in groups[indices]], dtype=bool)
    return indices[~val_mask], indices[val_mask]


def _score_candidate(
    model: Pipeline,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[tuple[float, float, float], float, np.ndarray, np.ndarray]:
    model.fit(x_train, y_train)
    val_probabilities = _positive_probabilities(model, x_val)
    threshold, balanced, accuracy = _best_threshold(y_val, val_probabilities)
    val_predictions = _threshold_predictions(val_probabilities, threshold)
    macro_f1 = f1_score(y_val, val_predictions, average="macro", zero_division=0)
    return (balanced, accuracy, macro_f1), threshold, val_predictions, val_probabilities


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    print(f"Using processed data: {args.data_dir}")
    split_counts = validate_prepared_all_idb1(args.data_dir)
    print(f"Validated split counts: {split_counts}")

    records = build_manifest(args.data_dir)
    print(f"Records: {len(records)} {summarize_records(records)}")
    class_names = list(DEFAULT_CLASS_NAMES)
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    labels = np.asarray([class_to_idx[record.label] for record in records], dtype=int)
    groups = (
        np.asarray(acquisition_group_ids(records))
        if args.grouping == "acquisition"
        else np.asarray(content_group_ids(records))
    )
    print(f"Grouping: {args.grouping} ({len(set(groups))} groups)")

    feature_dicts = [
        extract_stain_morphology_features(record.path, max_side=args.max_side)
        for record in records
    ]
    features, feature_names = feature_dicts_to_matrix(feature_dicts)
    print(f"Extracted {features.shape[1]} stain/morphology features from {features.shape[0]} images")

    result_dir = PROJECT_ROOT / "results" / args.result_name
    result_dir.mkdir(parents=True, exist_ok=True)
    feature_frame = pd.DataFrame(features, columns=feature_names)
    feature_frame.insert(0, "label", [record.label for record in records])
    feature_frame.insert(0, "group", groups)
    feature_frame.insert(0, "image_path", [str(record.path) for record in records])
    feature_frame.to_csv(result_dir / "feature_values.csv", index=False)

    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    candidates = _candidate_pipelines(args.seed, features.shape[1])
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []
    all_paths: list[str] = []
    all_test_groups: list[str] = []
    all_embeddings: list[np.ndarray] = []
    all_history: list[dict] = []
    all_inference_times_ms: list[float] = []
    fold_metric_rows: list[dict] = []
    candidate_rows: list[dict] = []
    selected_names: list[str] = []
    selected_thresholds: list[float] = []
    weighted_test_loss = 0.0
    total_test_images = 0

    for fold, (train_val_indices, test_indices) in enumerate(
        splitter.split(np.arange(len(records)), labels, groups),
        start=1,
    ):
        train_indices, val_indices = _inner_train_val_split(
            train_val_indices,
            labels,
            groups,
            args.inner_splits,
            args.seed + fold,
        )
        x_train, y_train = features[train_indices], labels[train_indices]
        x_val, y_val = features[val_indices], labels[val_indices]
        x_test, y_test = features[test_indices], labels[test_indices]

        best_name = ""
        best_score = (-1.0, -1.0, -1.0)
        best_threshold = 0.5
        best_model: Pipeline | None = None
        best_val_predictions: np.ndarray | None = None
        best_val_probabilities: np.ndarray | None = None

        for name, candidate in candidates.items():
            model = clone(candidate)
            try:
                score, threshold, val_predictions, val_probabilities = _score_candidate(
                    model,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                )
            except Exception as exc:
                candidate_rows.append(
                    {
                        "fold": fold,
                        "candidate": name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                continue

            candidate_rows.append(
                {
                    "fold": fold,
                    "candidate": name,
                    "status": "ok",
                    "val_balanced_accuracy": score[0],
                    "val_accuracy": score[1],
                    "val_f1_macro": score[2],
                    "threshold": threshold,
                }
            )
            if score > best_score:
                best_name = name
                best_score = score
                best_threshold = threshold
                best_model = model
                best_val_predictions = val_predictions
                best_val_probabilities = val_probabilities

        if best_model is None or best_val_predictions is None or best_val_probabilities is None:
            raise RuntimeError(f"No candidate model succeeded on fold {fold}")

        train_probabilities = _positive_probabilities(best_model, x_train)
        train_predictions = _threshold_predictions(train_probabilities, best_threshold)
        start_time = time.perf_counter()
        test_probabilities = _positive_probabilities(best_model, x_test)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        test_predictions = _threshold_predictions(test_probabilities, best_threshold)
        test_loss = _safe_log_loss(y_test, test_probabilities)
        fold_accuracy = accuracy_score(y_test, test_predictions)

        fold_metrics = compute_result_metrics(
            y_test.tolist(),
            test_predictions.tolist(),
            test_probabilities,
            class_names,
            test_loss,
            [elapsed_ms / max(len(test_indices), 1)] * len(test_indices),
            bootstrap_samples=0,
            seed=args.seed + fold,
        )
        fold_metrics.update(
            {
                "Fold": fold,
                "Selected_Model": best_name,
                "Decision/Threshold": best_threshold,
                "Decision/Val_Balanced_Accuracy": best_score[0],
                "Decision/Val_Accuracy": best_score[1],
                "Decision/Val_F1_Macro": best_score[2],
                "Evaluation/Test_Groups": len(set(groups[test_indices].tolist())),
                "Training/Train_Accuracy_At_Selected": accuracy_score(y_train, train_predictions),
                "Training/Val_Accuracy_At_Selected": accuracy_score(y_val, best_val_predictions),
                "Training/Generalization_Gap": accuracy_score(y_train, train_predictions)
                - accuracy_score(y_val, best_val_predictions),
            }
        )
        fold_metric_rows.append(fold_metrics)
        all_history.append(
            {
                "fold": fold,
                "epoch": 1,
                "train_loss": _safe_log_loss(y_train, train_probabilities),
                "train_accuracy": accuracy_score(y_train, train_predictions),
                "val_loss": _safe_log_loss(y_val, best_val_probabilities),
                "val_accuracy": accuracy_score(y_val, best_val_predictions),
                "learning_rate": np.nan,
            }
        )
        print(
            f"Fold {fold:02d}/{args.folds}: test_accuracy={fold_accuracy:.4f}, "
            f"selected={best_name}, val_balanced={best_score[0]:.4f}, "
            f"threshold={best_threshold:.4f}",
            flush=True,
        )

        selected_names.append(best_name)
        selected_thresholds.append(best_threshold)
        all_labels.extend(y_test.tolist())
        all_predictions.extend(test_predictions.tolist())
        all_probabilities.append(test_probabilities)
        all_paths.extend(str(records[index].path) for index in test_indices)
        all_test_groups.extend(groups[test_indices].tolist())
        all_embeddings.append(x_test)
        all_inference_times_ms.extend([elapsed_ms / max(len(test_indices), 1)] * len(test_indices))
        weighted_test_loss += test_loss * len(test_indices)
        total_test_images += len(test_indices)

    probabilities = np.concatenate(all_probabilities, axis=0)
    embeddings = np.concatenate(all_embeddings, axis=0)
    final_metrics = compute_result_metrics(
        all_labels,
        all_predictions,
        probabilities,
        class_names,
        weighted_test_loss / total_test_images,
        all_inference_times_ms,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        bootstrap_groups=all_test_groups,
    )
    final_metrics["Evaluation/Folds"] = args.folds
    final_metrics["Training/Pretrained_Weights"] = 0
    final_metrics["Training/Mean_Generalization_Gap"] = float(
        np.mean([row["Training/Generalization_Gap"] for row in fold_metric_rows])
    )
    final_metrics["Decision/Mean_Threshold"] = float(np.mean(selected_thresholds))

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
    pd.DataFrame(candidate_rows).to_csv(result_dir / "candidate_selection.csv", index=False)

    prediction_path = result_dir / "predictions.csv"
    prediction_frame = pd.read_csv(prediction_path)
    prediction_frame.insert(1, f"{args.grouping}_group", all_test_groups)
    prediction_frame.to_csv(prediction_path, index=False)

    best_deployment_name = Counter(selected_names).most_common(1)[0][0]
    deployment_model = clone(candidates[best_deployment_name])
    deployment_model.fit(features, labels)
    deployment_payload = {
        "pipeline": deployment_model,
        "class_names": class_names,
        "feature_names": feature_names,
        "threshold": float(np.mean(selected_thresholds)),
        "candidate_name": best_deployment_name,
        "feature_version": FEATURE_VERSION,
        "max_side": args.max_side,
        "source": "trained on all prepared ALL-IDB1 records after grouped CV evaluation",
    }
    joblib.dump(deployment_payload, result_dir / "deployment_model.joblib")

    (result_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model_family": "label_blind_stain_morphology_classifier",
                "feature_version": FEATURE_VERSION,
                "selected_models": selected_names,
                "deployment_model": best_deployment_name,
                "profile": "white_balance_annotation_suppressed_nuclear_morphology",
                "grouping": args.grouping,
                "folds": args.folds,
                "inner_splits": args.inner_splits,
                "max_side": args.max_side,
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
    print(f"- {result_dir / 'deployment_model.joblib'}")


if __name__ == "__main__":
    main()
