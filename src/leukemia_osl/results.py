"""Persist training and evaluation outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "leukemia_osl_matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def _bootstrap_interval(
    labels: np.ndarray,
    predictions: np.ndarray,
    scorer,
    seed: int,
    samples: int,
    groups: np.ndarray | None = None,
) -> tuple[float, float]:
    if len(labels) < 2 or samples < 1:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    values: list[float] = []
    unique_groups = np.unique(groups) if groups is not None else None
    for _ in range(samples):
        if unique_groups is None:
            indices = rng.integers(0, len(labels), size=len(labels))
        else:
            sampled_groups = rng.choice(
                unique_groups,
                size=len(unique_groups),
                replace=True,
            )
            indices = np.concatenate(
                [np.flatnonzero(groups == group) for group in sampled_groups]
            )
        try:
            value = float(scorer(labels[indices], predictions[indices]))
        except ValueError:
            continue
        if np.isfinite(value):
            values.append(value)

    if not values:
        return float("nan"), float("nan")
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def compute_result_metrics(
    labels: list[int],
    predictions: list[int],
    probabilities: np.ndarray,
    class_names: list[str],
    test_loss: float,
    inference_times_ms: list[float] | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    bootstrap_groups: list[str] | None = None,
) -> dict[str, float | int]:
    """Compute the full reference-style result table from actual test predictions."""

    labels_array = np.asarray(labels, dtype=int)
    predictions_array = np.asarray(predictions, dtype=int)
    groups_array = (
        np.asarray(bootstrap_groups)
        if bootstrap_groups is not None
        else None
    )
    if groups_array is not None and len(groups_array) != len(labels_array):
        raise ValueError("bootstrap_groups must align with labels.")
    class_indices = list(range(len(class_names)))
    one_hot_labels = np.eye(len(class_names), dtype=int)[labels_array]
    positive_index = (
        class_names.index("leukemia") if "leukemia" in class_names else len(class_names) - 1
    )

    accuracy = accuracy_score(labels_array, predictions_array)
    f1_macro = f1_score(
        labels_array,
        predictions_array,
        labels=class_indices,
        average="macro",
        zero_division=0,
    )
    mcc = matthews_corrcoef(labels_array, predictions_array)
    accuracy_ci = _bootstrap_interval(
        labels_array,
        predictions_array,
        accuracy_score,
        seed,
        bootstrap_samples,
        groups_array,
    )
    f1_ci = _bootstrap_interval(
        labels_array,
        predictions_array,
        lambda y_true, y_pred: f1_score(
            y_true,
            y_pred,
            labels=class_indices,
            average="macro",
            zero_division=0,
        ),
        seed + 1,
        bootstrap_samples,
        groups_array,
    )
    mcc_ci = _bootstrap_interval(
        labels_array,
        predictions_array,
        matthews_corrcoef,
        seed + 2,
        bootstrap_samples,
        groups_array,
    )

    negatives = labels_array != positive_index
    true_negatives = ((predictions_array != positive_index) & negatives).sum()
    specificity = float(true_negatives / negatives.sum()) if negatives.any() else float("nan")
    positives = labels_array == positive_index
    true_positives = ((predictions_array == positive_index) & positives).sum()
    sensitivity = float(true_positives / positives.sum()) if positives.any() else float("nan")

    metrics: dict[str, float | int] = {
        "Final/Accuracy": float(accuracy),
        "Final/Accuracy_CI_Lower": accuracy_ci[0],
        "Final/Accuracy_CI_Upper": accuracy_ci[1],
        "Final/Balanced_Accuracy": float(
            balanced_accuracy_score(labels_array, predictions_array)
        ),
        "Final/Precision_Macro": float(
            precision_score(
                labels_array,
                predictions_array,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "Final/Precision_Micro": float(
            precision_score(labels_array, predictions_array, average="micro", zero_division=0)
        ),
        "Final/Recall_Macro": float(
            recall_score(
                labels_array,
                predictions_array,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "Final/Recall_Micro": float(
            recall_score(labels_array, predictions_array, average="micro", zero_division=0)
        ),
        "Final/F1_Macro": float(f1_macro),
        "Final/F1_Macro_CI_Lower": f1_ci[0],
        "Final/F1_Macro_CI_Upper": f1_ci[1],
        "Final/F1_Micro": float(
            f1_score(labels_array, predictions_array, average="micro", zero_division=0)
        ),
        "Final/MCC": float(mcc),
        "Final/MCC_CI_Lower": mcc_ci[0],
        "Final/MCC_CI_Upper": mcc_ci[1],
        "Final/Cohen_Kappa": float(cohen_kappa_score(labels_array, predictions_array)),
        "Final/Sensitivity": sensitivity,
        "Final/Specificity": specificity,
        "Final/Test_Loss": float(test_loss),
        "Final/Test_Samples": int(len(labels_array)),
        "Final/Bootstrap_Groups": int(len(np.unique(groups_array)))
        if groups_array is not None
        else int(len(labels_array)),
        "Final/Avg_Inference_Time_ms": float(np.mean(inference_times_ms))
        if inference_times_ms
        else float("nan"),
    }

    try:
        metrics["Final/ROC_AUC_Macro"] = float(
            roc_auc_score(one_hot_labels, probabilities, average="macro")
        )
        metrics["Final/ROC_AUC_Micro"] = float(
            roc_auc_score(one_hot_labels, probabilities, average="micro")
        )
        metrics["Final/PR_AUC_Macro"] = float(
            average_precision_score(one_hot_labels, probabilities, average="macro")
        )
        metrics["Final/PR_AUC_Micro"] = float(
            average_precision_score(one_hot_labels, probabilities, average="micro")
        )
    except ValueError:
        metrics["Final/ROC_AUC_Macro"] = float("nan")
        metrics["Final/ROC_AUC_Micro"] = float("nan")
        metrics["Final/PR_AUC_Macro"] = float("nan")
        metrics["Final/PR_AUC_Micro"] = float("nan")

    for class_index, class_name in enumerate(class_names):
        metric_name = class_name.replace(" ", "_").title()
        binary_labels = one_hot_labels[:, class_index]
        try:
            metrics[f"Final/ROC_AUC_{metric_name}"] = float(
                roc_auc_score(binary_labels, probabilities[:, class_index])
            )
            metrics[f"Final/PR_AUC_{metric_name}"] = float(
                average_precision_score(binary_labels, probabilities[:, class_index])
            )
        except ValueError:
            metrics[f"Final/ROC_AUC_{metric_name}"] = float("nan")
            metrics[f"Final/PR_AUC_{metric_name}"] = float("nan")
    return metrics


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_summary(path: Path, metrics: dict) -> None:
    summary = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    )
    path.write_text(summary.to_string(index=False), encoding="utf-8")


def _save_predictions(
    path: Path,
    labels: list[int],
    predictions: list[int],
    probabilities: np.ndarray,
    class_names: list[str],
    image_paths: list[str] | None,
) -> None:
    rows = []
    for index, (label, prediction, class_probabilities) in enumerate(
        zip(labels, predictions, probabilities, strict=True)
    ):
        row = {
            "image_path": image_paths[index] if image_paths and index < len(image_paths) else "",
            "true_label": class_names[label],
            "predicted_label": class_names[prediction],
            "is_correct": label == prediction,
        }
        for class_name, probability in zip(class_names, class_probabilities, strict=True):
            row[f"probability_{class_name}"] = probability
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_training_curves(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="Train")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    path: Path,
    title: str,
    fmt: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_classification_heatmap(
    report_df: pd.DataFrame,
    class_names: list[str],
    path: Path,
) -> None:
    metrics = ["precision", "recall", "f1-score"]
    heatmap_data = report_df.loc[class_names, metrics].astype(float)
    fig, ax = plt.subplots(figsize=(7, max(3, len(class_names) * 0.7)))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Per-class Metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_tsne(
    embeddings: np.ndarray,
    labels: list[int],
    class_names: list[str],
    plot_path: Path,
    csv_path: Path,
) -> None:
    if len(embeddings) < 4:
        return

    perplexity = min(30, max(2, (len(embeddings) - 1) // 3))
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
        n_jobs=1,
    ).fit_transform(embeddings)
    frame = pd.DataFrame(
        {
            "tsne_1": coordinates[:, 0],
            "tsne_2": coordinates[:, 1],
            "label": [class_names[label] for label in labels],
        }
    )
    frame.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=frame,
        x="tsne_1",
        y="tsne_2",
        hue="label",
        style="label",
        s=70,
        ax=ax,
    )
    ax.set_title("t-SNE Visualization of Test Embeddings")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def _plot_roc_curves(
    labels: list[int],
    probabilities: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    labels_array = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(6, 5))

    plotted = False
    if len(class_names) == 2:
        positive_index = (
            class_names.index("leukemia") if "leukemia" in class_names else len(class_names) - 1
        )
        positive_labels = (labels_array == positive_index).astype(int)
        if len(np.unique(positive_labels)) == 2:
            fpr, tpr, _ = roc_curve(positive_labels, probabilities[:, positive_index])
            ax.plot(fpr, tpr, label=class_names[positive_index])
            plotted = True
    else:
        binarized = label_binarize(labels_array, classes=list(range(len(class_names))))
        for class_index, class_name in enumerate(class_names):
            if len(np.unique(binarized[:, class_index])) < 2:
                continue
            fpr, tpr, _ = roc_curve(binarized[:, class_index], probabilities[:, class_index])
            ax.plot(fpr, tpr, label=class_name)
            plotted = True

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Not enough class variation for ROC", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_pr_curves(
    labels: list[int],
    probabilities: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    labels_array = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    plotted = False

    if len(class_names) == 2:
        positive_index = (
            class_names.index("leukemia") if "leukemia" in class_names else len(class_names) - 1
        )
        positive_labels = (labels_array == positive_index).astype(int)
        if len(np.unique(positive_labels)) == 2:
            precision, recall, _ = precision_recall_curve(
                positive_labels,
                probabilities[:, positive_index],
            )
            ap = average_precision_score(positive_labels, probabilities[:, positive_index])
            ax.plot(recall, precision, label=f"{class_names[positive_index]} AP={ap:.3f}")
            plotted = True
    else:
        binarized = label_binarize(labels_array, classes=list(range(len(class_names))))
        for class_index, class_name in enumerate(class_names):
            if len(np.unique(binarized[:, class_index])) < 2:
                continue
            precision, recall, _ = precision_recall_curve(
                binarized[:, class_index],
                probabilities[:, class_index],
            )
            ap = average_precision_score(binarized[:, class_index], probabilities[:, class_index])
            ax.plot(recall, precision, label=f"{class_name} AP={ap:.3f}")
            plotted = True

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    if plotted:
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "Not enough class variation for precision-recall",
            ha="center",
            va="center",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_result_bundle(
    history: list[dict],
    metrics: dict,
    labels: list[int],
    predictions: list[int],
    probabilities: np.ndarray,
    class_names: list[str],
    report_dir: Path,
    image_paths: list[str] | None = None,
    embeddings: np.ndarray | None = None,
) -> list[Path]:
    report_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv(report_dir / "training_history.csv", index=False)

    report_dict = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(report_dir / "classification_metrics.csv")
    report_df.to_csv(report_dir / "classification_report.csv")
    (report_dir / "classification_report.txt").write_text(
        classification_report(
            labels,
            predictions,
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
        encoding="utf-8",
    )

    final_metrics = pd.DataFrame([metrics])
    final_metrics.to_csv(report_dir / "final_metrics.csv", index=False)
    _save_json(report_dir / "metrics.json", metrics)
    _save_summary(report_dir / "summary_table.txt", metrics)
    _save_predictions(
        report_dir / "predictions.csv",
        labels,
        predictions,
        probabilities,
        class_names,
        image_paths,
    )

    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    matrix_df = pd.DataFrame(matrix, index=class_names, columns=class_names)
    matrix_df.to_csv(report_dir / "confusion_matrix_raw.csv")

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )
    pd.DataFrame(normalized, index=class_names, columns=class_names).to_csv(
        report_dir / "confusion_matrix_normalized.csv"
    )

    _plot_training_curves(history_df, report_dir / "training_curves.png")
    _plot_confusion_matrix(
        matrix,
        class_names,
        report_dir / "confusion_matrix_raw.png",
        "Confusion Matrix",
        "d",
    )
    _plot_confusion_matrix(
        normalized,
        class_names,
        report_dir / "confusion_matrix_normalized.png",
        "Normalized Confusion Matrix",
        ".2f",
    )
    _plot_confusion_matrix(
        matrix,
        class_names,
        report_dir / "confusion_matrix.png",
        "Confusion Matrix",
        "d",
    )
    _plot_confusion_matrix(
        normalized,
        class_names,
        report_dir / "confusion_matrix_norm.png",
        "Normalized Confusion Matrix",
        ".2f",
    )
    _plot_classification_heatmap(
        report_df,
        class_names,
        report_dir / "classification_heatmap.png",
    )
    _plot_classification_heatmap(
        report_df,
        class_names,
        report_dir / "classification_report.png",
    )
    _plot_roc_curves(labels, probabilities, class_names, report_dir / "roc_curves.png")
    _plot_roc_curves(labels, probabilities, class_names, report_dir / "roc_curve.png")
    _plot_pr_curves(labels, probabilities, class_names, report_dir / "pr_curves.png")
    if embeddings is not None:
        pd.DataFrame(embeddings).to_csv(report_dir / "test_embeddings.csv", index=False)
        _plot_tsne(
            embeddings,
            labels,
            class_names,
            report_dir / "tsne_visualization.png",
            report_dir / "tsne_coordinates.csv",
        )
    return sorted(path for path in report_dir.iterdir() if path.is_file())
