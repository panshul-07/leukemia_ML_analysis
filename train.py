from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from leukemia_osl.data import estimate_channel_stats, make_imagefolder_loaders
from leukemia_osl.model import create_model
from leukemia_osl.preprocess import validate_prepared_all_idb1
from leukemia_osl.results import compute_result_metrics, save_result_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ALL-IDB1 model from scratch.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--model",
        choices=["resnet18_osl", "hybrid_cnn_transformer", "hybrid_cnn_gnn"],
        default=None,
    )
    parser.add_argument(
        "--cnn-backbone-name",
        choices=[
            "efficientnet_b0",
            "mobilenet_v3_large",
            "mobilenet_v2",
            "resnet18",
            "vggnet",
            "vgg11_bn",
            "vgg16_bn",
            "vgg19_bn",
        ],
        default=None,
    )
    parser.add_argument(
        "--preprocessing-profile",
        choices=[
            "paper",
            "hybrid",
            "hybrid_microscopy",
            "hybrid_segmented_pretrained",
            "gnn_microscopy_enhanced",
        ],
        default=None,
    )
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--gnn-layers", type=int, default=None)
    parser.add_argument("--gnn-dropout", type=float, default=None)
    parser.add_argument("--graph-neighbors", type=int, choices=[4, 8], default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    max_grad_norm: float | None = None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="train", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    collect_embeddings: bool = False,
    measure_inference: bool = False,
    test_time_augmentation: bool = False,
) -> tuple[
    float,
    float,
    list[int],
    list[int],
    np.ndarray,
    np.ndarray | None,
    list[float],
]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    inference_times_ms: list[float] = []

    for inputs, labels in tqdm(loader, desc="eval", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        if measure_inference and device.type == "cuda":
            torch.cuda.synchronize()
        if measure_inference and device.type == "mps":
            torch.mps.synchronize()
        start_time = time.perf_counter()
        views = [inputs]
        if test_time_augmentation:
            views.extend(
                [
                    torch.flip(inputs, dims=(-1,)),
                    torch.flip(inputs, dims=(-2,)),
                    torch.flip(inputs, dims=(-2, -1)),
                ]
            )
        view_logits = []
        view_embeddings = []
        for view in views:
            if collect_embeddings:
                logits, embeddings = model.forward_with_features(view)
                view_embeddings.append(embeddings)
            else:
                logits = model(view)
            view_logits.append(logits)
        logits = torch.stack(view_logits).mean(dim=0)
        if collect_embeddings:
            embeddings = torch.stack(view_embeddings).mean(dim=0)
            all_embeddings.append(embeddings.cpu().numpy())
        if measure_inference and device.type == "cuda":
            torch.cuda.synchronize()
        if measure_inference and device.type == "mps":
            torch.mps.synchronize()
        if measure_inference:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            inference_times_ms.extend([elapsed_ms / inputs.size(0)] * inputs.size(0))
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

        running_loss += loss.item() * inputs.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.append(probabilities.cpu().numpy())

    embeddings_array = (
        np.concatenate(all_embeddings, axis=0) if all_embeddings else None
    )
    return (
        running_loss / total,
        correct / total,
        all_labels,
        all_predictions,
        np.concatenate(all_probabilities, axis=0),
        embeddings_array,
        inference_times_ms,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training_cfg = config["training"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    if args.data_dir is not None:
        data_cfg["processed_dir"] = str(args.data_dir)
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        training_cfg["learning_rate"] = args.lr
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers
    if args.checkpoint_dir is not None:
        training_cfg["checkpoint_dir"] = str(args.checkpoint_dir)
    if args.report_dir is not None:
        training_cfg["report_dir"] = str(args.report_dir)
    if args.model is not None:
        model_cfg["name"] = args.model
    if args.cnn_backbone_name is not None:
        model_cfg["cnn_backbone_name"] = args.cnn_backbone_name
    if args.label_smoothing is not None:
        training_cfg["label_smoothing"] = args.label_smoothing
    if args.max_grad_norm is not None:
        training_cfg["max_grad_norm"] = args.max_grad_norm
    if args.gnn_layers is not None:
        model_cfg["gnn_layers"] = args.gnn_layers
    if args.gnn_dropout is not None:
        model_cfg["gnn_dropout"] = args.gnn_dropout
    if args.graph_neighbors is not None:
        model_cfg["graph_neighbors"] = args.graph_neighbors
    model_cfg["pretrained_backbone"] = False
    model_cfg["freeze_backbone"] = False

    model_name = model_cfg.get("name", "resnet18_osl")
    preprocessing_profile = args.preprocessing_profile or (
        "hybrid" if model_name in {"hybrid_cnn_transformer", "hybrid_cnn_gnn"} else "paper"
    )
    data_cfg["preprocessing_profile"] = preprocessing_profile

    set_seed(int(training_cfg["seed"]))
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    processed_dir = Path(data_cfg["processed_dir"])
    split_counts = validate_prepared_all_idb1(processed_dir)
    print(f"Validated real ALL-IDB1 split counts: {split_counts}")
    mean, std = estimate_channel_stats(
        processed_dir / "train",
        image_size=int(data_cfg["image_size"]),
        num_workers=int(data_cfg["num_workers"]),
    )
    data_cfg["normalization_mean"] = list(mean)
    data_cfg["normalization_std"] = list(std)
    print(f"Training-split RGB mean: {mean}")
    print(f"Training-split RGB std: {std}")

    loaders, class_names = make_imagefolder_loaders(
        processed_dir=processed_dir,
        image_size=int(data_cfg["image_size"]),
        batch_size=int(training_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        profile=preprocessing_profile,
        mean=mean,
        std=std,
    )
    print(f"Classes: {class_names}")

    model = create_model(num_classes=len(class_names), **model_cfg).to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0))
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_cfg.get("lr_reduce_factor", 0.5)),
        patience=int(training_cfg.get("lr_reduce_patience", 6)),
        min_lr=float(training_cfg.get("min_learning_rate", 1e-6)),
    )

    checkpoint_dir = Path(training_cfg["checkpoint_dir"])
    report_dir = Path(training_cfg["report_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"best_{model_name}.pth"

    best_val_loss = float("inf")
    best_val_accuracy = -1.0
    patience = int(training_cfg["patience"])
    min_delta = float(training_cfg.get("min_delta", 1e-4))
    stale_epochs = 0
    history: list[dict] = []
    max_grad_norm = training_cfg.get("max_grad_norm")
    max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            max_grad_norm=max_grad_norm,
        )
        val_loss, val_acc, _, _, _, _, _ = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
        )
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        generalization_gap = train_acc - val_acc
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "generalization_gap": generalization_gap,
            "learning_rate": current_lr,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"gap={generalization_gap:.4f} lr={current_lr:.2e}"
        )

        improved = val_loss < best_val_loss - min_delta
        tied_loss_better_accuracy = (
            abs(val_loss - best_val_loss) <= min_delta and val_acc > best_val_accuracy
        )
        if improved or tied_loss_better_accuracy:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "config": config,
                },
                best_path,
            )
            print(f"Saved improved checkpoint to {best_path}")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(
                    f"Early stopping after {patience} epochs without validation-loss improvement."
                )
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    (
        test_loss,
        test_acc,
        labels,
        predictions,
        probabilities,
        embeddings,
        inference_times_ms,
    ) = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        collect_embeddings=True,
        measure_inference=True,
    )
    metrics = compute_result_metrics(
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        class_names=class_names,
        test_loss=test_loss,
        inference_times_ms=inference_times_ms,
        seed=int(training_cfg["seed"]),
    )
    metrics["Final/Test_Accuracy_Direct"] = test_acc

    test_dataset = loaders["test"].dataset
    test_image_paths = [sample[0] for sample in getattr(test_dataset, "samples", [])]
    saved_paths = save_result_bundle(
        history=history,
        metrics=metrics,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        class_names=class_names,
        report_dir=report_dir,
        image_paths=test_image_paths,
        embeddings=embeddings,
    )

    print("Test metrics:")
    print(json.dumps(metrics, indent=2))
    print("Saved result files:")
    for path in saved_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
