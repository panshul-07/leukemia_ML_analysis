"""Train a three-fold SE-InceptionV3 + dynamic GNN ensemble on C-NMC."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from leukemia_osl.model import create_model
from leukemia_osl.results import compute_result_metrics, save_result_bundle

CLASS_NAMES = ["healthy", "leukemia"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result-name", default="kaggle_se_inception_v3_gnn_3fold_299px_v3")
    parser.add_argument("--image-size", type=int, default=299)
    parser.add_argument("--crop-size", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=7.5e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=7e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--patient-smoothing", type=float, default=0.60)
    parser.add_argument("--target-accuracy", type=float, default=0.93)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def patient_id(value: str) -> str:
    parts = Path(str(value)).stem.split("_")
    return parts[1] if len(parts) >= 2 and parts[0].lower() == "uid" else parts[0]


def load_dataset(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for fold in sorted((root / "training_data").glob("fold_*")):
        for folder, label in (("hem", 0), ("all", 1)):
            for path in sorted((fold / folder).glob("*.bmp")):
                rows.append(
                    {"path": str(path.resolve()), "label": label,
                     "patient_id": patient_id(path.name), "source": fold.name}
                )
    development = pd.DataFrame(rows)
    labels = pd.read_csv(root / "validation_data/C-NMC_test_prelim_phase_data_labels.csv")
    image_dir = root / "validation_data/C-NMC_test_prelim_phase_data"
    test = pd.DataFrame(
        {"path": [str((image_dir / str(name)).resolve()) for name in labels["new_names"]],
         "label": labels["labels"].astype(int),
         "patient_id": [patient_id(value) for value in labels["Patient_ID"]],
         "source": "official_preliminary_test"}
    )
    if len(development) != 10_661 or len(test) != 1_867:
        raise ValueError(f"Expected 10,661 development and 1,867 test images; "
                         f"found {len(development)} and {len(test)}")
    if development.groupby("patient_id")["label"].nunique().max() != 1:
        raise ValueError("A development patient has conflicting labels")
    if set(development.patient_id) & set(test.patient_id):
        raise ValueError("Patient leakage between development and official test")
    for frame in (development, test):
        if not frame.path.map(lambda value: Path(value).is_file()).all():
            raise FileNotFoundError("A C-NMC image listed by the dataset is missing")
    return development, test


def fold_split(development: pd.DataFrame, held_out_fold: str):
    train = development[development.source != held_out_fold].reset_index(drop=True)
    val = development[development.source == held_out_fold].reset_index(drop=True)
    if set(train.patient_id) & set(val.patient_id):
        raise ValueError(f"Patient leakage in {held_out_fold}")
    return train, val


class LeukemiaDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row.label), str(row.path), str(row.patient_id)


def make_transforms(image_size: int, crop_size: int):
    train_transform = transforms.Compose(
        [transforms.CenterCrop(crop_size),
         transforms.RandomResizedCrop(
             image_size, scale=(0.88, 1.0), ratio=(0.95, 1.05),
             interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
         transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
         transforms.RandomAffine(
             degrees=30, translate=(0.03, 0.03), scale=(0.95, 1.05),
             interpolation=transforms.InterpolationMode.BICUBIC, fill=0),
         transforms.ColorJitter(brightness=0.08, contrast=0.10, saturation=0.06),
         transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    eval_transform = transforms.Compose(
        [transforms.CenterCrop(crop_size),
         transforms.Resize((image_size, image_size), antialias=True),
         transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    return train_transform, eval_transform


def make_loaders(train, val, test, image_size, crop_size, batch_size, seed, num_workers):
    train_transform, eval_transform = make_transforms(image_size, crop_size)
    options = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available(),
               "persistent_workers": num_workers > 0}
    return {
        "train": DataLoader(
            LeukemiaDataset(train, train_transform), batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed), **options),
        "val": DataLoader(
            LeukemiaDataset(val, eval_transform), batch_size=batch_size, shuffle=False, **options),
        "test": DataLoader(
            LeukemiaDataset(test, eval_transform), batch_size=batch_size, shuffle=False, **options),
    }


def model_config(pretrained: bool) -> dict:
    return {"name": "hybrid_cnn_gnn", "cnn_backbone_name": "inception_v3",
            "pretrained_backbone": pretrained, "embed_dim": 128, "gnn_layers": 2,
            "gnn_dropout": 0.15, "graph_neighbors": 8, "dropout": 0.35,
            "channel_attention": True}


def class_weights(frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = frame.label.value_counts().sort_index().to_numpy(dtype=np.float32)
    weights = np.sqrt(counts.sum() / (len(counts) * counts))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def tta_views(images: torch.Tensor) -> list[torch.Tensor]:
    return [images, torch.flip(images, (-1,)), torch.flip(images, (-2, -1))]


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_tta=False, collect_embeddings=False):
    model.eval()
    labels, paths, groups, probabilities, embeddings = [], [], [], [], []
    weighted_loss = 0.0
    started = time.perf_counter()
    for images, targets, batch_paths, batch_groups in loader:
        images, targets = images.to(device), targets.to(device)
        views = tta_views(images) if use_tta else [images]
        view_probabilities, view_embeddings = [], []
        for view in views:
            if collect_embeddings:
                logits, features = model.forward_with_features(view)
                view_embeddings.append(features)
            else:
                logits = model(view)
            view_probabilities.append(logits.softmax(dim=1))
        probs = torch.stack(view_probabilities).mean(0)
        weighted_loss += float(criterion(probs.clamp_min(1e-8).log(), targets)) * len(targets)
        labels.extend(targets.cpu().tolist())
        probabilities.append(probs.cpu().numpy())
        paths.extend(batch_paths)
        groups.extend(batch_groups)
        if collect_embeddings:
            embeddings.append(torch.stack(view_embeddings).mean(0).cpu().numpy())
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {"loss": weighted_loss / len(labels), "labels": labels,
            "probabilities": np.concatenate(probabilities), "paths": paths, "groups": groups,
            "embeddings": np.concatenate(embeddings) if embeddings else None,
            "inference_times": [elapsed_ms / len(labels)] * len(labels)}


def train_epoch(model, loader, criterion, optimizer, device, fold: int, epoch: int):
    model.train()
    total_loss = 0.0
    correct = total = 0
    for batch_index, (images, targets, _, _) in enumerate(loader, 1):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach()) * len(targets)
        correct += int((logits.argmax(1) == targets).sum())
        total += len(targets)
        if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
            print(f"fold {fold}/3 epoch {epoch:02d} batch {batch_index:04d}/{len(loader):04d}: "
                  f"running_accuracy={correct / total:.4f}", flush=True)
    return total_loss / total, correct / total


def patient_smooth(probabilities: np.ndarray, groups: list[str], weight: float):
    positive = probabilities[:, 1]
    frame = pd.DataFrame({"group": groups, "positive": positive})
    group_mean = frame.groupby("group").positive.transform("mean").to_numpy()
    adjusted = (1.0 - weight) * positive + weight * group_mean
    return np.column_stack((1.0 - adjusted, adjusted))


def validation_threshold(probabilities: np.ndarray, labels: list[int]) -> float:
    positive = probabilities[:, 1]
    labels_array = np.asarray(labels)
    thresholds = np.unique(np.concatenate(([0.0], positive, [1.0])))
    best = (-1.0, -1.0, -1.0, 0.5)
    for threshold in thresholds:
        predictions = positive >= threshold
        balanced = balanced_accuracy_score(labels_array, predictions)
        accuracy = float((predictions == labels_array).mean())
        candidate = (balanced, accuracy, -abs(float(threshold) - 0.5), float(threshold))
        if candidate > best:
            best = candidate
    return best[3]


def apply_threshold_calibration(probabilities: np.ndarray, threshold: float):
    positive = np.clip(probabilities[:, 1], 1e-6, 1 - 1e-6)
    threshold = float(np.clip(threshold, 1e-6, 1 - 1e-6))
    offset = np.log(threshold / (1.0 - threshold))
    calibrated = 1.0 / (1.0 + np.exp(-(np.log(positive / (1.0 - positive)) - offset)))
    return np.column_stack((1.0 - calibrated, calibrated))


def train_fold(args, fold_index, held_out, development, test, device, result_dir):
    seed_everything(args.seed + fold_index)
    train, val = fold_split(development, held_out)
    loaders = make_loaders(
        train, val, test, args.image_size, args.crop_size, args.batch_size,
        args.seed + fold_index, args.num_workers)
    config = model_config(not args.no_pretrained)
    model = create_model(2, **config).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train, device), label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [{"params": model.cnn_backbone.parameters(),
          "lr": args.lr * args.backbone_lr_scale},
         {"params": [parameter for name, parameter in model.named_parameters()
                     if not name.startswith("cnn_backbone.")], "lr": args.lr}],
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.35, patience=2, min_lr=1e-6)
    history = []
    best_score = -1.0
    best_epoch = 0
    stale = 0
    checkpoint_path = result_dir / f"fold_{fold_index}_best.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model, loaders["train"], criterion, optimizer, device, fold_index, epoch)
        validation = evaluate(model, loaders["val"], eval_criterion, device)
        val_predictions = validation["probabilities"].argmax(1)
        val_accuracy = float(np.mean(val_predictions == np.asarray(validation["labels"])))
        val_balanced = balanced_accuracy_score(validation["labels"], val_predictions)
        scheduler.step(val_balanced)
        history.append(
            {"fold": fold_index, "epoch": epoch, "train_loss": train_loss,
             "train_accuracy": train_accuracy, "val_loss": validation["loss"],
             "val_accuracy": val_accuracy, "val_balanced_accuracy": val_balanced,
             "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"fold {fold_index}/3 epoch {epoch:02d}/{args.epochs}: "
              f"train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f} "
              f"val_balanced={val_balanced:.4f}", flush=True)
        if val_balanced > best_score + 1e-4:
            best_score, best_epoch, stale = val_balanced, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "model_config": config},
                       checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"fold {fold_index}: early stop; best epoch {best_epoch}", flush=True)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = evaluate(model, loaders["val"], eval_criterion, device, use_tta=True)
    smoothed_validation = patient_smooth(
        validation["probabilities"], validation["groups"], args.patient_smoothing)
    threshold = validation_threshold(smoothed_validation, validation["labels"])
    test_output = evaluate(
        model, loaders["test"], eval_criterion, device,
        use_tta=True, collect_embeddings=True)
    smoothed_test = patient_smooth(
        test_output["probabilities"], test_output["groups"], args.patient_smoothing)
    calibrated_test = apply_threshold_calibration(smoothed_test, threshold)
    return history, best_epoch, best_score, threshold, calibrated_test, test_output


def aggregate_history(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rows)
    columns = ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    return frame.groupby("epoch", as_index=False)[columns].mean().to_dict("records")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    root = args.data_dir.expanduser().resolve()
    development, test = load_dataset(root)
    pretrained = not args.no_pretrained
    print(f"Device: {device}; development/test images: {len(development)}/{len(test)}; "
          f"patients: {development.patient_id.nunique()}/{test.patient_id.nunique()}",
          flush=True)
    if args.dry_run:
        train, val = fold_split(development, "fold_0")
        loaders = make_loaders(
            train, val, test, args.image_size, args.crop_size, 2, args.seed, 0)
        model = create_model(2, **model_config(False)).to(device)
        images, targets, _, _ = next(iter(loaders["train"]))
        logits = model(images.to(device))
        nn.functional.cross_entropy(logits, targets.to(device)).backward()
        print(f"Dry run passed: input={tuple(images.shape)}, logits={tuple(logits.shape)}")
        return

    result_dir = ROOT / "results" / args.result_name
    result_dir.mkdir(parents=True, exist_ok=True)
    development.to_csv(result_dir / "development_split.csv", index=False)
    test.to_csv(result_dir / "test_split.csv", index=False)
    (result_dir / "run_config.json").write_text(
        json.dumps({"model": model_config(pretrained), **vars(args), "data_dir": str(root),
                    "protocol": "three predefined subject-disjoint folds; official test ensemble"},
                   indent=2, default=str))

    all_history, probabilities, embeddings, fold_rows = [], [], [], []
    final_output = None
    for fold_index, held_out in enumerate(("fold_0", "fold_1", "fold_2"), 1):
        history, best_epoch, best_score, threshold, fold_probs, output = train_fold(
            args, fold_index, held_out, development, test, device, result_dir)
        all_history.extend(history)
        probabilities.append(fold_probs)
        embeddings.append(output["embeddings"])
        final_output = output
        fold_rows.append(
            {"Fold": fold_index, "Held_Out": held_out, "Best_Epoch": best_epoch,
             "Validation_Balanced_Accuracy": best_score,
             "Validation_Threshold": threshold})
        pd.DataFrame(all_history).to_csv(
            result_dir / "fold_training_history_live.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(result_dir / "fold_metrics_live.csv", index=False)

    assert final_output is not None
    ensemble_probabilities = np.mean(probabilities, axis=0)
    ensemble_embeddings = np.mean(embeddings, axis=0)
    predictions = ensemble_probabilities.argmax(1).tolist()
    metrics = compute_result_metrics(
        final_output["labels"], predictions, ensemble_probabilities, CLASS_NAMES,
        final_output["loss"], final_output["inference_times"], args.seed,
        bootstrap_groups=final_output["groups"])
    metrics.update(
        {"Training/Folds": 3, "Training/Pretrained_Weights": int(pretrained),
         "Decision/Threshold": 0.5,
         "Decision/Policy": "mean of three validation-calibrated SE-InceptionV3-GNN folds",
         "Decision/Patient_Smoothing": args.patient_smoothing,
         "Target/Accuracy": args.target_accuracy,
         "Target/Reached": bool(metrics["Final/Accuracy"] >= args.target_accuracy)})
    pd.DataFrame(all_history).to_csv(result_dir / "fold_training_history.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(result_dir / "fold_metrics.csv", index=False)
    save_result_bundle(
        aggregate_history(all_history), metrics, final_output["labels"], predictions,
        ensemble_probabilities, CLASS_NAMES, result_dir,
        image_paths=final_output["paths"], embeddings=ensemble_embeddings)
    print(json.dumps(metrics, indent=2))
    print(f"Results saved to {result_dir}")


if __name__ == "__main__":
    main()
