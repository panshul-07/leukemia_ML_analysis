from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from leukemia_osl.data import _base_preprocessing, build_transforms
from leukemia_osl.gradcam import GradCAM, heatmap_to_rgb, overlay_heatmap
from leukemia_osl.model import create_model
from leukemia_osl.preprocess import (
    DEFAULT_CLASS_NAMES,
    acquisition_group_ids,
    build_manifest,
    content_group_ids,
    validate_prepared_all_idb1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for saved folds.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed/all_idb1")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-result-dir", type=Path, default=None)
    parser.add_argument("--result-name", type=str, default="gradcam_heatmaps")
    parser.add_argument("--grouping", choices=["acquisition", "content"], default="acquisition")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--target-class",
        choices=["predicted", "true", "leukemia", "healthy"],
        default="predicted",
    )
    parser.add_argument("--only-errors", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--fold", type=int, action="append", default=None)
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


def load_saved_predictions(path: Path | None) -> dict[str, dict] | None:
    if path is None:
        return None
    predictions_path = path / "predictions.csv"
    if not predictions_path.exists():
        return None
    frame = pd.read_csv(predictions_path)
    if "image_path" not in frame.columns:
        return None
    return frame.set_index("image_path").to_dict(orient="index")


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


def records_for_groups(records, groups: np.ndarray, selected_groups: list[str]):
    selected = set(selected_groups)
    return [
        (record, group)
        for record, group in zip(records, groups, strict=True)
        if group in selected
    ]


def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("/", "_")


def select_target_class(mode: str, true_index: int, predicted_index: int) -> int:
    if mode == "predicted":
        return predicted_index
    if mode == "true":
        return true_index
    return DEFAULT_CLASS_NAMES.index(mode)


def load_model(checkpoint: dict, device: torch.device):
    class_names = list(checkpoint["class_names"])
    model_config = infer_backbone_from_state_dict(checkpoint, dict(checkpoint["model_config"]))
    model_config["pretrained_backbone"] = False
    model = create_model(num_classes=len(class_names), **model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if not hasattr(model, "cnn_backbone"):
        raise ValueError("Grad-CAM script currently targets hybrid CNN checkpoints.")
    return model, class_names, model_config


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    source_result_dir = args.source_result_dir.expanduser().resolve() if args.source_result_dir else None
    run_config = load_run_config(source_result_dir)
    saved_predictions = load_saved_predictions(source_result_dir)
    profile = args.profile or run_config.get("preprocessing_profile") or run_config.get("profile") or "hybrid"
    image_size = args.image_size or int(run_config.get("image_size", 224))
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    output_dir = PROJECT_ROOT / "results" / args.result_name
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_prepared_all_idb1(args.data_dir)
    records = build_manifest(args.data_dir)
    groups = (
        np.asarray(acquisition_group_ids(records))
        if args.grouping == "acquisition"
        else np.asarray(content_group_ids(records))
    )
    class_to_idx = {name: index for index, name in enumerate(DEFAULT_CLASS_NAMES)}
    preprocessing = transforms.Compose(_base_preprocessing(image_size, profile))
    tensor_transform = build_transforms(
        image_size=image_size,
        train=False,
        profile=profile,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        preprocessed=True,
    )

    checkpoint_paths = sorted(checkpoint_dir.glob("fold_*_best.pth"))
    if args.fold is not None:
        selected_folds = set(args.fold)
        checkpoint_paths = [
            path
            for path in checkpoint_paths
            if int(path.stem.split("_")[1]) in selected_folds
        ]
    if not checkpoint_paths:
        raise FileNotFoundError(f"No matching checkpoints found in {checkpoint_dir}")

    rows: list[dict] = []
    generated = 0
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        fold = int(checkpoint.get("fold", checkpoint_path.stem.split("_")[1]))
        model, class_names, model_config = load_model(checkpoint, device)
        tensor_transform.transforms[-1] = transforms.Normalize(
            mean=tuple(checkpoint["normalization_mean"]),
            std=tuple(checkpoint["normalization_std"]),
        )
        fold_records = records_for_groups(records, groups, checkpoint["test_groups"])
        with GradCAM(model, model.cnn_backbone) as gradcam:
            for record, group in fold_records:
                with Image.open(record.path) as source:
                    model_image = preprocessing(source.convert("RGB"))
                input_tensor = tensor_transform(model_image).unsqueeze(0).to(device)
                result = gradcam(input_tensor)
                probabilities = result.probabilities
                raw_predicted_index = int(np.argmax(probabilities))
                predicted_index = raw_predicted_index
                true_index = class_to_idx[record.label]
                saved_prediction = None
                if saved_predictions is not None:
                    saved_prediction = saved_predictions.get(str(record.path))
                    if saved_prediction is not None:
                        predicted_index = class_names.index(saved_prediction["predicted_label"])
                is_correct = (
                    bool(saved_prediction["is_correct"])
                    if saved_prediction is not None
                    else predicted_index == true_index
                )
                if args.only_errors and is_correct:
                    continue
                target_index = select_target_class(
                    args.target_class,
                    true_index=true_index,
                    predicted_index=predicted_index,
                )
                if target_index != result.class_index:
                    result = gradcam(input_tensor, class_index=target_index)
                    probabilities = result.probabilities
                    predicted_index = int(np.argmax(probabilities))

                prefix = (
                    f"fold_{fold:02d}_{safe_name(record.path)}"
                    f"_true-{record.label}_pred-{class_names[predicted_index]}"
                    f"_target-{class_names[target_index]}"
                )
                input_path = output_dir / f"{prefix}_input.png"
                heatmap_path = output_dir / f"{prefix}_heatmap.png"
                overlay_path = output_dir / f"{prefix}_overlay.png"
                model_image.save(input_path)
                heatmap_to_rgb(result.heatmap).save(heatmap_path)
                overlay_heatmap(model_image, result.heatmap).save(overlay_path)
                rows.append(
                    {
                        "fold": fold,
                        f"{args.grouping}_group": group,
                        "image_path": str(record.path),
                        "true_label": record.label,
                        "predicted_label": class_names[predicted_index],
                        "raw_predicted_label": class_names[raw_predicted_index],
                        "saved_prediction_used": saved_prediction is not None,
                        "is_correct": is_correct,
                        "target_class": class_names[target_index],
                        "probability_healthy": float(probabilities[class_names.index("healthy")]),
                        "probability_leukemia": float(probabilities[class_names.index("leukemia")]),
                        "model_input": str(input_path),
                        "heatmap": str(heatmap_path),
                        "overlay": str(overlay_path),
                    }
                )
                generated += 1
                if args.max_images is not None and generated >= args.max_images:
                    break
        print(
            f"Fold {fold:02d}: generated {generated} total Grad-CAM heatmaps",
            flush=True,
        )
        if args.max_images is not None and generated >= args.max_images:
            break

    pd.DataFrame(rows).to_csv(output_dir / "gradcam_manifest.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "checkpoint_dir": str(checkpoint_dir),
                "source_result_dir": str(source_result_dir) if source_result_dir else None,
                "data_dir": str(args.data_dir.expanduser().resolve()),
                "profile": profile,
                "image_size": image_size,
                "grouping": args.grouping,
                "target_class": args.target_class,
                "only_errors": args.only_errors,
                "max_images": args.max_images,
                "model_config": model_config if "model_config" in locals() else None,
                "source": "prepared real ALL-IDB1 files evaluated with saved fold checkpoints",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(rows)} Grad-CAM rows to {output_dir / 'gradcam_manifest.csv'}")


if __name__ == "__main__":
    main()
