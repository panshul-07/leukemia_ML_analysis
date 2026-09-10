from __future__ import annotations

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from leukemia_osl.data import _base_preprocessing, build_transforms
from leukemia_osl.gradcam import GradCAM, overlay_heatmap
from leukemia_osl.model import create_model
from leukemia_osl.preprocess import DEFAULT_CLASS_NAMES
from leukemia_osl.stain_features import (
    extract_stain_morphology_features,
    feature_dicts_to_matrix,
    make_preprocessing_preview,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "results/stain_morphology_grouped/deployment_model.joblib"
DEFAULT_GRADCAM_CHECKPOINT_PATH = (
    PROJECT_ROOT / "results/checkpoints/hybrid_cnn_transformer_vgg_grouped/fold_01_best.pth"
)
DEFAULT_GRADCAM_SOURCE_DIR = PROJECT_ROOT / "results/hybrid_cnn_transformer_vgg_grouped"


def _load_model(model_path: str | Path) -> dict:
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. Run "
            "`python scripts/run_stain_morphology_experiment.py` first."
        )
    return joblib.load(model_path)


def predict_image(image: Image.Image, model_path: str = str(DEFAULT_MODEL_PATH)) -> tuple[str, Image.Image]:
    if image is None:
        raise ValueError("Upload a microscope image first.")

    payload = _load_model(model_path)
    max_side = int(payload.get("max_side", 768))
    with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
        image.convert("RGB").save(temp_file.name)
        features = extract_stain_morphology_features(Path(temp_file.name), max_side=max_side)
        matrix, _ = feature_dicts_to_matrix([features], payload["feature_names"])
        probabilities = payload["pipeline"].predict_proba(matrix)[0]
        classifier_classes = list(payload["pipeline"].named_steps["classifier"].classes_)
        probability_by_class = {
            payload["class_names"][int(class_id)]: float(probabilities[position])
            for position, class_id in enumerate(classifier_classes)
        }
        leukemia_probability = probability_by_class.get("leukemia", 0.0)
        threshold = float(payload.get("threshold", 0.5))
        predicted_label = "leukemia" if leukemia_probability >= threshold else "healthy"
        preview = make_preprocessing_preview(Path(temp_file.name), max_side=max_side)

    healthy_probability = probability_by_class.get("healthy", 1.0 - leukemia_probability)
    summary = (
        f"Prediction: {predicted_label}\n"
        f"Leukemia probability: {leukemia_probability:.4f}\n"
        f"Healthy probability: {healthy_probability:.4f}\n"
        f"Decision threshold: {threshold:.4f}\n"
        f"Artifact: {Path(model_path).name}"
    )
    return summary, preview


def _load_run_config(source_result_dir: str | Path) -> dict:
    config_path = Path(source_result_dir).expanduser().resolve() / "run_config.json"
    if not config_path.exists():
        return {}
    import json

    return json.loads(config_path.read_text(encoding="utf-8"))


def _infer_backbone(checkpoint: dict, model_config: dict) -> dict:
    if model_config.get("name") != "hybrid_cnn_transformer" or "cnn_backbone_name" in model_config:
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


def predict_gradcam(
    image: Image.Image,
    checkpoint_path: str = str(DEFAULT_GRADCAM_CHECKPOINT_PATH),
    source_result_dir: str = str(DEFAULT_GRADCAM_SOURCE_DIR),
    target_class: str = "predicted",
) -> tuple[str, Image.Image, Image.Image]:
    if image is None:
        raise ValueError("Upload a microscope image first.")

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    class_names = list(checkpoint["class_names"])
    model_config = _infer_backbone(checkpoint, dict(checkpoint["model_config"]))
    model_config["pretrained_backbone"] = False
    model = create_model(num_classes=len(class_names), **model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    run_config = _load_run_config(source_result_dir)
    profile = run_config.get("preprocessing_profile") or run_config.get("profile") or "hybrid"
    image_size = int(run_config.get("image_size", 224))
    preprocessing = transforms.Compose(_base_preprocessing(image_size, profile))
    tensor_transform = build_transforms(
        image_size=image_size,
        train=False,
        profile=profile,
        mean=tuple(checkpoint["normalization_mean"]),
        std=tuple(checkpoint["normalization_std"]),
        preprocessed=True,
    )
    model_image = preprocessing(image.convert("RGB"))
    input_tensor = tensor_transform(model_image).unsqueeze(0)

    with GradCAM(model, model.cnn_backbone) as gradcam:
        predicted_result = gradcam(input_tensor)
        predicted_index = int(np.argmax(predicted_result.probabilities))
        if target_class == "predicted":
            result = predicted_result
            target_index = predicted_index
        else:
            target_index = DEFAULT_CLASS_NAMES.index(target_class)
            result = gradcam(input_tensor, class_index=target_index)

    leukemia_probability = float(result.probabilities[class_names.index("leukemia")])
    healthy_probability = float(result.probabilities[class_names.index("healthy")])
    overlay = overlay_heatmap(model_image, result.heatmap)
    summary = (
        f"Prediction: {class_names[predicted_index]}\n"
        f"Grad-CAM target: {class_names[target_index]}\n"
        f"Leukemia probability: {leukemia_probability:.4f}\n"
        f"Healthy probability: {healthy_probability:.4f}\n"
        f"Checkpoint: {checkpoint_path.name}"
    )
    return summary, model_image, overlay


def build_demo():
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Gradio is not installed. Install project requirements with "
            "`.venv/bin/pip install -r requirements.txt`."
        ) from exc

    with gr.Blocks(title="ALL-IDB1 Leukemia Classifier") as demo:
        gr.Markdown("## ALL-IDB1 Leukemia Classifier")
        with gr.Tab("Prediction"):
            with gr.Row():
                image_input = gr.Image(type="pil", label="Microscope image")
                with gr.Column():
                    model_input = gr.Textbox(
                        value=str(DEFAULT_MODEL_PATH),
                        label="Model artifact",
                    )
                    run_button = gr.Button("Predict", variant="primary")
                    result_output = gr.Textbox(label="Result", lines=6)
            preview_output = gr.Image(type="pil", label="Preprocessing preview")
            run_button.click(
                predict_image,
                inputs=[image_input, model_input],
                outputs=[result_output, preview_output],
            )
        with gr.Tab("Grad-CAM"):
            with gr.Row():
                gradcam_image_input = gr.Image(type="pil", label="Microscope image")
                with gr.Column():
                    checkpoint_input = gr.Textbox(
                        value=str(DEFAULT_GRADCAM_CHECKPOINT_PATH),
                        label="Checkpoint",
                    )
                    source_input = gr.Textbox(
                        value=str(DEFAULT_GRADCAM_SOURCE_DIR),
                        label="Source result",
                    )
                    target_input = gr.Dropdown(
                        choices=["predicted", "leukemia", "healthy"],
                        value="predicted",
                        label="Target",
                    )
                    gradcam_button = gr.Button("Generate", variant="primary")
                    gradcam_result_output = gr.Textbox(label="Result", lines=6)
            with gr.Row():
                gradcam_input_output = gr.Image(type="pil", label="Model input")
                gradcam_overlay_output = gr.Image(type="pil", label="Grad-CAM overlay")
            gradcam_button.click(
                predict_gradcam,
                inputs=[gradcam_image_input, checkpoint_input, source_input, target_input],
                outputs=[
                    gradcam_result_output,
                    gradcam_input_output,
                    gradcam_overlay_output,
                ],
            )
    return demo


if __name__ == "__main__":
    build_demo().launch()
