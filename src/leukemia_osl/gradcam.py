"""Grad-CAM utilities for the hybrid CNN/Transformer model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class GradCAMResult:
    class_index: int
    logits: np.ndarray
    probabilities: np.ndarray
    heatmap: np.ndarray


class GradCAM:
    """Compute Grad-CAM over the final CNN feature map feeding the Transformer."""

    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module) -> None:
        self.model = model
        self.target_module = target_module
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._forward_handle = target_module.register_forward_hook(self._save_activations)
        self._backward_handle = target_module.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output) -> None:
        self._activations = output

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0]

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def __call__(
        self,
        inputs: torch.Tensor,
        class_index: int | None = None,
    ) -> GradCAMResult:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self._activations = None
        self._gradients = None
        inputs = inputs.detach().requires_grad_(True)

        logits = self.model(inputs)
        probabilities = torch.softmax(logits, dim=1)
        if class_index is None:
            class_index = int(probabilities.argmax(dim=1).item())
        score = logits[:, class_index].sum()
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activations = self._activations.detach()
        gradients = self._gradients.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=inputs.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        cam = cam[0, 0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)

        return GradCAMResult(
            class_index=class_index,
            logits=logits.detach().cpu().numpy()[0],
            probabilities=probabilities.detach().cpu().numpy()[0],
            heatmap=cam.detach().cpu().numpy(),
        )


def heatmap_to_rgb(heatmap: np.ndarray) -> Image.Image:
    heatmap = np.clip(heatmap, 0.0, 1.0)
    cold = np.array([28.0, 48.0, 86.0], dtype=np.float32)
    mid = np.array([245.0, 183.0, 66.0], dtype=np.float32)
    hot = np.array([214.0, 45.0, 76.0], dtype=np.float32)
    first = heatmap[..., None] <= 0.58
    t1 = np.clip(heatmap[..., None] / 0.58, 0.0, 1.0)
    t2 = np.clip((heatmap[..., None] - 0.58) / 0.42, 0.0, 1.0)
    rgb = np.where(first, cold * (1.0 - t1) + mid * t1, mid * (1.0 - t2) + hot * t2)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def overlay_heatmap(
    base_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.42,
) -> Image.Image:
    base = base_image.convert("RGB")
    heatmap_image = heatmap_to_rgb(heatmap).resize(base.size, Image.Resampling.BICUBIC)
    base_pixels = np.asarray(base, dtype=np.float32)
    heat_pixels = np.asarray(heatmap_image, dtype=np.float32)
    blended = base_pixels * (1.0 - alpha) + heat_pixels * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
