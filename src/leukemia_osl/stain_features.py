"""Label-blind stain and morphology features for ALL-IDB1 microscope fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


FEATURE_VERSION = "stain_morphology_v1"


@dataclass(frozen=True)
class StainPreprocessingResult:
    rgb: np.ndarray
    white_balanced: np.ndarray
    nuclear_score: np.ndarray
    nuclear_mask: np.ndarray
    artifact_mask: np.ndarray


def _as_float_rgb(image_path: Path, max_side: int = 768) -> np.ndarray:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.float32) / 255.0


def _white_balance(rgb: np.ndarray) -> np.ndarray:
    flat = rgb.reshape(-1, 3)
    white = np.percentile(flat, 95.0, axis=0)
    balanced = rgb / np.maximum(white, 1e-3)[None, None, :]
    low = np.percentile(balanced, 0.5, axis=(0, 1))
    high = np.percentile(balanced, 99.5, axis=(0, 1))
    balanced = (balanced - low[None, None, :]) / np.maximum(
        high - low,
        1e-3,
    )[None, None, :]
    return np.clip(balanced, 0.0, 1.0)


def _rgb_channels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return rgb[..., 0], rgb[..., 1], rgb[..., 2]


def preprocess_stain_image(
    image_path: Path,
    max_side: int = 768,
) -> StainPreprocessingResult:
    """Create fold-safe stain-normalized arrays and a nucleus-rich mask.

    The detector only uses image colors and geometry. It intentionally suppresses
    orange/yellow annotation-like pixels so the classifier cannot lean on ALL-IDB1
    arrows instead of cell morphology.
    """

    rgb = _as_float_rgb(image_path, max_side=max_side)
    balanced = _white_balance(rgb)
    red, green, blue = _rgb_channels(balanced)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    saturation = balanced.max(axis=2) - balanced.min(axis=2)

    artifact_mask = (
        (red > 0.55)
        & (green > 0.34)
        & (blue < 0.30)
        & ((red - blue) > 0.22)
        & ((green - blue) > 0.12)
    )

    purple_core = 2.0 * np.minimum(
        np.maximum(red - green, 0.0),
        np.maximum(blue - green, 0.0),
    )
    red_blue_excess = np.maximum((red + blue) * 0.5 - green, 0.0)
    dark_saturated = (1.0 - luminance) * saturation
    nuclear_score = 1.65 * purple_core + 0.42 * red_blue_excess * saturation
    nuclear_score += 0.25 * dark_saturated
    nuclear_score = np.where(artifact_mask, 0.0, nuclear_score)
    sigma = max(0.8, min(balanced.shape[:2]) * 0.0045)
    smoothed = ndimage.gaussian_filter(nuclear_score, sigma=sigma)

    valid_scores = smoothed[~artifact_mask]
    if valid_scores.size:
        percentile_threshold = float(np.percentile(valid_scores, 95.5))
        spread_threshold = float(valid_scores.mean() + 1.15 * valid_scores.std())
        threshold = max(min(percentile_threshold, spread_threshold), 0.028)
    else:
        threshold = 0.028
    nuclear_mask = (smoothed >= threshold) & ~artifact_mask
    nuclear_mask = ndimage.binary_opening(
        nuclear_mask,
        structure=np.ones((2, 2), dtype=bool),
    )
    nuclear_mask = ndimage.binary_closing(
        nuclear_mask,
        structure=np.ones((3, 3), dtype=bool),
    )

    labels, component_count = ndimage.label(nuclear_mask)
    image_area = nuclear_mask.size
    min_area = max(5, int(round(image_area * 0.000045)))
    max_area = int(round(image_area * 0.065))
    clean_mask = np.zeros_like(nuclear_mask)
    for component_id in range(1, component_count + 1):
        component = labels == component_id
        area = int(component.sum())
        if min_area <= area <= max_area:
            clean_mask |= component

    return StainPreprocessingResult(
        rgb=rgb,
        white_balanced=balanced,
        nuclear_score=smoothed,
        nuclear_mask=clean_mask,
        artifact_mask=artifact_mask,
    )


def _finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    return values[np.isfinite(values)]


def _add_stats(features: dict[str, float], prefix: str, values: np.ndarray) -> None:
    finite = _finite_values(values)
    if finite.size == 0:
        for suffix in (
            "mean",
            "std",
            "min",
            "p05",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
            "max",
            "iqr",
        ):
            features[f"{prefix}_{suffix}"] = 0.0
        return

    p05, p10, p25, p50, p75, p90, p95 = np.percentile(
        finite,
        [5, 10, 25, 50, 75, 90, 95],
    )
    features[f"{prefix}_mean"] = float(finite.mean())
    features[f"{prefix}_std"] = float(finite.std())
    features[f"{prefix}_min"] = float(finite.min())
    features[f"{prefix}_p05"] = float(p05)
    features[f"{prefix}_p10"] = float(p10)
    features[f"{prefix}_p25"] = float(p25)
    features[f"{prefix}_p50"] = float(p50)
    features[f"{prefix}_p75"] = float(p75)
    features[f"{prefix}_p90"] = float(p90)
    features[f"{prefix}_p95"] = float(p95)
    features[f"{prefix}_max"] = float(finite.max())
    features[f"{prefix}_iqr"] = float(p75 - p25)


def _add_histogram(
    features: dict[str, float],
    prefix: str,
    values: np.ndarray,
    bins: int,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> None:
    finite = _finite_values(values)
    if finite.size == 0:
        histogram = np.zeros(bins, dtype=np.float64)
    else:
        histogram, _ = np.histogram(finite, bins=bins, range=value_range)
        histogram = histogram.astype(np.float64)
        histogram /= max(float(histogram.sum()), 1.0)
    for index, value in enumerate(histogram):
        features[f"{prefix}_hist_{index:02d}"] = float(value)


def _component_features(
    features: dict[str, float],
    mask: np.ndarray,
    score: np.ndarray,
) -> None:
    labels, component_count = ndimage.label(mask)
    image_area = float(mask.size)
    areas: list[float] = []
    scores: list[float] = []
    fill_ratios: list[float] = []
    aspect_ratios: list[float] = []
    centers: list[tuple[float, float]] = []

    for component_id in range(1, component_count + 1):
        component = labels == component_id
        area = float(component.sum())
        if area <= 0:
            continue
        rows, cols = np.nonzero(component)
        height = float(rows.max() - rows.min() + 1)
        width = float(cols.max() - cols.min() + 1)
        box_area = max(height * width, 1.0)
        areas.append(area / image_area)
        scores.append(float(score[component].mean()))
        fill_ratios.append(float(area / box_area))
        aspect_ratios.append(float(min(height, width) / max(height, width, 1.0)))
        centers.append((float(cols.mean() / mask.shape[1]), float(rows.mean() / mask.shape[0])))

    features["nucleus_component_count"] = float(len(areas))
    features["nucleus_component_density"] = float(len(areas) / max(image_area / 1_000_000.0, 1e-6))
    features["nucleus_mask_fraction"] = float(mask.mean())
    _add_stats(features, "nucleus_area_fraction", np.asarray(areas))
    _add_stats(features, "nucleus_component_score", np.asarray(scores))
    _add_stats(features, "nucleus_fill_ratio", np.asarray(fill_ratios))
    _add_stats(features, "nucleus_aspect_ratio", np.asarray(aspect_ratios))

    if centers:
        center_array = np.asarray(centers, dtype=np.float64)
        features["nucleus_center_x_std"] = float(center_array[:, 0].std())
        features["nucleus_center_y_std"] = float(center_array[:, 1].std())
        center_distances = np.sqrt(
            (center_array[:, 0] - 0.5) ** 2 + (center_array[:, 1] - 0.5) ** 2
        )
        _add_stats(features, "nucleus_center_distance", center_distances)
    else:
        features["nucleus_center_x_std"] = 0.0
        features["nucleus_center_y_std"] = 0.0
        _add_stats(features, "nucleus_center_distance", np.asarray([]))


def extract_stain_morphology_features(
    image_path: Path,
    max_side: int = 768,
) -> dict[str, float]:
    """Extract deterministic preprocessing features from one microscope field."""

    processed = preprocess_stain_image(image_path, max_side=max_side)
    rgb = processed.white_balanced
    red, green, blue = _rgb_channels(rgb)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    nuclear_score = processed.nuclear_score
    nuclear_mask = processed.nuclear_mask
    background_mask = (~nuclear_mask) & (~processed.artifact_mask)
    valid_mask = ~processed.artifact_mask

    gradient_y, gradient_x = np.gradient(luminance)
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    laplacian = np.abs(ndimage.laplace(luminance))

    features: dict[str, float] = {
        "image_height": float(rgb.shape[0]),
        "image_width": float(rgb.shape[1]),
        "image_aspect_ratio": float(rgb.shape[1] / max(rgb.shape[0], 1)),
        "valid_pixel_fraction": float(valid_mask.mean()),
    }

    channel_maps = {
        "red": red,
        "green": green,
        "blue": blue,
        "luminance": luminance,
        "saturation": saturation,
        "red_minus_green": red - green,
        "blue_minus_green": blue - green,
        "red_blue_minus_green": (red + blue) * 0.5 - green,
        "nuclear_score": nuclear_score,
        "gradient": gradient,
        "laplacian": laplacian,
    }
    for name, values in channel_maps.items():
        _add_stats(features, f"global_{name}", values[valid_mask])
        _add_stats(features, f"nucleus_{name}", values[nuclear_mask])
        _add_stats(features, f"background_{name}", values[background_mask])
        features[f"contrast_{name}_nucleus_minus_background"] = (
            features[f"nucleus_{name}_mean"] - features[f"background_{name}_mean"]
        )

    _add_histogram(features, "global_nuclear_score", nuclear_score[valid_mask], bins=12)
    _add_histogram(features, "nucleus_luminance", luminance[nuclear_mask], bins=10)
    _add_histogram(features, "nucleus_saturation", saturation[nuclear_mask], bins=10)
    _component_features(features, nuclear_mask, nuclear_score)
    return features


def feature_dicts_to_matrix(
    feature_dicts: list[dict[str, float]],
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if feature_names is None:
        feature_names = sorted({name for row in feature_dicts for name in row})
    matrix = np.asarray(
        [
            [float(row.get(name, 0.0)) for name in feature_names]
            for row in feature_dicts
        ],
        dtype=np.float32,
    )
    return matrix, feature_names


def _float_to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def _score_to_heatmap(score: np.ndarray) -> Image.Image:
    finite = score[np.isfinite(score)]
    if finite.size:
        low = float(np.percentile(finite, 2.0))
        high = float(np.percentile(finite, 99.0))
    else:
        low, high = 0.0, 1.0
    normalized = np.clip((score - low) / max(high - low, 1e-6), 0.0, 1.0)
    dark = np.array([25.0, 33.0, 54.0], dtype=np.float32)
    mid = np.array([75.0, 76.0, 172.0], dtype=np.float32)
    bright = np.array([228.0, 80.0, 145.0], dtype=np.float32)
    first = normalized[..., None] <= 0.55
    t1 = np.clip(normalized[..., None] / 0.55, 0.0, 1.0)
    t2 = np.clip((normalized[..., None] - 0.55) / 0.45, 0.0, 1.0)
    colors = np.where(first, dark * (1.0 - t1) + mid * t1, mid * (1.0 - t2) + bright * t2)
    return Image.fromarray(np.clip(colors, 0, 255).astype(np.uint8), mode="RGB")


def make_preprocessing_preview(
    image_path: Path,
    max_side: int = 768,
    tile_size: int = 256,
) -> Image.Image:
    """Return a compact original / normalized / mask / score preview for Gradio."""

    processed = preprocess_stain_image(image_path, max_side=max_side)
    original = _float_to_image(processed.rgb).resize((tile_size, tile_size), Image.Resampling.BICUBIC)
    balanced = _float_to_image(processed.white_balanced).resize(
        (tile_size, tile_size),
        Image.Resampling.BICUBIC,
    )
    mask_rgb = np.repeat(processed.nuclear_mask[..., None], 3, axis=2).astype(np.float32)
    overlay = processed.white_balanced * 0.62
    overlay += mask_rgb * np.array([0.38, 0.02, 0.55], dtype=np.float32)
    overlay_image = _float_to_image(overlay).resize(
        (tile_size, tile_size),
        Image.Resampling.BICUBIC,
    )
    score_image = _score_to_heatmap(processed.nuclear_score).resize(
        (tile_size, tile_size),
        Image.Resampling.BICUBIC,
    )

    label_height = 24
    canvas = Image.new("RGB", (tile_size * 2, tile_size * 2 + label_height * 2), (245, 247, 250))
    tiles = [
        ("Original", original),
        ("White balanced", balanced),
        ("Nucleus mask", overlay_image),
        ("Stain score", score_image),
    ]
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for index, (label, image) in enumerate(tiles):
        col = index % 2
        row = index // 2
        x = col * tile_size
        y = row * (tile_size + label_height)
        draw.text((x + 8, y + 5), label, fill=(28, 31, 36), font=font)
        canvas.paste(image, (x, y + label_height))
    return canvas
