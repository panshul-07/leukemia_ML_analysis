"""Data loading and preprocessing transforms for ALL-IDB1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from .preprocess import DEFAULT_CLASS_NAMES, ImageRecord

DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)
VALID_PROFILES = {
    "gnn_microscopy_enhanced",
    "paper",
    "hybrid",
    "hybrid_microscopy",
    "hybrid_segmented_pretrained",
}


class AnnotationColorSuppressor:
    """Replace orange/yellow annotation-like pixels with background smear color."""

    def __call__(self, image: Image.Image) -> Image.Image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
        artifact_mask = (
            (red > 0.55)
            & (green > 0.30)
            & (blue < 0.38)
            & ((red - blue) > 0.20)
            & ((green - blue) > 0.10)
        )
        if not artifact_mask.any():
            return image.convert("RGB")

        artifact_mask = ndimage.binary_dilation(
            artifact_mask,
            structure=np.ones((3, 3), dtype=bool),
        )
        valid_pixels = pixels[~artifact_mask]
        if valid_pixels.size == 0:
            fill_color = np.percentile(pixels.reshape(-1, 3), 85.0, axis=0)
        else:
            fill_color = np.percentile(valid_pixels, 85.0, axis=0)
        pixels = pixels.copy()
        pixels[artifact_mask] = fill_color
        return Image.fromarray(
            np.round(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8),
            mode="RGB",
        )


class MicroscopyWhiteBalance:
    """Normalize slide background color without using class or fold-test statistics."""

    def __init__(self, white_percentile: float = 95.0, cutoff: float = 0.5) -> None:
        self.white_percentile = white_percentile
        self.cutoff = cutoff

    def __call__(self, image: Image.Image) -> Image.Image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        flat_pixels = pixels.reshape(-1, 3)
        white = np.percentile(flat_pixels, self.white_percentile, axis=0)
        pixels = pixels / np.maximum(white, 1e-3)[None, None, :]

        low = float(np.percentile(pixels, self.cutoff))
        high = float(np.percentile(pixels, 100.0 - self.cutoff))
        pixels = (pixels - low) / max(high - low, 1e-3)
        pixels = np.clip(pixels, 0.0, 1.0)
        return Image.fromarray(np.round(pixels * 255.0).astype(np.uint8), mode="RGB")


class LeukocyteMosaic:
    """Combine the full smear with three label-blind nucleus-rich crops."""

    def __init__(
        self,
        output_size: int,
        crop_fraction: float = 0.34,
        detector_size: int = 320,
    ) -> None:
        self.output_size = output_size
        self.crop_fraction = crop_fraction
        self.detector_size = detector_size

    @staticmethod
    def _fit_square(image: Image.Image, size: int, background: tuple[int, int, int]) -> Image.Image:
        contained = image.copy()
        contained.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), background)
        left = (size - contained.width) // 2
        top = (size - contained.height) // 2
        canvas.paste(contained, (left, top))
        return canvas

    @staticmethod
    def _background_color(image: Image.Image) -> tuple[int, int, int]:
        pixels = np.asarray(image, dtype=np.uint8).reshape(-1, 3)
        return tuple(np.percentile(pixels, 90.0, axis=0).round().astype(np.uint8).tolist())

    def _candidate_centers(self, image: Image.Image, count: int) -> list[tuple[float, float]]:
        detector = image.copy()
        detector.thumbnail((self.detector_size, self.detector_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(detector, dtype=np.float32) / 255.0
        red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        nuclear_score = (
            np.maximum(red - green, 0.0)
            + np.maximum(blue - green, 0.0)
            + 0.30 * (1.0 - luminance)
        )
        scale = max(float(np.percentile(nuclear_score, 99.5)), 1e-6)
        score_image = Image.fromarray(
            np.clip(nuclear_score / scale * 255.0, 0.0, 255.0).astype(np.uint8),
            mode="L",
        ).filter(ImageFilter.GaussianBlur(radius=max(1.0, min(detector.size) * 0.012)))
        score = np.asarray(score_image, dtype=np.float32).copy()

        height, width = score.shape
        border_y = max(1, int(round(height * 0.04)))
        border_x = max(1, int(round(width * 0.04)))
        score[:border_y] = 0
        score[-border_y:] = 0
        score[:, :border_x] = 0
        score[:, -border_x:] = 0
        suppression_radius = max(4, int(round(min(height, width) * self.crop_fraction * 0.55)))

        centers: list[tuple[float, float]] = []
        for _ in range(count):
            y, x = np.unravel_index(int(np.argmax(score)), score.shape)
            centers.append((x / width, y / height))
            x0, x1 = max(0, x - suppression_radius), min(width, x + suppression_radius + 1)
            y0, y1 = max(0, y - suppression_radius), min(height, y + suppression_radius + 1)
            score[y0:y1, x0:x1] = 0
        return centers

    def _crop_at(
        self,
        image: Image.Image,
        center: tuple[float, float],
        background: tuple[int, int, int],
    ) -> Image.Image:
        side = max(1, int(round(min(image.size) * self.crop_fraction)))
        center_x = int(round(center[0] * image.width))
        center_y = int(round(center[1] * image.height))
        left = center_x - side // 2
        top = center_y - side // 2
        source_left, source_top = max(0, left), max(0, top)
        source_right = min(image.width, left + side)
        source_bottom = min(image.height, top + side)
        crop = Image.new("RGB", (side, side), background)
        crop.paste(
            image.crop((source_left, source_top, source_right, source_bottom)),
            (source_left - left, source_top - top),
        )
        return crop

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        background = self._background_color(image)
        tile_size = self.output_size // 2
        centers = self._candidate_centers(image, count=3)
        tiles = [self._fit_square(image, tile_size, background)]
        tiles.extend(
            self._fit_square(self._crop_at(image, center, background), tile_size, background)
            for center in centers
        )
        mosaic = Image.new("RGB", (tile_size * 2, tile_size * 2), background)
        for index, tile in enumerate(tiles):
            mosaic.paste(tile, ((index % 2) * tile_size, (index // 2) * tile_size))
        if mosaic.size != (self.output_size, self.output_size):
            mosaic = mosaic.resize(
                (self.output_size, self.output_size), Image.Resampling.LANCZOS
            )
        return mosaic


class LeukocyteSegmentationMosaic(LeukocyteMosaic):
    """Full field, two segmented leukocyte crops, and a nuclear-mask overview."""

    def __init__(
        self,
        output_size: int,
        crop_fraction: float = 0.26,
        detector_size: int = 256,
    ) -> None:
        super().__init__(output_size, crop_fraction, detector_size)

    def _segment_nuclei(
        self, image: Image.Image
    ) -> tuple[list[tuple[float, float]], Image.Image]:
        detector = image.copy()
        detector.thumbnail((self.detector_size, self.detector_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(detector, dtype=np.float32) / 255.0
        red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        saturation = pixels.max(axis=2) - pixels.min(axis=2)

        # True purple nuclear stain requires both red and blue to exceed green.
        # This rejects yellow/orange arrows that can appear in ALL-IDB1 annotations.
        purple = 2.0 * np.minimum(
            np.maximum(red - green, 0.0),
            np.maximum(blue - green, 0.0),
        )
        nuclear_score = purple + 0.18 * (1.0 - luminance) * saturation
        sigma = max(0.8, min(detector.size) * 0.006)
        smoothed = ndimage.gaussian_filter(nuclear_score, sigma=sigma)
        threshold = max(float(np.percentile(smoothed, 96.0)), 0.035)
        mask = smoothed >= threshold
        mask = ndimage.binary_opening(mask, structure=np.ones((2, 2), dtype=bool))
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))

        labels, component_count = ndimage.label(mask)
        image_area = mask.size
        min_area = max(8, int(round(image_area * 0.00012)))
        max_area = int(round(image_area * 0.05))
        components: list[tuple[float, float, float]] = []
        clean_mask = np.zeros_like(mask)
        for component_id in range(1, component_count + 1):
            component = labels == component_id
            area = int(component.sum())
            if not min_area <= area <= max_area:
                continue
            component_score = float(smoothed[component].mean())
            if component_score < threshold:
                continue
            center_y, center_x = ndimage.center_of_mass(smoothed, labels, component_id)
            rank_score = component_score * np.sqrt(area)
            components.append((rank_score, float(center_x), float(center_y)))
            clean_mask |= component

        components.sort(reverse=True)
        centers = [
            (center_x / detector.width, center_y / detector.height)
            for _, center_x, center_y in components[:2]
        ]
        if len(centers) < 2:
            fallback = super()._candidate_centers(image, count=2)
            for center in fallback:
                if len(centers) >= 2:
                    break
                if all(
                    (center[0] - existing[0]) ** 2 + (center[1] - existing[1]) ** 2
                    > 0.01
                    for existing in centers
                ):
                    centers.append(center)

        mask_image = Image.fromarray((clean_mask * 255).astype(np.uint8), mode="L")
        return centers[:2], mask_image

    @staticmethod
    def _segmentation_overview(
        image: Image.Image,
        mask: Image.Image,
        size: int,
        background: tuple[int, int, int],
    ) -> Image.Image:
        base = LeukocyteMosaic._fit_square(image, size, background)
        mask_square = LeukocyteMosaic._fit_square(
            mask.convert("RGB"), size, (0, 0, 0)
        ).convert("L")
        pixels = np.asarray(base, dtype=np.float32)
        mask_pixels = np.asarray(mask_square, dtype=np.float32) / 255.0
        gray = (
            0.299 * pixels[..., 0]
            + 0.587 * pixels[..., 1]
            + 0.114 * pixels[..., 2]
        )
        overview = np.stack((gray, gray, gray), axis=2) * 0.72
        nuclear_color = np.array([105.0, 25.0, 155.0], dtype=np.float32)
        overview = overview * (1.0 - mask_pixels[..., None]) + nuclear_color * mask_pixels[..., None]
        return Image.fromarray(np.clip(overview, 0, 255).astype(np.uint8), mode="RGB")

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        background = self._background_color(image)
        tile_size = self.output_size // 2
        centers, mask = self._segment_nuclei(image)
        tiles = [self._fit_square(image, tile_size, background)]
        tiles.extend(
            self._fit_square(self._crop_at(image, center, background), tile_size, background)
            for center in centers
        )
        while len(tiles) < 3:
            tiles.append(tiles[0].copy())
        tiles.append(self._segmentation_overview(image, mask, tile_size, background))

        mosaic = Image.new("RGB", (tile_size * 2, tile_size * 2), background)
        for index, tile in enumerate(tiles):
            mosaic.paste(tile, ((index % 2) * tile_size, (index // 2) * tile_size))
        if mosaic.size != (self.output_size, self.output_size):
            mosaic = mosaic.resize(
                (self.output_size, self.output_size), Image.Resampling.LANCZOS
            )
        return mosaic


def _base_preprocessing(image_size: int, profile: str) -> list:
    if profile not in VALID_PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(sorted(VALID_PROFILES))}")
    if profile == "gnn_microscopy_enhanced":
        return [
            AnnotationColorSuppressor(),
            MicroscopyWhiteBalance(white_percentile=97.0, cutoff=0.25),
            LeukocyteSegmentationMosaic(
                output_size=image_size,
                crop_fraction=0.30,
                detector_size=384,
            ),
        ]
    if profile == "hybrid_microscopy":
        return [MicroscopyWhiteBalance(), LeukocyteMosaic(output_size=image_size)]
    if profile == "hybrid_segmented_pretrained":
        return [
            MicroscopyWhiteBalance(),
            LeukocyteSegmentationMosaic(output_size=image_size),
        ]
    return [
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    ]


def estimate_channel_stats(
    train_dir: Path,
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 0,
    profile: str = "paper",
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Estimate RGB normalization from the training split only."""

    transform = transforms.Compose(
        [
            *_base_preprocessing(image_size, profile),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    if not dataset:
        raise ValueError(f"Training split contains no images: {train_dir}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for images, _ in loader:
        images = images.to(torch.float64)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (images**2).sum(dim=(0, 2, 3))
        pixel_count += images.shape[0] * images.shape[2] * images.shape[3]

    mean = channel_sum / pixel_count
    variance = (channel_squared_sum / pixel_count) - mean.square()
    std = variance.clamp_min(1e-12).sqrt()
    return tuple(mean.tolist()), tuple(std.tolist())


def estimate_record_channel_stats(
    records: list[ImageRecord],
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 0,
    profile: str = "paper",
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Estimate RGB statistics from an explicit fold-training record list."""

    if not records:
        raise ValueError("Cannot estimate normalization from an empty record list.")
    transform = transforms.Compose(
        [
            *_base_preprocessing(image_size, profile),
            transforms.ToTensor(),
        ]
    )
    class_to_idx = {name: index for index, name in enumerate(DEFAULT_CLASS_NAMES)}
    dataset = ImageRecordDataset(records, class_to_idx, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for images, _ in loader:
        images = images.to(torch.float64)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (images**2).sum(dim=(0, 2, 3))
        pixel_count += images.shape[0] * images.shape[2] * images.shape[3]

    mean = channel_sum / pixel_count
    variance = (channel_squared_sum / pixel_count) - mean.square()
    std = variance.clamp_min(1e-12).sqrt()
    return tuple(mean.tolist()), tuple(std.tolist())


def build_transforms(
    image_size: int = 224,
    train: bool = False,
    profile: str = "paper",
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
    preprocessed: bool = False,
) -> transforms.Compose:
    """Build fold-safe microscopy transforms."""

    base_preprocessing = [] if preprocessed else _base_preprocessing(image_size, profile)

    if train:
        if profile == "paper":
            augmentation = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
            ]
        elif profile in {
            "gnn_microscopy_enhanced",
            "hybrid",
            "hybrid_microscopy",
            "hybrid_segmented_pretrained",
        }:
            augmentation = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(
                    degrees=14 if profile == "gnn_microscopy_enhanced" else 10,
                    translate=(0.03, 0.03) if profile == "gnn_microscopy_enhanced" else (0.02, 0.02),
                    scale=(0.94, 1.06) if profile == "gnn_microscopy_enhanced" else (0.97, 1.03),
                    shear=(-3, 3, -3, 3) if profile == "gnn_microscopy_enhanced" else (-2, 2, -2, 2),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.12 if profile == "gnn_microscopy_enhanced" else 0.08,
                            contrast=0.16 if profile == "gnn_microscopy_enhanced" else 0.10,
                            saturation=0.10 if profile == "gnn_microscopy_enhanced" else 0.06,
                            hue=0.015 if profile == "gnn_microscopy_enhanced" else 0.01,
                        )
                    ],
                    p=0.65 if profile == "gnn_microscopy_enhanced" else 0.5,
                ),
                transforms.RandomAutocontrast(p=0.20 if profile == "gnn_microscopy_enhanced" else 0.10),
                transforms.RandomAdjustSharpness(
                    sharpness_factor=1.35 if profile == "gnn_microscopy_enhanced" else 1.2,
                    p=0.20 if profile == "gnn_microscopy_enhanced" else 0.10,
                ),
            ]
        else:
            raise ValueError(f"profile must be one of: {', '.join(sorted(VALID_PROFILES))}")

        return transforms.Compose(
            [
                *base_preprocessing,
                *augmentation,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    if profile not in VALID_PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(sorted(VALID_PROFILES))}")

    return transforms.Compose(
        [
            *base_preprocessing,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def make_imagefolder_loaders(
    processed_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    profile: str = "paper",
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
) -> tuple[dict[str, DataLoader], list[str]]:
    processed_dir = processed_dir.expanduser().resolve()
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=build_transforms(
            image_size,
            train=True,
            profile=profile,
            mean=mean,
            std=std,
        ),
    )
    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=build_transforms(
            image_size,
            train=False,
            profile=profile,
            mean=mean,
            std=std,
        ),
    )
    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=build_transforms(
            image_size,
            train=False,
            profile=profile,
            mean=mean,
            std=std,
        ),
    )

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    return loaders, train_dataset.classes


def make_record_loaders(
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    test_records: list[ImageRecord],
    image_size: int,
    batch_size: int,
    num_workers: int,
    profile: str,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> tuple[dict[str, DataLoader], list[str]]:
    """Build fold-isolated loaders directly from ALL-IDB1 manifest records."""

    class_names = list(DEFAULT_CLASS_NAMES)
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    deterministic_transform = None
    if profile in {
        "gnn_microscopy_enhanced",
        "hybrid_microscopy",
        "hybrid_segmented_pretrained",
    }:
        deterministic_transform = transforms.Compose(_base_preprocessing(image_size, profile))
    datasets_by_split = {
        "train": ImageRecordDataset(
            train_records,
            class_to_idx,
            build_transforms(
                image_size,
                True,
                profile,
                mean,
                std,
                preprocessed=deterministic_transform is not None,
            ),
            deterministic_transform=deterministic_transform,
        ),
        "val": ImageRecordDataset(
            val_records,
            class_to_idx,
            build_transforms(
                image_size,
                False,
                profile,
                mean,
                std,
                preprocessed=deterministic_transform is not None,
            ),
            deterministic_transform=deterministic_transform,
        ),
        "test": ImageRecordDataset(
            test_records,
            class_to_idx,
            build_transforms(
                image_size,
                False,
                profile,
                mean,
                std,
                preprocessed=deterministic_transform is not None,
            ),
            deterministic_transform=deterministic_transform,
        ),
    }
    loaders = {
        split_name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split_name, dataset in datasets_by_split.items()
    }
    return loaders, class_names


class ImageRecordDataset(Dataset):
    """Dataset wrapper used for cross-validation directly from a manifest."""

    def __init__(
        self,
        records: list[ImageRecord],
        class_to_idx: dict[str, int],
        transform,
        deterministic_transform=None,
    ):
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.deterministic_transform = deterministic_transform
        self._image_cache: dict[int, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        if self.deterministic_transform is not None:
            if index not in self._image_cache:
                with Image.open(record.path) as source:
                    self._image_cache[index] = self.deterministic_transform(
                        source.convert("RGB")
                    )
            image = self._image_cache[index].copy()
        else:
            with Image.open(record.path) as source:
                image = source.convert("RGB")
        return self.transform(image), self.class_to_idx[record.label]
