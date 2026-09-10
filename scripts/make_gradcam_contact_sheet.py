from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a contact sheet from a Grad-CAM manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tile-width", type=int, default=280)
    parser.add_argument("--columns", type=int, default=3)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        font_path = Path(path)
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def make_label(row: pd.Series) -> str:
    image_name = Path(row["image_path"]).name
    return (
        f"{image_name}\n"
        f"true={row['true_label']} pred={row['predicted_label']} target={row['target_class']}"
    )


def main() -> None:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    output = args.output or manifest.with_name("gradcam_contact_sheet.png")
    frame = pd.read_csv(manifest)
    if frame.empty:
        raise ValueError(f"No rows found in {manifest}")

    font = load_font(14)
    padding = 12
    label_height = 54
    tile_width = args.tile_width
    image_height = tile_width
    tile_height = image_height + label_height + padding
    columns = max(1, args.columns)
    rows = (len(frame) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        color=(250, 250, 250),
    )
    draw = ImageDraw.Draw(sheet)

    for index, (_, row) in enumerate(frame.iterrows()):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        overlay = Image.open(row["overlay"]).convert("RGB")
        overlay.thumbnail(
            (tile_width - 2 * padding, image_height - 2 * padding),
            Image.Resampling.LANCZOS,
        )
        image_x = x + (tile_width - overlay.width) // 2
        image_y = y + padding
        sheet.paste(overlay, (image_x, image_y))
        label = "\n".join(textwrap.wrap(make_label(row), width=34))
        draw.text((x + padding, y + image_height), label, fill=(20, 20, 20), font=font)

    sheet.save(output)
    print(output)


if __name__ == "__main__":
    main()
