"""Create labelled screenshot-like training data from a folder of image assets."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CANVAS_SIZE = 768


def load_assets(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def random_ui(draw: ImageDraw.ImageDraw, rng: random.Random, width: int, height: int) -> None:
    dark = rng.choice((True, False))
    panel = (24, 27, 34) if dark else (245, 246, 248)
    muted = (75, 80, 90) if dark else (205, 208, 214)
    foreground = (225, 228, 234) if dark else (65, 70, 78)
    if rng.random() < 0.85:
        draw.rectangle((0, 0, width, rng.randint(42, 86)), fill=panel)
        for x in range(18, min(width - 12, 180), 28):
            draw.rounded_rectangle((x, 18, x + rng.randint(10, 19), 26), radius=4, fill=muted)
    if rng.random() < 0.65:
        y = rng.randint(height - 82, height - 42)
        draw.rounded_rectangle((12, y, width - 12, height - 12), radius=12, fill=panel)
        for x in range(28, width - 40, 56):
            draw.rounded_rectangle((x, y + 16, x + rng.randint(20, 42), y + 22), radius=3, fill=foreground)
    if rng.random() < 0.5:
        x = rng.choice((12, width - 112))
        y = rng.randint(100, max(101, height - 160))
        draw.rounded_rectangle((x, y, x + 96, y + 118), radius=10, fill=panel)
        for row in range(3):
            draw.rounded_rectangle((x + 10, y + 78 + row * 10, x + 70, y + 82 + row * 10), radius=2, fill=muted)


def make_sample(asset_paths: list[Path], rng: random.Random) -> tuple[Image.Image, Image.Image]:
    width = height = CANVAS_SIZE
    with Image.open(rng.choice(asset_paths)) as source:
        background = cover(source, (width, height)).filter(ImageFilter.GaussianBlur(rng.uniform(12, 34)))
    tint = Image.new("RGB", (width, height), rng.choice(((20, 25, 34), (238, 239, 242), (45, 38, 50))))
    screenshot = Image.blend(background, tint, rng.uniform(0.12, 0.48)).convert("RGBA")
    mask = Image.new("L", (width, height), 0)

    if rng.random() < 0.14:
        draw = ImageDraw.Draw(screenshot)
        random_ui(draw, rng, width, height)
        for _ in range(rng.randint(2, 5)):
            thumb_w, thumb_h = rng.randint(70, 180), rng.randint(65, 170)
            x, y = rng.randint(0, width - thumb_w), rng.randint(70, height - thumb_h)
            with Image.open(rng.choice(asset_paths)) as source:
                thumbnail = cover(source, (thumb_w, thumb_h)).convert("RGBA")
            screenshot.alpha_composite(thumbnail, (x, y))
        return screenshot.convert("RGB"), mask

    target_width, target_height = rng.randint(330, 650), rng.randint(300, 660)
    target_x = rng.randint(24, width - target_width - 24)
    target_y = rng.randint(65, height - target_height - 24)
    with Image.open(rng.choice(asset_paths)) as source:
        target = cover(source, (target_width, target_height)).convert("RGBA")

    radius = rng.choice((0, 0, 0, 8, 16, 28, 42))
    alpha = rounded_mask((target_width, target_height), radius)
    if rng.random() < 0.7:
        shadow = Image.new("RGBA", (target_width + 36, target_height + 36), (0, 0, 0, 0))
        shadow_mask = Image.new("L", shadow.size, 0)
        ImageDraw.Draw(shadow_mask).rounded_rectangle(
            (18, 18, target_width + 17, target_height + 17), radius=radius + 5, fill=150
        )
        shadow.putalpha(shadow_mask.filter(ImageFilter.GaussianBlur(12)))
        screenshot.alpha_composite(shadow, (target_x - 18, target_y - 18))

    target.putalpha(alpha)
    screenshot.alpha_composite(target, (target_x, target_y))
    mask.paste(alpha, (target_x, target_y))
    draw = ImageDraw.Draw(screenshot)
    random_ui(draw, rng, width, height)
    if rng.random() < 0.4:
        x = target_x + rng.randint(12, max(12, target_width - 52))
        y = target_y + rng.randint(12, max(12, target_height - 52))
        draw.ellipse((x, y, x + 34, y + 34), fill=(18, 20, 24, 190))
    return screenshot.convert("RGB"), mask


def generate_split(paths: list[Path], destination: Path, count: int, seed: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for index in range(count):
        image, mask = make_sample(paths, rng)
        image.save(destination / f"{index:06d}.jpg", quality=rng.randint(82, 96), optimize=True)
        mask.save(destination / f"{index:06d}.png", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--train-count", type=int, default=12000)
    parser.add_argument("--validation-count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7319)
    args = parser.parse_args()
    assets = load_assets(args.assets)
    if len(assets) < 20:
        parser.error("At least 20 distinct image assets are required for disjoint training and validation pools.")

    rng = random.Random(args.seed)
    rng.shuffle(assets)
    split = max(4, int(len(assets) * 0.8))
    train_assets, validation_assets = assets[:split], assets[split:]
    generate_split(train_assets, args.output / "train", args.train_count, args.seed)
    generate_split(validation_assets, args.output / "validation", args.validation_count, args.seed + 1)
    print(f"Generated {args.train_count} training and {args.validation_count} validation samples.")
    print("Training and validation use disjoint source-image pools.")


if __name__ == "__main__":
    main()
