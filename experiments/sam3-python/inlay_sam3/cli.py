from __future__ import annotations

import argparse
from pathlib import Path

from .runner import DEFAULT_PROMPTS, run_experiment


def _parse_box(value: str) -> tuple[float, float, float, float]:
    try:
        parts = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Box must be x,y,width,height in image pixels."
        ) from error
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise argparse.ArgumentTypeError(
            "Box must be x,y,width,height with positive width and height."
        )
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inlay-sam3",
        description="Run SAM 3.1 text- and box-guided image segmentation.",
    )
    parser.add_argument("image", type=Path, help="Screenshot or image to inspect.")
    parser.add_argument(
        "--out",
        type=Path,
        help="Output directory (default: <image-name>.sam3)",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Text prompt to test; repeat to compare prompts.",
    )
    parser.add_argument(
        "--box",
        type=_parse_box,
        help="Optional seed rectangle in source pixels: x,y,width,height.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.out or args.image.with_name(f"{args.image.stem}.sam3")
    prompts = tuple(args.prompts) if args.prompts else DEFAULT_PROMPTS
    try:
        run_experiment(args.image, output_dir, prompts, args.box)
    except Exception as error:
        print(f"ERROR: {error}")
        return 1
    return 0
