from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .client import LMStudioVisionClient
from .lmstudio import ensure_server_ready
from .runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inlay-qwen3vl",
        description="Find and iteratively verify the foreground image panel in a screenshot.",
    )
    parser.add_argument(
        "image", type=Path, help="Screenshot or image to inspect."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("INLAY_QWEN_MODEL", "qwen3vl-8b-thinking"),
        help="API model identifier to assign when loading the local model.",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get(
            "INLAY_LMSTUDIO_API_URL",
            "http://localhost:1234/api/v1/chat",
        ),
        help="LM Studio native chat endpoint.",
    )
    parser.add_argument(
        "--lms-executable",
        default=os.environ.get("INLAY_LMS_EXECUTABLE", "lms"),
        help="Path or command name for the LM Studio CLI.",
    )
    parser.add_argument(
        "--load-model",
        default=os.environ.get("INLAY_LMS_LOAD_MODEL", ""),
        help="Optional exact local model key; discovered automatically when omitted.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of model refinement passes (minimum 2).",
    )
    parser.add_argument(
        "--max-image-dimension",
        type=int,
        default=2048,
        help="Maximum side length of images sent to Qwen (original pixels are preserved for cropping).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Diagnostics output directory (default: <image-name>.qwen3vl).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f"input image does not exist: {args.image}")
    if args.max_iterations < 2:
        parser.error("--max-iterations must be at least 2.")
    if args.max_image_dimension < 256:
        parser.error("--max-image-dimension must be at least 256.")

    output_dir = args.output or args.image.with_name(f"{args.image.stem}.qwen3vl")
    if not ensure_server_ready(
        args.api_url,
        args.model,
        lms_executable=args.lms_executable,
        load_model=args.load_model,
    ):
        return 1

    client = LMStudioVisionClient(
        args.api_url,
        args.model,
        max_image_dimension=args.max_image_dimension,
    )
    try:
        result = run_experiment(
            args.image,
            output_dir,
            client,
            max_iterations=args.max_iterations,
        )
    except Exception as error:
        print(f"Experiment failed: {error}", file=sys.stderr)
        return 1

    if result["accepted"]:
        print(f"Confirmed crop: {result['output_crop']}")
        return 0
    print(
        f"No confirmed crop ({result['status']}). "
        "Inspect candidate.png and overlay.png before using the result."
    )
    return 2
