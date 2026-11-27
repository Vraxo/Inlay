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
        description="Find a foreground image panel using OpenCV and Qwen vision selection.",
    )
    parser.add_argument(
        "image", type=Path, help="Screenshot or image to inspect."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("INLAY_QWEN_MODEL", "qwen3vl-8b"),
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
        "--max-candidates",
        type=int,
        default=8,
        help="Maximum number of OpenCV rectangles sent to Qwen (minimum 2).",
    )
    parser.add_argument(
        "--max-image-dimension",
        type=int,
        default=2048,
        help="Maximum side length of images sent to Qwen.",
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
    if args.max_candidates < 2:
        parser.error("--max-candidates must be at least 2.")
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
            max_candidates=args.max_candidates,
        )
    except Exception as error:
        print(f"Experiment failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Selected candidate {result['selected_candidate_id']}/"
        f"{result['candidate_count']}: {result['output_crop']}"
    )
    print(
        "This is an unconfirmed candidate. Inspect overlay.png, candidate.png, "
        "and proposals.png before using it."
    )
    return 2
