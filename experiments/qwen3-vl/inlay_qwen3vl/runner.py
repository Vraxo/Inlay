from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol

from PIL import Image, ImageDraw, ImageOps

from .geometry import Box1000, box_1000_to_pixels
from .protocol import ModelReply


class VisionClient(Protocol):
    def inspect(
        self,
        image: Image.Image,
        current_box: Box1000 | None,
        iteration: int,
    ) -> ModelReply: ...


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source)
        return oriented.convert("RGB")


def _save_diagnostics(
    image: Image.Image,
    box: Box1000,
    output_dir: Path,
    iteration: int | None = None,
) -> tuple[Path, Path]:
    pixels = box_1000_to_pixels(box, image.width, image.height)
    left, top, right, bottom = pixels
    crop = image.crop(pixels)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    line_width = max(4, round(max(image.size) / 500))
    for offset in range(line_width):
        draw.rectangle(
            (left + offset, top + offset, right - 1 - offset, bottom - 1 - offset),
            outline=(255, 0, 255),
        )
    prefix = f"iteration-{iteration:02d}-" if iteration is not None else ""
    overlay_path = output_dir / f"{prefix}overlay.png"
    crop_path = output_dir / f"{prefix}candidate.png"
    overlay.save(overlay_path)
    crop.save(crop_path)
    return overlay_path, crop_path


def run_experiment(
    image_path: Path,
    output_dir: Path,
    client: VisionClient,
    *,
    max_iterations: int = 5,
    report: Callable[[str], None] = print,
) -> dict[str, Any]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if max_iterations < 2:
        raise ValueError("At least two iterations are required for visual verification.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("crop.png", "candidate.png", "overlay.png", "report.json"):
        stale = output_dir / name
        if stale.exists():
            stale.unlink()
    for stale in output_dir.glob("iteration-*-*.png"):
        stale.unlink()

    image = _load_image(image_path)
    current_box: Box1000 | None = None
    iterations: list[dict[str, Any]] = []
    no_progress_count = 0
    status = "max_iterations"

    for iteration in range(1, max_iterations + 1):
        previous_box = current_box
        reply = client.inspect(image, current_box, iteration)
        current_box = reply.box
        changed = previous_box is None or current_box != previous_box
        overlay_path, candidate_path = _save_diagnostics(
            image, current_box, output_dir, iteration
        )
        pixels = box_1000_to_pixels(current_box, image.width, image.height)

        iterations.append(
            {
                "iteration": iteration,
                "bbox_1000": list(current_box),
                "bbox_pixels_xyxy": list(pixels),
                "done": reply.done,
                "changed_from_previous": changed,
                "reason": reply.reason,
                "raw_response": reply.raw_text,
                "overlay_file": overlay_path.name,
                "candidate_file": candidate_path.name,
            }
        )
        report(
            f"Pass {iteration}/{max_iterations}: bbox={list(current_box)}, "
            f"done={reply.done} — {reply.reason or 'No explanation returned.'}"
        )

        # A first localization is never accepted. At least one later pass
        # must independently confirm the exact same rectangle.
        if previous_box is not None and not changed and reply.done:
            status = "accepted"
            break

        if changed:
            no_progress_count = 0
        else:
            no_progress_count += 1
            if no_progress_count >= 2 and not reply.done:
                status = "stalled"
                break

    overlay_path, candidate_path = _save_diagnostics(image, current_box, output_dir)
    accepted = status == "accepted"
    if accepted:
        final_crop_path = output_dir / "crop.png"
        candidate_path.replace(final_crop_path)
    else:
        final_crop_path = candidate_path

    final_overlay_path = overlay_path
    result = {
        "input": str(image_path.resolve()),
        "image_size": {"width": image.width, "height": image.height},
        "model": getattr(client, "model_name", None),
        "status": status,
        "accepted": accepted,
        "max_iterations": max_iterations,
        "final_bbox_1000": list(current_box),
        "final_bbox_pixels_xyxy": list(box_1000_to_pixels(current_box, image.width, image.height)),
        "output_crop": final_crop_path.name,
        "output_overlay": final_overlay_path.name,
        "iterations": iterations,
        "note": (
            "Accepted means the model confirmed an unchanged rectangle on a later pass. "
            "It is not a measured accuracy guarantee. Compare the crop against labelled "
            "ground truth before using this as an automatic extraction decision."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report(f"Status: {status}. Diagnostics: {output_dir.resolve()}")
    return result
