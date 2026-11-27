from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Callable, Protocol

from PIL import Image, ImageDraw, ImageOps

from .geometry import Box1000, box_1000_to_pixels
from .protocol import ModelReply


STABILITY_WINDOW = 4
STABILITY_MIN_IOU = 0.95


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


def _intersection_over_union(first: Box1000, second: Box1000) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _stable_consensus(
    boxes: list[Box1000],
) -> tuple[Box1000, dict[str, Any]] | None:
    if len(boxes) < STABILITY_WINDOW:
        return None

    recent = boxes[-STABILITY_WINDOW:]
    overlaps = [
        _intersection_over_union(recent[first], recent[second])
        for first in range(len(recent))
        for second in range(first + 1, len(recent))
    ]
    minimum_iou = min(overlaps)
    if minimum_iou < STABILITY_MIN_IOU:
        return None

    consensus = tuple(
        int(round(median(box[index] for box in recent)))
        for index in range(4)
    )
    coordinate_spread = [
        max(box[index] for box in recent) - min(box[index] for box in recent)
        for index in range(4)
    ]
    return consensus, {
        "method": "coordinate-wise median of the last four proposals",
        "window_size": STABILITY_WINDOW,
        "minimum_pairwise_iou": round(minimum_iou, 4),
        "minimum_iou_threshold": STABILITY_MIN_IOU,
        "coordinate_spread_1000": coordinate_spread,
    }


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
    boxes: list[Box1000] = []
    iterations: list[dict[str, Any]] = []
    no_progress_count = 0
    status = "max_iterations"
    stable_box: Box1000 | None = None
    stability: dict[str, Any] | None = None

    for iteration in range(1, max_iterations + 1):
        previous_box = current_box
        reply = client.inspect(image, current_box, iteration)
        current_box = reply.box
        boxes.append(current_box)
        changed = previous_box is None or current_box != previous_box
        overlap = (
            _intersection_over_union(previous_box, current_box)
            if previous_box is not None
            else None
        )
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
                "iou_with_previous": round(overlap, 4) if overlap is not None else None,
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

        # Some vision models keep nudging coordinates while describing the
        # same panel. Preserve that as an explicitly unconfirmed consensus
        # candidate instead of wasting every remaining review pass.
        stable = _stable_consensus(boxes)
        if stable is not None:
            stable_box, stability = stable
            status = "stable_candidate"
            report(
                f"Stable candidate: last {STABILITY_WINDOW} proposals have "
                f"minimum pairwise IoU {stability['minimum_pairwise_iou']:.4f}. "
                "The model did not explicitly confirm the crop."
            )
            break

    final_box = stable_box or current_box
    overlay_path, candidate_path = _save_diagnostics(image, final_box, output_dir)
    accepted = status == "accepted"
    if accepted:
        final_crop_path = output_dir / "crop.png"
        candidate_path.replace(final_crop_path)
    else:
        final_crop_path = candidate_path

    if accepted:
        note = (
            "Accepted means the model confirmed an unchanged rectangle on a later pass. "
            "It is not a measured accuracy guarantee. Compare the crop against labelled "
            "ground truth before using this as an automatic extraction decision."
        )
    elif status == "stable_candidate":
        note = (
            "Stable candidate means the last four proposals had pairwise IoU of at least "
            "95 percent. The final rectangle is their coordinate-wise median. The model "
            "did not explicitly confirm correctness, so this remains candidate.png and "
            "must be visually checked before use."
        )
    else:
        note = (
            "The model did not confirm a stable rectangle. candidate.png is an unconfirmed "
            "proposal and must be visually checked before use."
        )

    result = {
        "input": str(image_path.resolve()),
        "image_size": {"width": image.width, "height": image.height},
        "model": getattr(client, "model_name", None),
        "status": status,
        "accepted": accepted,
        "max_iterations": max_iterations,
        "final_bbox_1000": list(final_box),
        "final_bbox_pixels_xyxy": list(
            box_1000_to_pixels(final_box, image.width, image.height)
        ),
        "stability": stability,
        "output_crop": final_crop_path.name,
        "output_overlay": overlay_path.name,
        "iterations": iterations,
        "note": note,
    }
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report(f"Status: {status}. Diagnostics: {output_dir.resolve()}")
    return result
