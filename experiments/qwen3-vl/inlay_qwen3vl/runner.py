from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol

from PIL import Image, ImageDraw, ImageOps

from .geometry import Box1000, box_1000_to_pixels
from .protocol import CandidateSelection
from .vision import (
    RectangleCandidate,
    draw_candidate_overview,
    generate_candidates,
    refine_rectangle,
)


class VisionClient(Protocol):
    def select_candidate(
        self,
        image: Image.Image,
        candidates: list[RectangleCandidate],
    ) -> CandidateSelection: ...


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source)
        return oriented.convert("RGB")


def _select_candidate(
    image: Image.Image,
    candidates: list[RectangleCandidate],
    client: VisionClient,
) -> tuple[CandidateSelection, str, dict[str, float | str]]:
    ranked = sorted(
        enumerate(candidates, start=1),
        key=lambda item: item[1].score,
        reverse=True,
    )
    best_id, best = ranked[0]
    second_score = ranked[1][1].score if len(ranked) > 1 else 0.0
    score_margin = best.score - second_score

    # A proposal with a strong edge on every side and a clear score lead is
    # more trustworthy than a semantic model's coordinate-free guess.
    # These are conservative starting thresholds, not a trained confidence.
    if (
        best.score >= 0.80
        and best.weakest_edge_support >= 0.55
        and score_margin >= 0.06
    ):
        selection = CandidateSelection(
            best_id,
            (
                "Selected automatically: strong support on all four edges "
                "and a clear lead over the next geometric candidate."
            ),
            "{}",
        )
        return selection, "opencv_high_confidence", {
            "method": "opencv_high_confidence",
            "best_score": best.score,
            "weakest_edge_support": best.weakest_edge_support,
            "margin_over_second": round(score_margin, 4),
        }

    selection = client.select_candidate(image, candidates)
    return selection, "qwen_semantic", {
        "method": "qwen_semantic",
        "best_geometric_score": best.score,
        "weakest_edge_support_of_best": best.weakest_edge_support,
        "margin_over_second": round(score_margin, 4),
    }


def _save_selected_crop(
    image: Image.Image,
    box: Box1000,
    output_dir: Path,
) -> tuple[Path, Path]:
    pixels = box_1000_to_pixels(box, image.width, image.height)
    left, top, right, bottom = pixels
    crop_path = output_dir / "candidate.png"
    image.crop(pixels).save(crop_path)

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    line_width = max(4, round(max(image.size) / 500))
    for offset in range(line_width):
        draw.rectangle(
            (
                left + offset,
                top + offset,
                right - 1 - offset,
                bottom - 1 - offset,
            ),
            outline=(255, 0, 255),
        )
    overlay_path = output_dir / "overlay.png"
    overlay.save(overlay_path)
    return crop_path, overlay_path


def run_experiment(
    image_path: Path,
    output_dir: Path,
    client: VisionClient,
    *,
    max_candidates: int = 8,
    report: Callable[[str], None] = print,
) -> dict[str, Any]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if max_candidates < 2:
        raise ValueError("At least two computer-vision candidates are required.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("crop.png", "candidate.png", "overlay.png", "proposals.png", "report.json"):
        stale = output_dir / name
        if stale.exists():
            stale.unlink()

    image = _load_image(image_path)
    candidates = generate_candidates(image, max_candidates=max_candidates)
    if not candidates:
        raise RuntimeError(
            "OpenCV could not find plausible rectangular image-panel candidates. "
            "No crop was produced."
        )

    overview = draw_candidate_overview(image, candidates)
    proposals_path = output_dir / "proposals.png"
    overview.save(proposals_path)

    report(
        f"OpenCV generated {len(candidates)} candidate rectangles. "
        f"Top geometric score: {max(item.score for item in candidates):.3f}."
    )
    selection, selection_method, selection_metrics = _select_candidate(
        image, candidates, client
    )
    if not 1 <= selection.candidate_id <= len(candidates):
        raise ValueError("The model selected a candidate ID outside the proposal list.")

    selected = candidates[selection.candidate_id - 1]
    refined_box, refinement = refine_rectangle(image, selected.box)
    crop_path, overlay_path = _save_selected_crop(image, refined_box, output_dir)
    selected_pixels = box_1000_to_pixels(
        refined_box,
        image.width,
        image.height,
    )

    proposal_data = []
    for index, candidate in enumerate(candidates, start=1):
        proposal_data.append(
            {
                "candidate_id": index,
                "bbox_1000": list(candidate.box),
                "bbox_pixels_xyxy": list(
                    box_1000_to_pixels(candidate.box, image.width, image.height)
                ),
                "score": candidate.score,
                "edge_support": candidate.edge_support,
                "weakest_edge_support": candidate.weakest_edge_support,
                "projection_support": candidate.projection_support,
                "source": candidate.source,
            }
        )

    result = {
        "input": str(image_path.resolve()),
        "image_size": {"width": image.width, "height": image.height},
        "model": getattr(client, "model_name", None),
        "pipeline": "OpenCV edge/contour proposals -> confidence-gated OpenCV/Qwen selection -> OpenCV edge refinement",
        "selection_method": selection_method,
        "selection_metrics": selection_metrics,
        "status": "selected_candidate",
        "accepted": False,
        "candidate_count": len(candidates),
        "selected_candidate_id": selection.candidate_id,
        "selection_reason": selection.reason,
        "selected_proposal_bbox_1000": list(selected.box),
        "final_bbox_1000": list(refined_box),
        "final_bbox_pixels_xyxy": list(selected_pixels),
        "refinement": refinement,
        "candidates": proposal_data,
        "output_crop": crop_path.name,
        "output_overlay": overlay_path.name,
        "output_proposals": proposals_path.name,
        "note": (
            "Qwen selected one OpenCV-generated rectangle; Python refined its edges "
            "using local Sobel gradients. This is an unconfirmed candidate, not an "
            "accuracy guarantee. Inspect overlay.png and candidate.png before use."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report(
        f"{'OpenCV' if selection_method == 'opencv_high_confidence' else 'Qwen'} "
        f"selected candidate {selection.candidate_id}: "
        f"{selection.reason or 'No explanation returned.'}"
    )
    report(
        f"Refined bbox={list(refined_box)}; edge support "
        f"{refinement['edge_support_before']:.3f} -> "
        f"{refinement['edge_support_after']:.3f}."
    )
    report(f"Status: selected_candidate. Diagnostics: {output_dir.resolve()}")
    return result
