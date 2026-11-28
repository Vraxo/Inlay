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
    draw_sharpness_overview,
    generate_candidates,
    rank_sharpness_boundaries,
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
) -> tuple[CandidateSelection, str, dict[str, Any]]:
    boundary_rankings = rank_sharpness_boundaries(image, candidates)
    sharpness_metrics: dict[str, dict[str, float | int]] = {}
    selected_boundary_ids: dict[str, int] = {}
    sharpness_is_confident = True

    for side, options in boundary_rankings.items():
        best_option = options[0]
        second_score = float(options[1]["score"]) if len(options) > 1 else 0.0
        margin = float(best_option["score"]) - second_score
        sharpness_metrics[side] = {
            "candidate_id": int(best_option["candidate_id"]),
            "coordinate": int(best_option["coordinate"]),
            "score": float(best_option["score"]),
            "contrast": float(best_option["contrast"]),
            "inside_detail": float(best_option["inside_detail"]),
            "outside_detail": float(best_option["outside_detail"]),
            "inside_ratio": float(best_option["inside_ratio"]),
            "margin_over_second": round(margin, 4),
        }
        selected_boundary_ids[side] = int(best_option["candidate_id"])
        if (
            float(best_option["score"]) < 0.08
            or float(best_option["inside_ratio"]) < 0.10
            or (len(options) > 1 and margin < 0.035)
        ):
            sharpness_is_confident = False

    if sharpness_is_confident:
        selection = CandidateSelection(
            None,
            (
                "Selected independently: each boundary has strong sharp-inside "
                "versus blurred-outside detail contrast."
            ),
            "{}",
            left_candidate_id=selected_boundary_ids["left"],
            top_candidate_id=selected_boundary_ids["top"],
            right_candidate_id=selected_boundary_ids["right"],
            bottom_candidate_id=selected_boundary_ids["bottom"],
        )
        return selection, "sharpness_boundary_confidence", {
            "method": "sharpness_boundary_confidence",
            "boundary_evidence": sharpness_metrics,
        }

    ranked = sorted(
        enumerate(candidates, start=1),
        key=lambda item: item[1].score,
        reverse=True,
    )
    best_id, best = ranked[0]
    second_score = ranked[1][1].score if len(ranked) > 1 else 0.0
    score_margin = best.score - second_score

    # A proposal with strong support on every side and a clear score lead is
    # more trustworthy than asking the model to judge ambiguous boundaries.
    # These are conservative starting thresholds, not a trained confidence.
    if (
        len(ranked) >= 2
        and best.score >= 0.80
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
        "sharpness_boundary_evidence": sharpness_metrics,
    }


def _resolve_selected_box(
    selection: CandidateSelection,
    candidates: list[RectangleCandidate],
) -> tuple[Box1000, int | None, dict[str, int] | None]:
    if selection.uses_independent_boundaries:
        boundary_ids = {
            "left": selection.left_candidate_id,
            "top": selection.top_candidate_id,
            "right": selection.right_candidate_id,
            "bottom": selection.bottom_candidate_id,
        }
        if any(
            candidate_id is None or not 1 <= candidate_id <= len(candidates)
            for candidate_id in boundary_ids.values()
        ):
            raise ValueError("A selected boundary references an unknown candidate.")

        box = (
            candidates[selection.left_candidate_id - 1].box[0],
            candidates[selection.top_candidate_id - 1].box[1],
            candidates[selection.right_candidate_id - 1].box[2],
            candidates[selection.bottom_candidate_id - 1].box[3],
        )
        left, top, right, bottom = box
        if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
            raise ValueError(
                "The independently selected boundaries do not form a valid rectangle."
            )

        matching_id = next(
            (
                index
                for index, candidate in enumerate(candidates, start=1)
                if candidate.box == box
            ),
            None,
        )
        return box, matching_id, {
            key: int(candidate_id)
            for key, candidate_id in boundary_ids.items()
        }

    if (
        selection.candidate_id is None
        or not 1 <= selection.candidate_id <= len(candidates)
    ):
        raise ValueError("The selected candidate ID is outside the proposal list.")

    return candidates[selection.candidate_id - 1].box, selection.candidate_id, None


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
    for name in (
        "crop.png",
        "candidate.png",
        "overlay.png",
        "proposals.png",
        "sharpness.png",
        "report.json",
    ):
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

    sharpness_overview = draw_sharpness_overview(image, candidates)
    sharpness_path = output_dir / "sharpness.png"
    sharpness_overview.save(sharpness_path)

    report(
        f"OpenCV generated {len(candidates)} candidate rectangles. "
        f"Top geometric score: {max(item.score for item in candidates):.3f}."
    )
    selection, selection_method, selection_metrics = _select_candidate(
        image, candidates, client
    )
    (
        selected_box,
        selected_candidate_id,
        boundary_candidate_ids,
    ) = _resolve_selected_box(selection, candidates)
    refined_box, refinement = refine_rectangle(image, selected_box)
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
        "pipeline": "OpenCV edge/contour proposals -> confidence-gated OpenCV/Qwen boundary selection with sharpness guidance -> OpenCV edge refinement",
        "selection_method": selection_method,
        "selection_metrics": selection_metrics,
        "status": "selected_candidate",
        "accepted": False,
        "candidate_count": len(candidates),
        "selected_candidate_id": selected_candidate_id,
        "selected_boundary_candidate_ids": boundary_candidate_ids,
        "selection_reason": selection.reason,
        "selected_proposal_bbox_1000": list(selected_box),
        "final_bbox_1000": list(refined_box),
        "final_bbox_pixels_xyxy": list(selected_pixels),
        "refinement": refinement,
        "candidates": proposal_data,
        "output_crop": crop_path.name,
        "output_overlay": overlay_path.name,
        "output_proposals": proposals_path.name,
        "output_sharpness": sharpness_path.name,
        "note": (
            "When geometric confidence is ambiguous, Qwen independently selects the "
            "left, top, right, and bottom boundaries using the original screenshot, "
            "numbered proposals, and a sharpness map. Python combines those edges and "
            "refines them using local Sobel gradients. High-confidence proposals may "
            "be selected directly by OpenCV. This is an unconfirmed candidate, not an "
            "accuracy guarantee. Inspect overlay.png, candidate.png, and sharpness.png."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if boundary_candidate_ids:
        selected_description = (
            "boundaries "
            f"(L{boundary_candidate_ids['left']}, "
            f"T{boundary_candidate_ids['top']}, "
            f"R{boundary_candidate_ids['right']}, "
            f"B{boundary_candidate_ids['bottom']})"
        )
    else:
        selected_description = f"candidate {selected_candidate_id}"

    report(
        f"{'OpenCV' if selection_method == 'opencv_high_confidence' else 'Qwen'} "
        f"selected {selected_description}: "
        f"{selection.reason or 'No explanation returned.'}"
    )
    report(
        f"Refined bbox={list(refined_box)}; edge support "
        f"{refinement['edge_support_before']:.3f} -> "
        f"{refinement['edge_support_after']:.3f}."
    )
    report(f"Status: selected_candidate. Diagnostics: {output_dir.resolve()}")
    return result
