from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import hypot
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .geometry import Box1000, box_1000_to_pixels, pixels_to_box_1000


@dataclass(frozen=True)
class RectangleCandidate:
    box: Box1000
    score: float
    edge_support: float
    weakest_edge_support: float
    projection_support: float
    source: str


@dataclass(frozen=True)
class _Analysis:
    width: int
    height: int
    scale: float
    gray: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    profile_x: np.ndarray
    profile_y: np.ndarray
    normalized_profile_x: np.ndarray
    normalized_profile_y: np.ndarray


COLORS = (
    (255, 75, 75),
    (40, 190, 255),
    (255, 190, 35),
    (70, 220, 130),
    (210, 105, 255),
    (255, 135, 35),
    (70, 220, 230),
    (235, 235, 235),
)


def _detail_energy(gray: np.ndarray) -> np.ndarray:
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return cv2.GaussianBlur(
        np.abs(laplacian),
        (0, 0),
        sigmaX=7,
        sigmaY=7,
    )


def _analyse(image: Image.Image, max_dimension: int = 1600) -> _Analysis:
    if max_dimension < 256:
        raise ValueError("max_dimension must be at least 256.")

    rgb = np.asarray(image.convert("RGB"))
    original_height, original_width = rgb.shape[:2]
    scale = min(1.0, max_dimension / max(original_width, original_height))
    width = max(1, round(original_width * scale))
    height = max(1, round(original_height * scale))

    if scale < 1:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    gradient_x = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))

    # Long outer-panel sides produce coherent responses in these central bands.
    # The corner regions are deliberately excluded from the projection profiles.
    vertical_band = gradient_x[int(0.16 * height):int(0.84 * height), :]
    horizontal_band = gradient_y[:, int(0.06 * width):int(0.94 * width)]
    profile_x = vertical_band.mean(axis=0)
    profile_y = horizontal_band.mean(axis=1)

    normalized_profile_x = _normalize_profile(profile_x)
    normalized_profile_y = _normalize_profile(profile_y)

    return _Analysis(
        width,
        height,
        scale,
        gray,
        gradient_x,
        gradient_y,
        profile_x,
        profile_y,
        normalized_profile_x,
        normalized_profile_y,
    )


def _normalize_profile(profile: np.ndarray) -> np.ndarray:
    reference = max(float(np.percentile(profile, 95)), 1.0)
    return np.clip(profile / reference, 0.0, 1.5) / 1.5


def _find_peaks(profile: np.ndarray, count: int, minimum_distance: int) -> list[int]:
    smooth = cv2.GaussianBlur(
        profile.astype(np.float32).reshape(-1, 1),
        (0, 0),
        sigmaX=0,
        sigmaY=2.0,
    ).ravel()

    peaks = [
        index
        for index in range(2, len(smooth) - 2)
        if smooth[index] >= smooth[index - 1]
        and smooth[index] >= smooth[index + 1]
    ]
    peaks.sort(key=lambda index: float(smooth[index]), reverse=True)

    selected: list[int] = []
    for peak in peaks:
        if all(abs(peak - existing) >= minimum_distance for existing in selected):
            selected.append(peak)
        if len(selected) >= count:
            break

    return selected


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


def _side_support(analysis: _Analysis, box: tuple[int, int, int, int]) -> list[float]:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top

    y_start = int(top + 0.12 * height)
    y_end = int(bottom - 0.12 * height)
    x_start = int(left + 0.12 * width)
    x_end = int(right - 0.12 * width)

    def vertical(coordinate: int) -> float:
        strip = analysis.gradient_x[
            y_start:y_end,
            max(0, coordinate - 2):min(analysis.width, coordinate + 3),
        ]
        if strip.size == 0:
            return 0.0
        return float(np.mean(np.clip(strip / 120.0, 0.0, 1.0)))

    def horizontal(coordinate: int) -> float:
        strip = analysis.gradient_y[
            max(0, coordinate - 2):min(analysis.height, coordinate + 3),
            x_start:x_end,
        ]
        if strip.size == 0:
            return 0.0
        return float(np.mean(np.clip(strip / 120.0, 0.0, 1.0)))

    return [
        vertical(left),
        vertical(right),
        horizontal(top),
        horizontal(bottom),
    ]


def _score_box(
    analysis: _Analysis,
    box: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    left, top, right, bottom = box
    box_width = right - left
    box_height = bottom - top
    if box_width <= 0 or box_height <= 0:
        return 0.0, 0.0, 0.0, 0.0

    sides = _side_support(analysis, box)
    edge_support = float(np.mean(sides))
    weakest_edge = min(sides)

    area_fraction = (box_width * box_height) / (
        analysis.width * analysis.height
    )
    area_score = min(area_fraction / 0.55, 1.0)

    center_x = (left + right) / (2 * analysis.width) - 0.5
    center_y = (top + bottom) / (2 * analysis.height) - 0.5
    center_distance = hypot(center_x, center_y)
    center_score = max(0.0, 1.0 - center_distance / 0.5)

    projection_support = float(
        np.mean(
            (
                analysis.normalized_profile_x[left],
                analysis.normalized_profile_x[right],
                analysis.normalized_profile_y[top],
                analysis.normalized_profile_y[bottom],
            )
        )
    )

    # A rectangle needs evidence on all four sides. The weakest side matters,
    # while broad, centrally placed regions are more plausible photo panels.
    score = (
        0.30 * edge_support
        + 0.15 * weakest_edge
        + 0.25 * projection_support
        + 0.18 * area_score
        + 0.12 * center_score
    )
    return score, edge_support, weakest_edge, projection_support


def _valid_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> bool:
    left, top, right, bottom = box
    box_width = right - left
    box_height = bottom - top

    if left < 0 or top < 0 or right > width or bottom > height:
        return False
    if box_width < 0.25 * width or box_height < 0.20 * height:
        return False
    if box_width > 0.995 * width or box_height > 0.97 * height:
        return False

    aspect_ratio = box_width / box_height
    return 0.25 <= aspect_ratio <= 2.5


def _contour_boxes(analysis: _Analysis) -> Iterable[tuple[int, int, int, int]]:
    edges = cv2.Canny(analysis.gray, 40, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    minimum_area = analysis.width * analysis.height * 0.10
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
        if cv2.contourArea(contour) < minimum_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        yield x, y, x + width, y + height


def _to_candidate(
    analysis: _Analysis,
    pixel_box: tuple[int, int, int, int],
    source: str,
    original_size: tuple[int, int],
) -> RectangleCandidate:
    score, edge_support, weakest_edge, projection_support = _score_box(
        analysis, pixel_box
    )
    original_width, original_height = original_size
    original_box = tuple(
        min(limit, max(0, round(value / analysis.scale)))
        for value, limit in zip(
            pixel_box,
            (original_width, original_height, original_width, original_height),
            strict=True,
        )
    )
    normalized = pixels_to_box_1000(
        original_box,
        original_width,
        original_height,
    )
    return RectangleCandidate(
        normalized,
        round(score, 4),
        round(edge_support, 4),
        round(weakest_edge, 4),
        round(projection_support, 4),
        source,
    )


def generate_candidates(
    image: Image.Image,
    *,
    max_candidates: int = 8,
    max_dimension: int = 1600,
) -> list[RectangleCandidate]:
    if max_candidates < 2:
        raise ValueError("max_candidates must be at least 2.")

    analysis = _analyse(image, max_dimension)
    original_size = image.size
    x_peaks = _find_peaks(
        analysis.profile_x,
        count=18,
        minimum_distance=max(4, round(analysis.width * 0.01)),
    )
    y_peaks = _find_peaks(
        analysis.profile_y,
        count=20,
        minimum_distance=max(4, round(analysis.height * 0.008)),
    )
    if float(np.max(analysis.profile_x)) < 2.0 or float(np.max(analysis.profile_y)) < 2.0:
        return []

    candidates: list[RectangleCandidate] = []
    seen: set[tuple[int, int, int, int]] = set()

    def add_box(
        box: tuple[int, int, int, int],
        source: str,
        *,
        force_first: bool = False,
    ) -> None:
        if not _valid_box(box, analysis.width, analysis.height):
            return

        candidate = _to_candidate(analysis, box, source, original_size)
        if any(
            _intersection_over_union(candidate.box, existing.box) >= 0.94
            for existing in candidates
        ):
            return

        if force_first:
            candidates.insert(0, candidate)
        else:
            candidates.append(candidate)

    # The strongest pair of vertical and horizontal projections is a dedicated
    # proposal. This prevents a weaker local edge score from discarding a long,
    # coherent panel boundary before the semantic model can inspect it.
    strongest_x = sorted(
        x_peaks,
        key=lambda index: float(analysis.profile_x[index]),
        reverse=True,
    )[:2]
    strongest_y = sorted(
        y_peaks,
        key=lambda index: float(analysis.profile_y[index]),
        reverse=True,
    )[:2]
    if len(strongest_x) == 2 and len(strongest_y) == 2:
        dominant_box = (
            min(strongest_x),
            min(strongest_y),
            max(strongest_x),
            max(strongest_y),
        )
        add_box(dominant_box, "dominant-edge-projections", force_first=True)

    generated: list[tuple[float, tuple[int, int, int, int], str]] = []
    for left, right in combinations(sorted(x_peaks), 2):
        for top, bottom in combinations(sorted(y_peaks), 2):
            box = (left, top, right, bottom)
            if not _valid_box(box, analysis.width, analysis.height):
                continue
            score, _, _, _ = _score_box(analysis, box)
            generated.append((score, box, "edge-projection"))

    for _, box, source in sorted(generated, key=lambda item: item[0], reverse=True):
        if len(candidates) >= max_candidates * 4:
            break
        add_box(box, source)

    for box in _contour_boxes(analysis):
        add_box(box, "closed-contour")
        if len(candidates) >= max_candidates * 4:
            break

    # Preserve the dominant projection proposal first; rank the alternatives
    # by the combined edge, projection, size, and centrality score.
    if candidates:
        first = candidates[0]
        alternatives = sorted(candidates[1:], key=lambda item: item.score, reverse=True)
        return [first, *alternatives[:max_candidates - 1]]

    return []


def rank_sharpness_boundaries(
    image: Image.Image,
    candidates: list[RectangleCandidate],
    *,
    max_dimension: int = 1600,
) -> dict[str, list[dict[str, int | float]]]:
    if not candidates:
        raise ValueError("At least one candidate is required.")

    analysis = _analyse(image, max_dimension)
    detail = _detail_energy(analysis.gray)
    reference = max(float(np.percentile(detail, 98)), 1.0)

    scaled_boxes = []
    for candidate_id, candidate in enumerate(candidates, start=1):
        pixels = box_1000_to_pixels(candidate.box, image.width, image.height)
        scaled_boxes.append(
            (
                candidate_id,
                tuple(round(value * analysis.scale) for value in pixels),
                candidate.score,
            )
        )

    def shared_span(
        start_index: int,
        end_index: int,
        length: int,
    ) -> tuple[int, int]:
        start = max(box[1][start_index] for box in scaled_boxes)
        end = min(box[1][end_index] for box in scaled_boxes)
        if end - start >= length * 0.12:
            return start, end

        centers = [
            (box[1][start_index] + box[1][end_index]) / 2
            for box in scaled_boxes
        ]
        sizes = [
            box[1][end_index] - box[1][start_index]
            for box in scaled_boxes
        ]
        center = float(np.median(centers))
        span = min(float(np.median(sizes)) * 0.55, length * 0.65)
        return (
            max(0, round(center - span / 2)),
            min(length, round(center + span / 2)),
        )

    common_x = shared_span(0, 2, analysis.width)
    common_y = shared_span(1, 3, analysis.height)
    gap = max(2, round(min(analysis.width, analysis.height) * 0.003))
    band = max(6, round(min(analysis.width, analysis.height) * 0.02))

    def mean_detail(strip: np.ndarray) -> float:
        return float(np.mean(strip)) if strip.size else 0.0

    def edge_evidence(
        side: str,
        coordinate: int,
    ) -> tuple[float, float, float, float]:
        if side == "left":
            inside = detail[
                common_y[0]:common_y[1],
                coordinate + gap:coordinate + gap + band,
            ]
            outside = detail[
                common_y[0]:common_y[1],
                max(0, coordinate - gap - band):max(0, coordinate - gap),
            ]
        elif side == "right":
            inside = detail[
                common_y[0]:common_y[1],
                max(0, coordinate - gap - band):max(0, coordinate - gap),
            ]
            outside = detail[
                common_y[0]:common_y[1],
                coordinate + gap:coordinate + gap + band,
            ]
        elif side == "top":
            inside = detail[
                coordinate + gap:coordinate + gap + band,
                common_x[0]:common_x[1],
            ]
            outside = detail[
                max(0, coordinate - gap - band):max(0, coordinate - gap),
                common_x[0]:common_x[1],
            ]
        else:
            inside = detail[
                max(0, coordinate - gap - band):max(0, coordinate - gap),
                common_x[0]:common_x[1],
            ]
            outside = detail[
                coordinate + gap:coordinate + gap + band,
                common_x[0]:common_x[1],
            ]

        inside_mean = mean_detail(inside)
        outside_mean = mean_detail(outside)
        contrast = (
            (inside_mean - outside_mean)
            / (inside_mean + outside_mean + 1e-6)
        )
        inside_ratio = inside_mean / reference
        score = max(0.0, contrast) * min(inside_ratio, 1.0)
        return score, contrast, inside_mean, outside_mean

    side_indexes = {
        "left": 0,
        "top": 1,
        "right": 2,
        "bottom": 3,
    }
    rankings: dict[str, list[dict[str, int | float]]] = {}

    for side, coordinate_index in side_indexes.items():
        by_coordinate: dict[int, tuple[int, float]] = {}
        for candidate_id, box, geometry_score in scaled_boxes:
            coordinate = round(box[coordinate_index] / analysis.scale)
            current = by_coordinate.get(coordinate)
            if current is None or geometry_score > current[1]:
                by_coordinate[coordinate] = (candidate_id, geometry_score)

        options = []
        for coordinate, (candidate_id, _) in by_coordinate.items():
            scaled_coordinate = round(coordinate * analysis.scale)
            score, contrast, inside_mean, outside_mean = edge_evidence(
                side,
                scaled_coordinate,
            )
            options.append(
                {
                    "candidate_id": candidate_id,
                    "coordinate": coordinate,
                    "score": round(score, 4),
                    "contrast": round(contrast, 4),
                    "inside_detail": round(inside_mean, 3),
                    "outside_detail": round(outside_mean, 3),
                    "inside_ratio": round(inside_mean / reference, 4),
                }
            )

        rankings[side] = sorted(
            options,
            key=lambda option: float(option["score"]),
            reverse=True,
        )

    return rankings


def draw_candidate_overview(
    image: Image.Image,
    candidates: list[RectangleCandidate],
) -> Image.Image:
    overview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overview)
    line_width = max(3, round(max(image.size) / 500))

    for index, candidate in enumerate(candidates):
        left, top, right, bottom = box_1000_to_pixels(
            candidate.box,
            overview.width,
            overview.height,
        )
        color = COLORS[index % len(COLORS)]
        draw.rectangle(
            (left, top, right - 1, bottom - 1),
            outline=color,
            width=line_width,
        )

        label = str(index + 1)
        padding = max(4, line_width)
        label_width = max(24, int(draw.textlength(label)) + padding * 2)
        label_height = max(22, line_width * 5)
        label_left = min(max(0, left), max(0, overview.width - label_width))
        label_top = min(max(0, top), max(0, overview.height - label_height))
        draw.rounded_rectangle(
            (
                label_left,
                label_top,
                label_left + label_width,
                label_top + label_height,
            ),
            radius=max(3, line_width),
            fill=color,
        )
        draw.text(
            (label_left + padding, label_top + max(1, padding // 2)),
            label,
            fill=(15, 18, 22),
        )

    return overview


def draw_sharpness_overview(
    image: Image.Image,
    candidates: list[RectangleCandidate],
) -> Image.Image:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    local_detail = _detail_energy(gray)
    reference = max(float(np.percentile(local_detail, 98)), 1.0)
    normalized = np.clip(local_detail / reference, 0.0, 1.0)
    heatmap = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO,
    )
    overview = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(overview)
    line_width = max(2, round(max(image.size) / 700))

    for index, candidate in enumerate(candidates):
        left, top, right, bottom = box_1000_to_pixels(
            candidate.box,
            overview.width,
            overview.height,
        )
        color = COLORS[index % len(COLORS)]
        draw.rectangle(
            (left, top, right - 1, bottom - 1),
            outline=color,
            width=line_width,
        )

        label = str(index + 1)
        padding = max(4, line_width)
        label_width = max(24, int(draw.textlength(label)) + padding * 2)
        label_height = max(22, line_width * 5)
        label_left = min(max(0, left), max(0, overview.width - label_width))
        label_top = min(max(0, top), max(0, overview.height - label_height))
        draw.rounded_rectangle(
            (
                label_left,
                label_top,
                label_left + label_width,
                label_top + label_height,
            ),
            radius=max(3, line_width),
            fill=color,
        )
        draw.text(
            (label_left + padding, label_top + max(1, padding // 2)),
            label,
            fill=(15, 18, 22),
        )

    footer_height = max(34, round(max(image.size) / 60))
    canvas = Image.new(
        "RGB",
        (overview.width, overview.height + footer_height),
        (20, 20, 20),
    )
    canvas.paste(overview, (0, 0))
    footer = ImageDraw.Draw(canvas)
    footer.text(
        (10, overview.height + max(8, footer_height // 4)),
        "SHARPNESS MAP  |  Brighter = sharper detail, darker = blurrier",
        fill=(245, 245, 245),
    )
    return canvas


def draw_boundary_options_overview(
    image: Image.Image,
    rankings: dict[str, list[dict[str, int | float]]],
) -> Image.Image:
    tile_width = max(320, image.width // 2)
    tile_height = max(240, round(image.height * tile_width / image.width))
    header_height = max(42, round(tile_width * 0.09))
    canvas = Image.new(
        "RGB",
        (tile_width * 2, (tile_height + header_height) * 2),
        (18, 20, 24),
    )
    font = None
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=max(14, round(tile_width * 0.035)))
    except (AttributeError, TypeError):
        pass

    panels = (
        ("left", "LEFT EDGE OPTIONS", 0, 0),
        ("right", "RIGHT EDGE OPTIONS", 1, 0),
        ("top", "TOP EDGE OPTIONS", 0, 1),
        ("bottom", "BOTTOM EDGE OPTIONS", 1, 1),
    )
    for side, title, column, row in panels:
        x_offset = column * tile_width
        y_offset = row * (tile_height + header_height)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            (x_offset, y_offset, x_offset + tile_width, y_offset + header_height),
            fill=(28, 32, 40),
        )
        draw.text(
            (x_offset + 10, y_offset + 10),
            title,
            fill=(245, 245, 245),
            font=font,
        )

        view = image.convert("RGB").resize(
            (tile_width, tile_height),
            Image.Resampling.LANCZOS,
        )
        canvas.paste(view, (x_offset, y_offset + header_height))
        draw = ImageDraw.Draw(canvas)
        options = rankings[side]
        for option_index, option in enumerate(options):
            coordinate = int(option["coordinate"])
            candidate_id = int(option["candidate_id"])
            color = COLORS[option_index % len(COLORS)]
            label = f"ID {candidate_id}  {coordinate}px"

            if side in {"left", "right"}:
                line_x = round(coordinate * tile_width / image.width)
                draw.line(
                    (
                        x_offset + line_x,
                        y_offset + header_height,
                        x_offset + line_x,
                        y_offset + header_height + tile_height,
                    ),
                    fill=color,
                    width=max(2, round(tile_width / 270)),
                )
                label_x = min(
                    max(0, line_x + 5),
                    max(0, tile_width - 115),
                )
                label_y = y_offset + header_height + 8 + option_index * 26
            else:
                line_y = round(coordinate * tile_height / image.height)
                draw.line(
                    (
                        x_offset,
                        y_offset + header_height + line_y,
                        x_offset + tile_width,
                        y_offset + header_height + line_y,
                    ),
                    fill=color,
                    width=max(2, round(tile_width / 270)),
                )
                label_x = x_offset + 8 + option_index * 105
                label_y = min(
                    y_offset + header_height + line_y + 4,
                    y_offset + header_height + tile_height - 22,
                )

            label_width = 105
            label_height = 22
            draw.rounded_rectangle(
                (
                    label_x,
                    label_y,
                    label_x + label_width,
                    label_y + label_height,
                ),
                radius=4,
                fill=(18, 20, 24),
            )
            draw.text(
                (label_x + 5, label_y + 4),
                label,
                fill=color,
                font=font,
            )

    return canvas


def _coordinate_support(
    gradient: np.ndarray,
    coordinate: int,
    start: int,
    end: int,
    *,
    vertical: bool,
) -> float:
    if vertical:
        strip = gradient[
            start:end,
            max(0, coordinate - 2):min(gradient.shape[1], coordinate + 3),
        ]
    else:
        strip = gradient[
            max(0, coordinate - 2):min(gradient.shape[0], coordinate + 3),
            start:end,
        ]
    if strip.size == 0:
        return 0.0
    return float(np.mean(np.clip(strip / 120.0, 0.0, 1.0)))


def refine_rectangle(
    image: Image.Image,
    box: Box1000,
    *,
    max_dimension: int = 1600,
) -> tuple[Box1000, dict[str, object]]:
    analysis = _analyse(image, max_dimension)
    original_pixels = box_1000_to_pixels(box, image.width, image.height)
    scaled = tuple(round(value * analysis.scale) for value in original_pixels)
    left, top, right, bottom = scaled

    original_side_support = _side_support(analysis, scaled)
    x_radius = max(2, round(analysis.width * 0.006))
    y_radius = max(2, round(analysis.height * 0.006))

    def best_vertical(current: int) -> int:
        y_start = int(top + 0.12 * (bottom - top))
        y_end = int(bottom - 0.12 * (bottom - top))
        options = range(
            max(1, current - x_radius),
            min(analysis.width - 1, current + x_radius) + 1,
        )
        return max(
            options,
            key=lambda value: (
                _coordinate_support(
                    analysis.gradient_x,
                    value,
                    y_start,
                    y_end,
                    vertical=True,
                )
                - abs(value - current) * 0.0005
            ),
        )

    def best_horizontal(current: int) -> int:
        x_start = int(left + 0.12 * (right - left))
        x_end = int(right - 0.12 * (right - left))
        options = range(
            max(1, current - y_radius),
            min(analysis.height - 1, current + y_radius) + 1,
        )
        return max(
            options,
            key=lambda value: (
                _coordinate_support(
                    analysis.gradient_y,
                    value,
                    x_start,
                    x_end,
                    vertical=False,
                )
                - abs(value - current) * 0.0005
            ),
        )

    refined = (
        best_vertical(left),
        best_horizontal(top),
        best_vertical(right),
        best_horizontal(bottom),
    )
    if refined[2] <= refined[0] or refined[3] <= refined[1]:
        refined = scaled

    original_refined = tuple(
        min(limit, max(0, round(value / analysis.scale)))
        for value, limit in zip(
            refined,
            (image.width, image.height, image.width, image.height),
            strict=True,
        )
    )
    refined_box = pixels_to_box_1000(
        original_refined,
        image.width,
        image.height,
    )
    refined_side_support = _side_support(analysis, refined)

    return refined_box, {
        "method": "local Sobel edge snapping",
        "search_radius_pixels": {
            "horizontal_edges": round(y_radius / analysis.scale),
            "vertical_edges": round(x_radius / analysis.scale),
        },
        "adjustment_pixels_xyxy": [
            round((new - old) / analysis.scale)
            for new, old in zip(refined, scaled, strict=True)
        ],
        "edge_support_before": round(float(np.mean(original_side_support)), 4),
        "edge_support_after": round(float(np.mean(refined_side_support)), 4),
    }
