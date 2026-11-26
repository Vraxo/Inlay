from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pixel_box_to_normalized_center(
    box: Sequence[float], image_width: int, image_height: int
) -> list[float]:
    """Convert x, y, width, height pixels to SAM's normalized cx, cy, w, h."""
    if len(box) != 4:
        raise ValueError("A box must contain x, y, width, and height.")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    x, y, width, height = (float(value) for value in box)
    if width <= 0 or height <= 0:
        raise ValueError("Box width and height must be positive.")

    left = min(max(x, 0.0), float(image_width))
    top = min(max(y, 0.0), float(image_height))
    right = min(max(x + width, 0.0), float(image_width))
    bottom = min(max(y + height, 0.0), float(image_height))
    if right <= left or bottom <= top:
        raise ValueError("The box does not overlap the image.")

    return [
        (left + right) / (2 * image_width),
        (top + bottom) / (2 * image_height),
        (right - left) / image_width,
        (bottom - top) / image_height,
    ]


def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the tight x, y, width, height bounds of a binary mask."""
    binary = np.asarray(mask, dtype=bool).squeeze()
    if binary.ndim != 2:
        raise ValueError("A mask must be a two-dimensional array.")

    rows, columns = np.nonzero(binary)
    if len(columns) == 0:
        return None

    left = int(columns.min())
    top = int(rows.min())
    right = int(columns.max()) + 1
    bottom = int(rows.max()) + 1
    return left, top, right - left, bottom - top


def box_xyxy_to_xywh(box: Sequence[float]) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError("A box must contain x0, y0, x1, and y1.")
    x0, y0, x1, y1 = (float(value) for value in box)
    left, top = int(round(x0)), int(round(y0))
    right, bottom = int(round(x1)), int(round(y1))
    return left, top, max(0, right - left), max(0, bottom - top)
