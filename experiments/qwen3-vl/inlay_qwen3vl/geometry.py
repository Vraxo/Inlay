from __future__ import annotations

import math
from typing import TypeAlias

Box1000: TypeAlias = tuple[int, int, int, int]
PixelBox: TypeAlias = tuple[int, int, int, int]


def validate_box_1000(value: object) -> Box1000:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox_1000 must contain exactly four coordinates.")
    coordinates: list[int] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError("bbox_1000 coordinates must be numbers.")
        if not math.isfinite(float(coordinate)):
            raise ValueError("bbox_1000 coordinates must be finite.")
        rounded = round(coordinate)
        if rounded < 0 or rounded > 1000:
            raise ValueError("bbox_1000 coordinates must be between 0 and 1000.")
        coordinates.append(int(rounded))
    left, top, right, bottom = coordinates
    if right <= left or bottom <= top:
        raise ValueError("bbox_1000 must have positive width and height.")
    return left, top, right, bottom


def box_1000_to_pixels(box: Box1000, width: int, height: int) -> PixelBox:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    left, top, right, bottom = validate_box_1000(box)
    x1 = min(width - 1, max(0, round(left * width / 1000)))
    y1 = min(height - 1, max(0, round(top * height / 1000)))
    x2 = min(width, max(x1 + 1, round(right * width / 1000)))
    y2 = min(height, max(y1 + 1, round(bottom * height / 1000)))
    return x1, y1, x2, y2
