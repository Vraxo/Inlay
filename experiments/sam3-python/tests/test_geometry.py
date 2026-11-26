import numpy as np
import pytest

from inlay_sam3.geometry import (
    box_xyxy_to_xywh,
    mask_bounds,
    pixel_box_to_normalized_center,
)


def test_pixel_box_converts_to_normalized_center():
    assert pixel_box_to_normalized_center((10, 20, 30, 40), 100, 100) == [
        0.25, 0.4, 0.3, 0.4
    ]


def test_pixel_box_is_clipped_to_image():
    assert pixel_box_to_normalized_center((-10, 10, 30, 30), 100, 100) == [
        0.1, 0.25, 0.2, 0.3
    ]


def test_invalid_box_is_rejected():
    with pytest.raises(ValueError):
        pixel_box_to_normalized_center((0, 0, 0, 10), 100, 100)


def test_box_xyxy_converts_to_xywh():
    assert box_xyxy_to_xywh((2, 3, 12, 23)) == (2, 3, 10, 20)


def test_mask_bounds_returns_tight_rectangle():
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:8] = True
    assert mask_bounds(mask) == (3, 2, 5, 4)


def test_empty_mask_has_no_bounds():
    assert mask_bounds(np.zeros((4, 4), dtype=bool)) is None
