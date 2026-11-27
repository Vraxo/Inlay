import pytest

from inlay_qwen3vl.geometry import box_1000_to_pixels, validate_box_1000


def test_validates_and_rounds_normalized_coordinates():
    assert validate_box_1000([100.4, 200, 899.6, 800]) == (100, 200, 900, 800)


@pytest.mark.parametrize(
    "value",
    [
        [0, 0, 0, 100],
        [0, 0, 100, 0],
        [-1, 0, 100, 100],
        [0, 0, 1001, 100],
        [0, 0, float("nan"), 100],
        [0, 0, True, 100],
        [0, 0, 100],
    ],
)
def test_rejects_invalid_normalized_coordinates(value):
    with pytest.raises(ValueError):
        validate_box_1000(value)


def test_converts_normalized_box_to_exclusive_pixel_bounds():
    assert box_1000_to_pixels((100, 200, 900, 800), 1000, 500) == (
        100, 100, 900, 400
    )


def test_box_at_image_edges_stays_inside_image():
    assert box_1000_to_pixels((999, 999, 1000, 1000), 100, 50) == (
        99, 49, 100, 50
    )
