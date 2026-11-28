from PIL import Image, ImageDraw, ImageFilter

from inlay_qwen3vl.geometry import box_1000_to_pixels, pixels_to_box_1000
from inlay_qwen3vl.vision import (
    draw_boundary_options_overview,
    draw_sharpness_overview,
    generate_candidates,
    rank_sharpness_boundaries,
    refine_rectangle,
    RectangleCandidate,
)


def make_screenshot():
    width, height = 600, 1000
    image = Image.new("RGB", (width, height), (20, 24, 28))
    draw = ImageDraw.Draw(image)
    draw.ellipse((25, 115, 150, 240), fill=(42, 46, 50))
    draw.rectangle((0, 0, width - 1, 60), fill=(10, 14, 18))
    draw.rectangle((0, 930, width - 1, 999), fill=(12, 16, 20))

    panel = Image.new("RGB", (480, 600), (110, 140, 170))
    panel_pixels = panel.load()
    for y in range(panel.height):
        for x in range(panel.width):
            tone = (x * 3 + y * 2) % 35
            panel_pixels[x, y] = (
                90 + tone,
                125 + tone,
                155 + tone,
            )

    mask = Image.new("L", panel.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, panel.width - 1, panel.height - 1),
        radius=40,
        fill=255,
    )
    image.paste(panel.filter(ImageFilter.GaussianBlur(0.5)), (60, 220), mask)
    return image


def _iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def test_detects_the_complete_rounded_photo_panel():
    image = make_screenshot()
    candidates = generate_candidates(image, max_candidates=8)

    assert len(candidates) >= 2
    expected = (60, 220, 540, 820)
    candidate_pixels = [
        box_1000_to_pixels(candidate.box, image.width, image.height)
        for candidate in candidates
    ]
    assert any(_iou(box, expected) >= 0.85 for box in candidate_pixels)


def test_dominant_projection_candidate_is_kept_for_semantic_review():
    candidates = generate_candidates(make_screenshot(), max_candidates=8)

    assert candidates
    assert candidates[0].source == "dominant-edge-projections"


def test_candidate_generation_is_deterministic():
    image = make_screenshot()
    first = generate_candidates(image)
    second = generate_candidates(image)

    assert [candidate.box for candidate in first] == [
        candidate.box for candidate in second
    ]


def test_refinement_stays_near_the_selected_rectangle():
    image = make_screenshot()
    candidates = generate_candidates(image)
    original = candidates[0].box

    refined, details = refine_rectangle(image, original)
    original_pixels = box_1000_to_pixels(original, image.width, image.height)
    refined_pixels = box_1000_to_pixels(refined, image.width, image.height)

    assert all(
        abs(after - before) <= 8
        for before, after in zip(original_pixels, refined_pixels, strict=True)
    )
    assert details["method"] == "local Sobel edge snapping"



def test_sharpness_overview_marks_candidate_boundaries():
    image = make_screenshot()
    candidates = generate_candidates(image)
    overview = draw_sharpness_overview(image, candidates)

    assert overview.mode == "RGB"
    assert overview.width == image.width
    assert overview.height > image.height
    sharp_photo_detail = sum(overview.getpixel((300, 500)))
    blurred_surround_detail = sum(overview.getpixel((300, 100)))
    assert sharp_photo_detail > blurred_surround_detail



def test_sharpness_contrast_ranks_the_actual_photo_edges_first():
    image = make_screenshot()
    candidates = [
        RectangleCandidate(
            pixels_to_box_1000((60, 60, 540, 940), *image.size),
            0.70, 0.5, 0.2, 0.8, "test",
        ),
        RectangleCandidate(
            pixels_to_box_1000((60, 220, 540, 820), *image.size),
            0.68, 0.5, 0.3, 0.8, "test",
        ),
        RectangleCandidate(
            pixels_to_box_1000((150, 60, 450, 940), *image.size),
            0.67, 0.5, 0.2, 0.8, "test",
        ),
        RectangleCandidate(
            pixels_to_box_1000((60, 300, 540, 820), *image.size),
            0.66, 0.5, 0.3, 0.8, "test",
        ),
        RectangleCandidate(
            pixels_to_box_1000((60, 220, 450, 820), *image.size),
            0.65, 0.5, 0.3, 0.8, "test",
        ),
    ]

    rankings = rank_sharpness_boundaries(image, candidates)

    assert rankings["left"][0]["coordinate"] == 60
    assert rankings["top"][0]["coordinate"] == 220
    assert rankings["right"][0]["coordinate"] == 540
    assert rankings["bottom"][0]["coordinate"] == 820


def test_boundary_options_overview_separates_the_four_edge_directions():
    image = make_screenshot()
    candidates = generate_candidates(image)
    rankings = rank_sharpness_boundaries(image, candidates)

    overview = draw_boundary_options_overview(image, rankings)

    assert overview.mode == "RGB"
    assert overview.width >= image.width
    assert overview.height > image.height
