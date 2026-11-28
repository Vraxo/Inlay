from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from inlay_qwen3vl.protocol import CandidateSelection
from inlay_qwen3vl.runner import (
    _resolve_selected_box,
    _save_selected_crop,
    _select_candidate,
    run_experiment,
)
from inlay_qwen3vl.vision import RectangleCandidate


class FakeVisionClient:
    model_name = "fake-qwen"

    def __init__(self, candidate_id=1):
        self.candidate_id = candidate_id
        self.candidates = None

    def select_candidate(self, image, candidates):
        self.candidates = candidates
        return CandidateSelection(
            self.candidate_id,
            "Selected the complete foreground image panel.",
            "{}",
        )


def make_image(path: Path):
    image = Image.new("RGB", (600, 1000), (20, 24, 28))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 599, 999), fill=(24, 28, 32))
    panel = Image.new("RGB", (480, 600), (110, 140, 170))
    mask = Image.new("L", panel.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 479, 599), radius=40, fill=255)
    image.paste(panel.filter(ImageFilter.GaussianBlur(0.5)), (60, 220), mask)
    image.save(path)


def test_runs_opencv_proposals_qwen_selection_and_edge_refinement(tmp_path: Path):
    source = tmp_path / "source.png"
    output = tmp_path / "output"
    make_image(source)
    client = FakeVisionClient()

    result = run_experiment(
        source,
        output,
        client,
        report=lambda _: None,
    )

    assert result["status"] == "selected_candidate"
    assert result["accepted"] is False
    assert result["pipeline"].startswith("OpenCV edge/contour proposals")
    assert result["candidate_count"] >= 2
    assert result["selected_candidate_id"] == 1
    assert result["selection_method"] in {
        "opencv_high_confidence",
        "qwen_semantic",
        "sharpness_boundary_confidence",
    }
    assert result["refinement"]["method"] == "local Sobel edge snapping"
    assert (output / "candidate.png").is_file()
    assert (output / "overlay.png").is_file()
    assert (output / "proposals.png").is_file()
    assert (output / "sharpness.png").is_file()
    assert (output / "boundary_options.png").is_file()
    assert (output / "report.json").is_file()
    assert not (output / "crop.png").exists()



def test_high_confidence_opencv_rectangle_overrides_weak_qwen_choice():
    image = Image.new("RGB", (1080, 2400), "white")
    candidates = [
        RectangleCandidate((31, 219, 969, 781), 0.8401, 0.688, 0.6098, 1.0, "edge"),
        RectangleCandidate((31, 219, 969, 987), 0.7543, 0.558, 0.4102, 1.0, "edge"),
        RectangleCandidate((31, 29, 969, 781), 0.7035, 0.4773, 0.2213, 1.0, "edge"),
        RectangleCandidate((31, 29, 969, 987), 0.7003, 0.3966, 0.2213, 1.0, "edge"),
    ]

    class ModelMustNotOverride:
        def select_candidate(self, image, candidates):
            raise AssertionError("Qwen should not override strong four-edge evidence.")

    selection, method, metrics = _select_candidate(
        image, candidates, ModelMustNotOverride()
    )

    assert selection.candidate_id == 1
    assert method == "opencv_high_confidence"
    assert metrics["best_score"] == 0.8401
    assert metrics["weakest_edge_support"] == 0.6098
    assert metrics["margin_over_second"] == 0.0858


def test_ambiguous_opencv_rectangles_are_sent_to_qwen():
    image = Image.new("RGB", (600, 1000), "white")
    candidates = [
        RectangleCandidate((40, 100, 960, 900), 0.76, 0.60, 0.39, 0.8, "edge"),
        RectangleCandidate((60, 120, 940, 880), 0.74, 0.59, 0.38, 0.8, "edge"),
    ]
    client = FakeVisionClient(candidate_id=2)

    selection, method, metrics = _select_candidate(image, candidates, client)

    assert selection.candidate_id == 2
    assert method == "qwen_semantic"
    assert metrics["method"] == "qwen_semantic"



def test_independent_boundaries_can_form_a_new_rectangle():
    candidates = [
        RectangleCandidate((31, 29, 969, 987), 0.70, 0.50, 0.20, 0.80, "test"),
        RectangleCandidate((31, 289, 969, 711), 0.68, 0.50, 0.30, 0.80, "test"),
        RectangleCandidate((83, 29, 914, 987), 0.67, 0.50, 0.20, 0.80, "test"),
    ]
    selection = CandidateSelection(
        None,
        "Selected each edge independently.",
        "{}",
        left_candidate_id=1,
        top_candidate_id=2,
        right_candidate_id=1,
        bottom_candidate_id=2,
    )

    selected_box, matching_id, boundary_ids = _resolve_selected_box(
        selection,
        candidates,
    )

    assert selected_box == (31, 289, 969, 711)
    assert matching_id == 2
    assert boundary_ids == {"left": 1, "top": 2, "right": 1, "bottom": 2}


def test_independent_boundaries_can_create_a_rectangle_not_in_candidates():
    candidates = [
        RectangleCandidate((31, 29, 969, 987), 0.70, 0.50, 0.20, 0.80, "test"),
        RectangleCandidate((83, 289, 914, 711), 0.68, 0.50, 0.30, 0.80, "test"),
    ]
    selection = CandidateSelection(
        None,
        "The best edges come from different proposals.",
        "{}",
        left_candidate_id=1,
        top_candidate_id=2,
        right_candidate_id=1,
        bottom_candidate_id=2,
    )

    selected_box, matching_id, boundary_ids = _resolve_selected_box(
        selection,
        candidates,
    )

    assert selected_box == (31, 289, 969, 711)
    assert matching_id is None
    assert boundary_ids == {"left": 1, "top": 2, "right": 1, "bottom": 2}



def test_sharpness_contrast_selects_all_four_boundaries_without_qwen():
    from inlay_qwen3vl.geometry import pixels_to_box_1000

    image = Image.new("RGB", (600, 1000), (20, 24, 28))
    panel = Image.new("RGB", (480, 600), (110, 140, 170))
    pixels = panel.load()
    for y in range(panel.height):
        for x in range(panel.width):
            tone = (x * 3 + y * 2) % 35
            pixels[x, y] = (90 + tone, 125 + tone, 155 + tone)
    mask = Image.new("L", panel.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, 479, 599),
        radius=40,
        fill=255,
    )
    image.paste(panel.filter(ImageFilter.GaussianBlur(0.5)), (60, 220), mask)

    boxes = (
        (60, 60, 540, 940),
        (60, 220, 540, 820),
        (150, 60, 450, 940),
        (60, 300, 540, 820),
        (60, 220, 450, 820),
    )
    candidates = [
        RectangleCandidate(
            pixels_to_box_1000(box, *image.size),
            score,
            0.5,
            0.3,
            0.8,
            "test",
        )
        for box, score in zip(
            boxes,
            (0.70, 0.68, 0.67, 0.66, 0.65),
            strict=True,
        )
    ]

    class ModelMustNotBeCalled:
        def select_candidate(self, image, candidates):
            raise AssertionError("Clear sharpness contrast should not need Qwen.")

    selection, method, metrics = _select_candidate(
        image,
        candidates,
        ModelMustNotBeCalled(),
    )
    selected_box, matching_id, boundary_ids = _resolve_selected_box(
        selection,
        candidates,
    )

    assert method == "sharpness_boundary_confidence"
    assert selected_box == candidates[1].box
    assert matching_id == 2
    assert boundary_ids == {"left": 1, "top": 2, "right": 1, "bottom": 2}
    assert metrics["boundary_evidence"]["left"]["coordinate"] == 60
    assert metrics["boundary_evidence"]["top"]["coordinate"] == 220
    assert metrics["boundary_evidence"]["right"]["coordinate"] == 540
    assert metrics["boundary_evidence"]["bottom"]["coordinate"] == 820



def test_saved_crop_uses_transparent_rounded_corners(tmp_path: Path):
    from inlay_qwen3vl.geometry import pixels_to_box_1000

    image = Image.new("RGB", (100, 100), (100, 140, 180))
    box = pixels_to_box_1000((10, 10, 90, 90), *image.size)

    crop_path, _ = _save_selected_crop(
        image,
        box,
        tmp_path,
        corner_radius=18,
    )

    with Image.open(crop_path) as crop:
        assert crop.mode == "RGBA"
        assert crop.getpixel((0, 0))[3] == 0
        assert crop.getpixel((40, 40))[3] == 255
        assert crop.getpixel((0, 40))[3] == 255


def test_saved_square_crop_keeps_all_corner_pixels_opaque(tmp_path: Path):
    from inlay_qwen3vl.geometry import pixels_to_box_1000

    image = Image.new("RGB", (100, 100), (100, 140, 180))
    box = pixels_to_box_1000((10, 10, 90, 90), *image.size)

    crop_path, _ = _save_selected_crop(image, box, tmp_path)

    with Image.open(crop_path) as crop:
        assert crop.mode == "RGBA"
        assert crop.getpixel((0, 0))[3] == 255
        assert crop.getpixel((40, 40))[3] == 255
