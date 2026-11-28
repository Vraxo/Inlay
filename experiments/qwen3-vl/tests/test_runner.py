from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from inlay_qwen3vl.protocol import CandidateSelection
from inlay_qwen3vl.runner import _select_candidate, run_experiment
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
    assert result["selection_method"] in {"opencv_high_confidence", "qwen_semantic"}
    assert result["refinement"]["method"] == "local Sobel edge snapping"
    assert (output / "candidate.png").is_file()
    assert (output / "overlay.png").is_file()
    assert (output / "proposals.png").is_file()
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
