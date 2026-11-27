from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from inlay_qwen3vl.protocol import CandidateSelection
from inlay_qwen3vl.runner import run_experiment


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
    assert result["refinement"]["method"] == "local Sobel edge snapping"
    assert (output / "candidate.png").is_file()
    assert (output / "overlay.png").is_file()
    assert (output / "proposals.png").is_file()
    assert (output / "report.json").is_file()
    assert not (output / "crop.png").exists()
