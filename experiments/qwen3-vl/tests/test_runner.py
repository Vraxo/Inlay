from pathlib import Path

from PIL import Image

from inlay_qwen3vl.protocol import ModelReply
from inlay_qwen3vl.runner import run_experiment


class FakeVisionClient:
    model_name = "fake-qwen"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def inspect(self, image, current_box, iteration):
        self.calls.append((current_box, iteration))
        return next(self.replies)


def test_refines_then_accepts_only_an_unchanged_reviewed_box(tmp_path: Path):
    source = tmp_path / "source.png"
    output = tmp_path / "output"
    Image.new("RGB", (1000, 500), "white").save(source)

    initial = ModelReply(False, (100, 100, 900, 900), "Initial", "{}")
    adjusted = ModelReply(False, (120, 120, 880, 880), "Tightened", "{}")
    verified = ModelReply(True, (120, 120, 880, 880), "Complete panel", "{}")
    client = FakeVisionClient([initial, adjusted, verified])

    result = run_experiment(
        source, output, client, max_iterations=5, report=lambda _: None
    )

    assert result["status"] == "accepted"
    assert result["accepted"] is True
    assert result["final_bbox_1000"] == [120, 120, 880, 880]
    assert client.calls == [
        (None, 1),
        ((100, 100, 900, 900), 2),
        ((120, 120, 880, 880), 3),
    ]
    assert (output / "crop.png").is_file()
    assert (output / "overlay.png").is_file()
    assert (output / "report.json").is_file()
    assert not (output / "candidate.png").exists()


def test_never_accepts_a_rectangle_on_the_initial_pass(tmp_path: Path):
    source = tmp_path / "source.png"
    output = tmp_path / "output"
    Image.new("RGB", (100, 100), "white").save(source)

    client = FakeVisionClient([
        ModelReply(True, (100, 100, 900, 900), "Initial guess", "{}"),
        ModelReply(False, (100, 100, 900, 900), "Needs more checking", "{}"),
        ModelReply(False, (100, 100, 900, 900), "Still uncertain", "{}"),
    ])

    result = run_experiment(
        source, output, client, max_iterations=5, report=lambda _: None
    )

    assert result["status"] == "stalled"
    assert result["accepted"] is False
    assert (output / "candidate.png").is_file()
    assert not (output / "crop.png").exists()


def test_stops_after_iteration_limit_with_candidate_output(tmp_path: Path):
    source = tmp_path / "source.png"
    output = tmp_path / "output"
    Image.new("RGB", (200, 100), "white").save(source)

    client = FakeVisionClient([
        ModelReply(False, (100, 100, 900, 900), "Initial", "{}"),
        ModelReply(False, (110, 110, 890, 890), "Adjusted", "{}"),
    ])

    result = run_experiment(
        source, output, client, max_iterations=2, report=lambda _: None
    )

    assert result["status"] == "max_iterations"
    assert result["accepted"] is False
    assert (output / "candidate.png").is_file()
