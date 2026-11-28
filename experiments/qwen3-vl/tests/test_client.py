from unittest.mock import patch

from PIL import Image
import requests

from inlay_qwen3vl.client import LMStudioVisionClient, MAX_OUTPUT_TOKENS
from inlay_qwen3vl.vision import RectangleCandidate


def patch_response(content):
    from unittest.mock import Mock

    response = Mock()
    response.ok = True
    response.status_code = 200
    response.text = ""
    response.json.return_value = {
        "output": [{"type": "message", "content": content}],
        "stats": {"total_output_tokens": 40, "reasoning_output_tokens": 0},
    }
    return response


def sample_candidates():
    return [
        RectangleCandidate((40, 100, 960, 900), 0.8, 0.7, 0.6, 0.8, "test"),
        RectangleCandidate((60, 120, 940, 880), 0.7, 0.6, 0.5, 0.7, "test"),
    ]


@patch("inlay_qwen3vl.client.requests.post")
def test_sends_original_and_numbered_opencv_candidates(post):
    post.return_value = patch_response(
        '{"left_candidate_id": 1, "top_candidate_id": 2, '
        '"right_candidate_id": 1, "bottom_candidate_id": 2, '
        '"reason": "These edges enclose the sharp foreground panel."}'
    )
    image = Image.new("RGB", (640, 960), "white")
    candidates = sample_candidates()

    client = LMStudioVisionClient("http://localhost:1234/api/v1/chat", "qwen-test")
    selection = client.select_candidate(image, candidates)

    payload = post.call_args.kwargs["json"]
    request_input = payload["input"]
    assert selection.candidate_id is None
    assert selection.uses_independent_boundaries
    assert selection.left_candidate_id == 1
    assert selection.top_candidate_id == 2
    assert selection.right_candidate_id == 1
    assert selection.bottom_candidate_id == 2
    assert payload["max_output_tokens"] == MAX_OUTPUT_TOKENS == 384
    assert payload["store"] is False
    assert "sharpness map" in payload["system_prompt"]
    assert "independently" in payload["system_prompt"]
    assert len(request_input) == 4
    assert request_input[0]["type"] == "text"
    assert all(item["type"] == "image" for item in request_input[1:])
    assert payload["model"] == "qwen-test"


@patch("inlay_qwen3vl.client.requests.post")
def test_rejects_unknown_candidate_id(post):
    post.return_value = patch_response(
        '{"left_candidate_id": 9, "top_candidate_id": 2, '
        '"right_candidate_id": 1, "bottom_candidate_id": 2, '
        '"reason": "Not on the overlay."}'
    )
    client = LMStudioVisionClient("http://localhost:1234/api/v1/chat", "qwen-test")

    try:
        client.select_candidate(Image.new("RGB", (100, 100)), sample_candidates())
    except RuntimeError as error:
        assert "did not select a valid numbered candidate" in str(error)
    else:
        raise AssertionError("Expected an invalid candidate selection to fail.")


@patch("inlay_qwen3vl.client.requests.post")
def test_does_not_retry_for_unrelated_bad_requests(post):
    from unittest.mock import Mock
    response = Mock()
    response.ok = False
    response.status_code = 400
    response.reason = "Bad Request"
    response.text = "Invalid image input."
    post.return_value = response

    client = LMStudioVisionClient("http://localhost:1234/api/v1/chat", "qwen-test")
    try:
        client.select_candidate(Image.new("RGB", (100, 100)), sample_candidates())
    except requests.HTTPError:
        pass
    else:
        raise AssertionError("Expected the unrelated bad request to propagate.")

    post.assert_called_once()
