from copy import deepcopy
from unittest.mock import patch

from PIL import Image

from inlay_qwen3vl.client import LMStudioVisionClient


def successful_response():
    response = patch_response()
    response.json.return_value = {
        "output": [
            {
                "type": "message",
                "content": (
                    '{"done": false, "bbox_1000": [100, 120, 900, 880], '
                    '"reason": "Located the full image panel."}'
                ),
            }
        ],
        "stats": {
            "total_output_tokens": 42,
            "reasoning_output_tokens": 0,
        },
    }
    return response


def patch_response():
    from unittest.mock import Mock

    response = Mock()
    response.ok = True
    response.status_code = 200
    response.text = ""
    return response


@patch("inlay_qwen3vl.client.requests.post")
def test_initial_request_uses_native_chat_reasoning_off_and_one_image(post):
    post.return_value = successful_response()
    client = LMStudioVisionClient("http://localhost:1234/api/v1/chat", "qwen-test")
    reply = client.inspect(Image.new("RGB", (640, 480)), None, 1)

    payload = post.call_args.kwargs["json"]
    request_input = payload["input"]
    assert reply.box == (100, 120, 900, 880)
    assert payload["reasoning"] == "off"
    assert payload["max_output_tokens"] == 1024
    assert payload["store"] is False
    assert payload["system_prompt"]
    assert len(request_input) == 2
    assert request_input[0]["type"] == "text"
    assert request_input[1]["type"] == "image"
    assert request_input[1]["data_url"].startswith("data:image/png;base64,")


@patch("inlay_qwen3vl.client.requests.post")
def test_review_request_sends_original_overlay_and_candidate(post):
    post.return_value = successful_response()
    client = LMStudioVisionClient("http://localhost:1234/api/v1/chat", "qwen-test")
    client.inspect(Image.new("RGB", (640, 480)), (100, 100, 900, 900), 2)

    request_input = post.call_args.kwargs["json"]["input"]
    assert sum(item["type"] == "image" for item in request_input) == 3


@patch("inlay_qwen3vl.client.requests.post")
def test_reads_a_json_answer_even_when_reasoning_is_also_returned(post):
    response = successful_response()
    response.json.return_value["output"].insert(
        0,
        {"type": "reasoning", "content": "Brief image analysis."},
    )
    post.return_value = response

    client = LMStudioVisionClient("http://localhost:1234/api/v1/chat", "qwen-test")
    reply = client.inspect(Image.new("RGB", (100, 100)), None, 1)

    assert reply.box == (100, 120, 900, 880)
    assert reply.done is False


@patch("inlay_qwen3vl.client.requests.post")
def test_reports_when_native_api_returns_only_incomplete_reasoning(post):
    response = successful_response()
    response.json.return_value = {
        "output": [
            {"type": "reasoning", "content": "The screenshot shows a centered image panel..."}
        ],
        "stats": {
            "total_output_tokens": 1024,
            "reasoning_output_tokens": 1024,
        },
    }
    post.return_value = response

    client = LMStudioVisionClient("http://localhost:1234/api/v1/chat", "qwen-test")
    try:
        client.inspect(Image.new("RGB", (100, 100)), None, 1)
    except RuntimeError as error:
        message = str(error)
        assert "did not return a valid crop rectangle" in message
        assert "reasoning tokens: 1024" in message
    else:
        raise AssertionError("Expected incomplete reasoning to be rejected.")


@patch("inlay_qwen3vl.client.requests.post")
def test_retries_transient_server_errors(post):
    response = successful_response()
    failed = patch_response()
    failed.ok = False
    failed.status_code = 503
    failed.reason = "Service Unavailable"
    failed.text = "temporarily unavailable"
    post.side_effect = [failed, response]

    client = LMStudioVisionClient(
        "http://localhost:1234/api/v1/chat",
        "qwen-test",
        report=lambda _: None,
    )
    with patch("inlay_qwen3vl.client.time.sleep"):
        reply = client.inspect(Image.new("RGB", (100, 100)), None, 1)

    assert reply.box == (100, 120, 900, 880)
    assert post.call_count == 2
