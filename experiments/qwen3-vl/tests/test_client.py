from unittest.mock import Mock, patch

from PIL import Image

from inlay_qwen3vl.client import LMStudioVisionClient


def successful_response():
    response = Mock()
    response.ok = True
    response.status_code = 200
    response.text = ""
    response.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"done": false, "bbox_1000": [100, 120, 900, 880], "reason": "Located"}',
                "reasoning_content": "",
            }
        }]
    }
    return response


@patch("inlay_qwen3vl.client.requests.post")
def test_initial_request_sends_one_image_and_structured_output(post):
    post.return_value = successful_response()
    client = LMStudioVisionClient("http://localhost:1234/v1/chat/completions", "qwen-test")
    reply = client.inspect(Image.new("RGB", (640, 480)), None, 1)

    payload = post.call_args.kwargs["json"]
    content = payload["messages"][1]["content"]
    assert reply.box == (100, 120, 900, 880)
    assert payload["response_format"]["type"] == "json_schema"
    assert sum(part["type"] == "image_url" for part in content) == 1
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@patch("inlay_qwen3vl.client.requests.post")
def test_review_request_sends_original_overlay_and_candidate(post):
    post.return_value = successful_response()
    client = LMStudioVisionClient("http://localhost:1234/v1/chat/completions", "qwen-test")
    client.inspect(Image.new("RGB", (640, 480)), (100, 100, 900, 900), 2)

    content = post.call_args.kwargs["json"]["messages"][1]["content"]
    assert sum(part["type"] == "image_url" for part in content) == 3


@patch("inlay_qwen3vl.client.requests.post")
def test_retries_without_schema_when_lm_studio_rejects_it(post):
    rejected = Mock()
    rejected.ok = False
    rejected.status_code = 400
    rejected.reason = "Bad Request"
    rejected.text = "response_format json_schema is not supported"

    post.side_effect = [rejected, successful_response()]
    client = LMStudioVisionClient(
        "http://localhost:1234/v1/chat/completions",
        "qwen-test",
        report=lambda _: None,
    )
    reply = client.inspect(Image.new("RGB", (100, 100)), None, 1)

    assert reply.box == (100, 120, 900, 880)
    assert post.call_count == 2
    assert "response_format" in post.call_args_list[0].kwargs["json"]
    assert "response_format" not in post.call_args_list[1].kwargs["json"]
