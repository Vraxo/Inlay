import pytest

from inlay_qwen3vl.protocol import parse_reply


def test_parses_plain_json_response():
    reply = parse_reply(
        '{"done": false, "bbox_1000": [100, 120, 900, 880], "reason": "Panel found"}'
    )
    assert reply.done is False
    assert reply.box == (100, 120, 900, 880)
    assert reply.reason == "Panel found"


def test_parses_json_with_surrounding_commentary():
    reply = parse_reply(
        'Result follows:\n{"done": true, "bbox_1000": [10, 20, 900, 950], "reason": "Verified"}\nFinished.'
    )
    assert reply.done is True
    assert reply.box == (10, 20, 900, 950)


def test_uses_reasoning_when_content_is_empty():
    reply = parse_reply(
        "",
        '<think>Coordinates checked.</think>\n{"done": false, "bbox_1000": [10, 20, 900, 950], "reason": "Adjusted"}',
    )
    assert reply.box == (10, 20, 900, 950)


def test_rejects_non_boolean_done():
    with pytest.raises(ValueError, match="'done' must be a boolean"):
        parse_reply('{"done": "yes", "bbox_1000": [10, 20, 900, 950]}')


def test_rejects_missing_rectangle():
    with pytest.raises(ValueError, match="bbox_1000"):
        parse_reply('{"done": false, "reason": "No rectangle"}')


def test_rejects_responses_without_json():
    with pytest.raises(ValueError, match="valid JSON object"):
        parse_reply("I think the picture is centered.")
