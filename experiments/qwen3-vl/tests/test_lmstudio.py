import json
import subprocess
from unittest.mock import Mock, patch

from inlay_qwen3vl.lmstudio import (
    _api_base,
    _local_model_keys,
    _match_local_model_key,
    ensure_server_ready,
)


def test_api_base_supports_native_and_compatible_chat_endpoints():
    assert _api_base("http://localhost:1234/api/v1/chat") == "http://localhost:1234"
    assert _api_base("http://localhost:1234/v1/chat/completions") == "http://localhost:1234"


def test_lists_downloaded_llm_model_keys_from_lms_json():
    output = json.dumps([
        {"type": "llm", "modelKey": "qwen/qwen3-vl-8b-thinking"},
        {"type": "llm", "modelKey": "meta/llama-3.1-8b"},
    ])
    completed = subprocess.CompletedProcess(
        args=["lms", "ls", "--llm", "--json"],
        returncode=0,
        stdout=output,
        stderr="",
    )
    with patch("inlay_qwen3vl.lmstudio.subprocess.run", return_value=completed) as run:
        assert _local_model_keys("lms") == [
            "qwen/qwen3-vl-8b-thinking",
            "meta/llama-3.1-8b",
        ]
    run.assert_called_once_with(
        ["lms", "ls", "--llm", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_matches_model_id_across_publisher_and_separator_differences():
    keys = [
        "qwen/qwen3-vl-8b-thinking",
        "meta/llama-3.1-8b",
    ]
    assert _match_local_model_key("qwen3vl-8b-thinking", keys) == (
        "qwen/qwen3-vl-8b-thinking"
    )


def test_matches_a_model_key_with_a_quantization_suffix():
    keys = ["qwen/qwen3-vl-8b-thinking-GGUF"]
    assert _match_local_model_key("qwen3vl-8b-thinking", keys) == keys[0]


def test_generic_qwen3vl_model_prefers_instruct_checkpoint():
    keys = [
        "lmstudio-community/Qwen3-VL-8B-Thinking-GGUF",
        "lmstudio-community/Qwen3-VL-8B-Instruct-GGUF",
    ]
    assert _match_local_model_key("qwen3vl-8b", keys) == keys[1]


def test_generic_qwen3vl_model_never_substitutes_thinking_checkpoint():
    keys = ["lmstudio-community/Qwen3-VL-8B-Thinking-GGUF"]
    assert _match_local_model_key("qwen3vl-8b", keys) is None


def test_does_not_guess_when_multiple_exact_local_models_match():
    keys = [
        "qwen/qwen3-vl-8b-thinking",
        "community/qwen3-vl-8b-thinking",
    ]
    assert _match_local_model_key("qwen3vl-8b-thinking", keys) is None


@patch("inlay_qwen3vl.lmstudio.time.sleep")
@patch("inlay_qwen3vl.lmstudio._spawn")
@patch("inlay_qwen3vl.lmstudio._local_model_keys")
@patch("inlay_qwen3vl.lmstudio.requests.get")
@patch("inlay_qwen3vl.lmstudio._model_is_available", side_effect=[False, False, True])
def test_automatically_loads_matching_downloaded_model(
    is_available,
    get,
    local_model_keys,
    spawn,
    sleep,
):
    get.return_value.ok = True
    local_model_keys.return_value = ["qwen/qwen3-vl-8b-thinking"]
    spawn.return_value = Mock()

    assert ensure_server_ready(
        "http://localhost:1234/api/v1/chat",
        "qwen3vl-8b-thinking",
        report=lambda _: None,
    )

    spawn.assert_called_once_with(
        "lms",
        [
            "load",
            "qwen/qwen3-vl-8b-thinking",
            "--identifier",
            "qwen3vl-8b-thinking",
            "--yes",
        ],
    )
    assert is_available.call_count == 3


@patch("inlay_qwen3vl.lmstudio._spawn")
@patch(
    "inlay_qwen3vl.lmstudio._local_model_keys",
    return_value=["lmstudio-community/Qwen3-VL-8B-Thinking-GGUF"],
)
@patch("inlay_qwen3vl.lmstudio.requests.get")
@patch("inlay_qwen3vl.lmstudio._model_is_available", return_value=False)
def test_explains_when_only_the_thinking_variant_is_downloaded(
    is_available, get, local_model_keys, spawn
):
    get.return_value.ok = True
    messages = []

    assert not ensure_server_ready(
        "http://localhost:1234/api/v1/chat",
        "qwen3vl-8b",
        report=messages.append,
    )

    spawn.assert_not_called()
    assert any("separate model variant" in message for message in messages)
    assert any("Qwen3-VL-8B" in message and "Instruct" in message for message in messages)


@patch("inlay_qwen3vl.lmstudio._spawn")
@patch("inlay_qwen3vl.lmstudio._local_model_keys", return_value=[])
@patch("inlay_qwen3vl.lmstudio.requests.get")
@patch("inlay_qwen3vl.lmstudio._model_is_available", return_value=False)
def test_reports_when_no_local_llm_matches(is_available, get, local_model_keys, spawn):
    get.return_value.ok = True
    messages = []

    assert not ensure_server_ready(
        "http://localhost:1234/api/v1/chat",
        "qwen3vl-8b",
        report=messages.append,
    )

    spawn.assert_not_called()
    assert any("No downloaded LLM models" in message for message in messages)


@patch("inlay_qwen3vl.lmstudio.time.sleep")
@patch("inlay_qwen3vl.lmstudio._spawn")
@patch("inlay_qwen3vl.lmstudio.requests.get")
@patch("inlay_qwen3vl.lmstudio._model_is_available", side_effect=[False, False, True])
def test_explicit_local_model_key_overrides_auto_discovery(
    is_available,
    get,
    spawn,
    sleep,
):
    get.return_value.ok = True
    spawn.return_value = Mock()

    assert ensure_server_ready(
        "http://localhost:1234/api/v1/chat",
        "qwen3vl-8b-thinking",
        load_model="custom/qwen-vl-model",
        report=lambda _: None,
    )

    spawn.assert_called_once_with(
        "lms",
        [
            "load",
            "custom/qwen-vl-model",
            "--identifier",
            "qwen3vl-8b-thinking",
            "--yes",
        ],
    )
