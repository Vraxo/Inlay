from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from typing import Callable
from urllib.parse import urlsplit

import requests


def _api_base(api_url: str) -> str:
    parsed = urlsplit(api_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _model_ids(api_url: str) -> list[str]:
    response = requests.get(f"{_api_base(api_url)}/v1/models", timeout=5)
    response.raise_for_status()
    return [
        str(model.get("id"))
        for model in response.json().get("data", [])
        if model.get("id")
    ]


def _model_is_available(api_url: str, model_name: str) -> bool:
    try:
        return any(
            model_id.casefold() == model_name.casefold()
            for model_id in _model_ids(api_url)
        )
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return False


def _local_model_keys(lms_executable: str) -> list[str]:
    """Read downloaded LLM model keys from the LM Studio CLI."""
    result = subprocess.run(
        [lms_executable, "ls", "--llm", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            detail or f"'lms ls --llm --json' exited with code {result.returncode}."
        )

    models = json.loads(result.stdout)
    if not isinstance(models, list):
        raise ValueError("'lms ls --llm --json' did not return a JSON array.")

    return [
        str(model["modelKey"])
        for model in models
        if isinstance(model, dict) and model.get("modelKey")
    ]


def _normalise_model_key(value: str) -> str:
    model_name = value.split("@", 1)[0].rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9]", "", model_name.casefold())


def _match_local_model_key(model_name: str, model_keys: list[str]) -> str | None:
    """Match one unambiguous local model, never substituting a Thinking checkpoint."""
    target = _normalise_model_key(model_name)
    if not target:
        return None

    exact = [
        key for key in model_keys
        if _normalise_model_key(key) == target
    ]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None

    compatible = []
    for key in model_keys:
        candidate = _normalise_model_key(key)
        if not candidate.startswith(target):
            continue
        suffix = candidate[len(target):]
        if "thinking" in suffix or "think" in suffix:
            continue
        compatible.append(key)

    return compatible[0] if len(compatible) == 1 else None


def _spawn(lms_executable: str, arguments: list[str]) -> subprocess.Popen | None:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        return subprocess.Popen(
            [lms_executable, *arguments],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
    except FileNotFoundError:
        return None


def ensure_server_ready(
    api_url: str,
    model_name: str,
    *,
    lms_executable: str = "lms",
    load_model: str = "",
    report: Callable[[str], None] = print,
) -> bool:
    """Start LM Studio, auto-load a matching downloaded model, and verify its API ID."""
    if not _model_is_available(api_url, model_name):
        report("LM Studio server/model not ready; checking the local server...")
        try:
            server_reachable = bool(
                requests.get(f"{_api_base(api_url)}/v1/models", timeout=5).ok
            )
        except requests.RequestException:
            server_reachable = False

        if not server_reachable:
            if _spawn(lms_executable, ["server", "start"]) is None:
                report(f"Could not start LM Studio: '{lms_executable}' was not found.")
                report("Start the server manually or configure INLAY_LMS_EXECUTABLE.")
                return False
            report("Waiting for the LM Studio server...")
            for _ in range(30):
                time.sleep(2)
                try:
                    if requests.get(
                        f"{_api_base(api_url)}/v1/models", timeout=5
                    ).ok:
                        server_reachable = True
                        break
                except requests.RequestException:
                    pass
            if not server_reachable:
                report("Timed out waiting for the LM Studio server.")
                return False

    if _model_is_available(api_url, model_name):
        report(f"LM Studio model '{model_name}' is available.")
        return True

    if load_model.strip():
        model_key = load_model.strip()
    else:
        report(f"Model '{model_name}' is not loaded; looking for its local download...")
        try:
            model_keys = _local_model_keys(lms_executable)
        except (OSError, subprocess.SubprocessError, ValueError, TypeError) as error:
            report(f"Could not list downloaded LM Studio models: {error}")
            report("Load the model in LM Studio or set --load-model to its exact local model key.")
            return False

        model_key = _match_local_model_key(model_name, model_keys)
        if model_key is None:
            if _normalise_model_key(model_name) == "qwen3vl8b":
                thinking_keys = [
                    key for key in model_keys
                    if _normalise_model_key(key).startswith("qwen3vl8b")
                    and "thinking" in _normalise_model_key(key)
                ]
                if thinking_keys:
                    report(
                        "The downloaded Qwen3-VL Thinking checkpoint is a separate model "
                        "variant. Inlay needs the Qwen3-VL-8B Instruct checkpoint for "
                        "short, machine-readable crop coordinates."
                    )
                    report(
                        "Download 'Qwen3-VL-8B' (Instruct) in LM Studio, then rerun this command."
                    )
            report(f"Could not find one unambiguous local model matching '{model_name}'.")
            if model_keys:
                report("Downloaded LLM model keys:")
                for key in model_keys:
                    report(f"  - {key}")
            else:
                report("No downloaded LLM models were found by 'lms ls --llm'.")
            report("Set --load-model to the exact key shown above, or download the vision model first.")
            return False

    report(f"Loading local model '{model_key}' as API model '{model_name}'...")
    process = _spawn(
        lms_executable,
        ["load", model_key, "--identifier", model_name, "--yes"],
    )
    if process is None:
        report(f"Could not run '{lms_executable} load'.")
        return False

    for _ in range(90):
        time.sleep(2)
        if _model_is_available(api_url, model_name):
            report(f"LM Studio model '{model_name}' is ready.")
            return True
        if process.poll() is not None and process.returncode not in (0, None):
            report(f"'lms load' exited with code {process.returncode}.")
            return False

    report("Timed out waiting for the model to load and appear in /v1/models.")
    return False
