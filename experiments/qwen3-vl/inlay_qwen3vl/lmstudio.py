from __future__ import annotations

import subprocess
import sys
import time
from typing import Callable

import requests


def _api_base(api_url: str) -> str:
    return api_url.split("/v1/", 1)[0].rstrip("/")


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
    """Start LM Studio's server if needed and verify the configured model ID."""
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

    if not load_model.strip():
        report(f"Model '{model_name}' is not currently listed by LM Studio.")
        report("Load it in LM Studio, or rerun with --load-model <local-model-key>.")
        try:
            available = _model_ids(api_url)
        except (requests.RequestException, ValueError, KeyError, TypeError):
            available = []
        if available:
            report("Models currently listed by the server:")
            for model_id in available:
                report(f"  - {model_id}")
        return False

    report(f"Loading '{load_model}' through the LM Studio CLI...")
    process = _spawn(lms_executable, ["load", load_model, "--yes"])
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

    report("Timed out waiting for the model to load.")
    return False
