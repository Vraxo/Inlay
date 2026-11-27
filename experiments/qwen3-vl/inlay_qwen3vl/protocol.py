from __future__ import annotations

import json
from dataclasses import dataclass

from .geometry import Box1000, validate_box_1000


@dataclass(frozen=True)
class ModelReply:
    done: bool
    box: Box1000
    reason: str
    raw_text: str


def _find_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("The model response did not contain a valid JSON object.")


def parse_reply(content: str, reasoning_content: str = "") -> ModelReply:
    errors: list[str] = []
    for candidate in (content, reasoning_content):
        if not candidate.strip():
            continue
        try:
            data = _find_json_object(candidate)
            done = data.get("done")
            if not isinstance(done, bool):
                raise ValueError("'done' must be a boolean.")
            box = validate_box_1000(data.get("bbox_1000"))
            reason = data.get("reason", "")
            if not isinstance(reason, str):
                raise ValueError("'reason' must be a string.")
            return ModelReply(done, box, reason.strip(), candidate)
        except (ValueError, TypeError) as error:
            errors.append(str(error))
    detail = "; ".join(errors) if errors else "The response was empty."
    raise ValueError(f"Could not parse a valid crop response: {detail}")
