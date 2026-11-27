from __future__ import annotations

import json
from dataclasses import dataclass

from .geometry import (
    Box1000,
    pixels_to_box_1000,
    validate_box_1000,
    validate_pixel_box,
)


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


def _parse_box(data: dict, image_size: tuple[int, int] | None) -> Box1000:
    if "bbox_pixels_xyxy" in data:
        if image_size is None:
            raise ValueError("Image dimensions are required for pixel coordinates.")
        width, height = image_size
        pixel_box = validate_pixel_box(data["bbox_pixels_xyxy"], width, height)
        return pixels_to_box_1000(pixel_box, width, height)

    if "bbox_1000" not in data:
        raise ValueError("The response did not contain bbox_pixels_xyxy or bbox_1000.")

    try:
        return validate_box_1000(data["bbox_1000"])
    except ValueError as normalized_error:
        # Older prompt versions called the field bbox_1000 even when smaller
        # vision models answered with full-resolution pixel coordinates.
        if image_size is None:
            raise normalized_error
        width, height = image_size
        try:
            pixel_box = validate_pixel_box(data["bbox_1000"], width, height)
            return pixels_to_box_1000(pixel_box, width, height)
        except ValueError:
            raise normalized_error


def parse_reply(
    content: str,
    reasoning_content: str = "",
    *,
    image_size: tuple[int, int] | None = None,
) -> ModelReply:
    errors: list[str] = []
    for candidate in (content, reasoning_content):
        if not candidate.strip():
            continue
        try:
            data = _find_json_object(candidate)
            done = data.get("done")
            if not isinstance(done, bool):
                raise ValueError("'done' must be a boolean.")
            box = _parse_box(data, image_size)
            reason = data.get("reason", "")
            if not isinstance(reason, str):
                raise ValueError("'reason' must be a string.")
            return ModelReply(done, box, reason.strip(), candidate)
        except (ValueError, TypeError) as error:
            errors.append(str(error))
    detail = "; ".join(errors) if errors else "The response was empty."
    raise ValueError(f"Could not parse a valid crop response: {detail}")
