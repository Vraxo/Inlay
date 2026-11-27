from __future__ import annotations

import base64
import io
import os
import time
from typing import Any, Callable

import requests
from PIL import Image, ImageDraw

from .geometry import Box1000, box_1000_to_pixels
from .protocol import ModelReply, parse_reply


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "inlay_crop_refinement",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "done": {"type": "boolean"},
                "bbox_1000": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "reason": {"type": "string"},
            },
            "required": ["done", "bbox_1000", "reason"],
            "additionalProperties": False,
        },
    },
}


SYSTEM_PROMPT = """You are a precise image-boundary locator for Inlay.

Your sole target is the COMPLETE foreground photograph or rectangular image panel
shown in the screenshot. Locate the outer boundary of that entire panel, including
all of its picture content. Do NOT locate a person, animal, object, face, or other
subject inside the photograph. Do NOT return the blurred background, an application
window, or surrounding interface elements.

Coordinates must be normalized integers from 0 to 1000 and ordered as:
[left, top, right, bottom]. They refer to the full original screenshot, not a crop
or an overlay. The right and bottom values mark the outer edge of the rectangle.

Return only the requested JSON object. Be conservative: a rectangle that cuts off
any part of the foreground image is not acceptable. On a review pass, set done=true
only when the current rectangle is correct and you are returning exactly the same
coordinates. Otherwise return your best corrected rectangle and set done=false."""


class LMStudioVisionClient:
    def __init__(
        self,
        api_url: str,
        model_name: str,
        *,
        timeout: int = 300,
        max_image_dimension: int = 2048,
        report: Callable[[str], None] = print,
    ) -> None:
        if max_image_dimension < 256:
            raise ValueError("max_image_dimension must be at least 256.")
        self.api_url = api_url
        self.model_name = model_name
        self.timeout = timeout
        self.max_image_dimension = max_image_dimension
        self.report = report

    def inspect(
        self,
        image: Image.Image,
        current_box: Box1000 | None,
        iteration: int,
    ) -> ModelReply:
        preview = self._resize(image)
        width, height = image.size

        if current_box is None:
            instruction = (
                f"Initial localization pass. The original screenshot is {width} x {height} pixels. "
                "Find the complete foreground photo/image panel and return its outer rectangle. "
                "This first pass must set done=false; the next pass will visually verify the crop."
            )
            content = [{"type": "text", "text": instruction}]
            content.append(self._image_part(preview))
        else:
            pixel_box = box_1000_to_pixels(current_box, width, height)
            crop = image.crop(pixel_box)
            overlay = self._draw_box(preview, current_box)
            instruction = (
                f"Review pass {iteration}. The original screenshot is {width} x {height} pixels. "
                "Image 1 is the original screenshot. Image 2 shows the current candidate rectangle "
                "outlined in magenta. Image 3 is the exact crop produced from that rectangle. "
                "Check whether image 3 contains the COMPLETE foreground photo/image panel and "
                "excludes surrounding background/interface. Compare it with image 1. "
                "Return coordinates relative to image 1. If the current rectangle is correct, "
                "return the same coordinates and set done=true. If it is too tight, too loose, "
                "or misplaced, return corrected coordinates and set done=false. "
                f"Current bbox_1000: {list(current_box)}."
            )
            content = [{"type": "text", "text": instruction}]
            content.append(self._image_part(preview))
            content.append(self._image_part(overlay))
            content.append(self._image_part(self._resize(crop)))

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
            "stream": False,
            "response_format": RESPONSE_FORMAT,
        }

        response = self._post_with_retries(payload)
        if response.status_code in (400, 422) and self._is_schema_error(response.text):
            self.report("LM Studio rejected JSON-schema mode; retrying with ordinary JSON output.")
            payload.pop("response_format", None)
            response = self._post_with_retries(payload)

        try:
            message = response.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                f"LM Studio returned an unexpected chat response: {response.text}"
            ) from error

        content_text = message.get("content") or ""
        reasoning_text = message.get("reasoning_content") or ""
        try:
            return parse_reply(content_text, reasoning_text)
        except ValueError as error:
            raise RuntimeError(
                "Qwen returned a response that could not be parsed as a crop rectangle. "
                f"Content: {content_text!r}; reasoning: {reasoning_text!r}"
            ) from error

    def _post_with_retries(self, payload: dict[str, Any]) -> requests.Response:
        headers = {"Content-Type": "application/json"}
        api_token = os.environ.get("LM_API_TOKEN", "").strip()
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        for attempt in range(3):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.ok:
                    return response
                error = requests.HTTPError(
                    f"HTTP {response.status_code} {response.reason}: {response.text}",
                    response=response,
                )
                if response.status_code < 500 or attempt == 2:
                    raise error
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 2:
                    raise
            if attempt < 2:
                delay = 2 ** (attempt + 1)
                self.report(f"LM Studio request failed; retrying in {delay}s...")
                time.sleep(delay)

        raise RuntimeError("LM Studio request failed after transient retries.")

    @staticmethod
    def _is_schema_error(message: str) -> bool:
        lowered = message.casefold()
        return any(
            term in lowered
            for term in ("response_format", "json_schema", "structured output", "grammar")
        )

    def _resize(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        longest = max(width, height)
        if longest <= self.max_image_dimension:
            return image.convert("RGB")
        scale = self.max_image_dimension / longest
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return image.resize(size, Image.Resampling.LANCZOS).convert("RGB")

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _image_part(self, image: Image.Image) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": self._encode_image(image)},
        }

    @staticmethod
    def _draw_box(image: Image.Image, box: Box1000) -> Image.Image:
        preview = image.copy()
        rectangle = box_1000_to_pixels(box, preview.width, preview.height)
        left, top, right, bottom = rectangle
        draw = ImageDraw.Draw(preview)
        line_width = max(3, round(max(preview.size) / 500))
        for offset in range(line_width):
            draw.rectangle(
                (left + offset, top + offset, right - 1 - offset, bottom - 1 - offset),
                outline=(255, 0, 255),
            )
        return preview
