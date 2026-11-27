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


MAX_OUTPUT_TOKENS = 1024

SYSTEM_PROMPT = """You are Inlay's image-boundary locator.

Find the OUTER RECTANGLE of the complete foreground photograph or rectangular
image panel in the supplied screenshot. Locate the whole panel, never a person,
face, object, or other subject inside the photograph. Exclude blurred background
and surrounding phone/application interface.

Return exactly one JSON object and nothing else:
{"done": false, "bbox_1000": [left, top, right, bottom], "reason": "brief reason"}

Coordinates are integers from 0 to 1000, relative to the FULL original screenshot.
The order is left, top, right, bottom. The right and bottom values mark the outer
edges. Include the complete image panel, even if its corners are rounded.

Do not estimate from typical screen layouts or describe your visual reasoning.
Use the supplied image. Keep the reason to one short sentence.

On the first pass, set done=false. On review passes, set done=true only when the
current rectangle is correct and you return exactly the same coordinates.
Otherwise return the best corrected rectangle and set done=false."""


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
                f"Initial localization pass. Original screenshot size: {width} x {height} pixels. "
                "Return the outer rectangle of the complete foreground photo/image panel. "
                "This first pass must set done=false."
            )
            images = [preview]
        else:
            pixel_box = box_1000_to_pixels(current_box, width, height)
            crop = image.crop(pixel_box)
            overlay = self._draw_box(preview, current_box)
            instruction = (
                f"Review pass {iteration}. Original screenshot size: {width} x {height} pixels. "
                "Input image 1 is the original screenshot. Image 2 has the current candidate "
                "rectangle outlined in magenta. Image 3 is the crop made from that rectangle. "
                "Check whether image 3 contains the COMPLETE foreground photo/image panel and "
                "excludes surrounding background/interface. Return coordinates relative to image 1. "
                "If the current rectangle is correct, return exactly the same coordinates and set "
                "done=true. Otherwise return corrected coordinates and set done=false. "
                f"Current bbox_1000: {list(current_box)}."
            )
            images = [preview, overlay, self._resize(crop)]

        request_input: list[dict[str, Any]] = [
            {"type": "text", "content": instruction}
        ]
        request_input.extend(
            {"type": "image", "data_url": self._encode_image(item)}
            for item in images
        )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "system_prompt": SYSTEM_PROMPT,
            "input": request_input,
            "temperature": 0.1,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
        }

        response = self._post_with_retries(payload)
        try:
            response_data = response.json()
            output = response_data["output"]
            if not isinstance(output, list):
                raise TypeError("The 'output' field was not an array.")
        except (ValueError, KeyError, TypeError) as error:
            raise RuntimeError(
                f"LM Studio returned an unexpected native chat response: {response.text}"
            ) from error

        message_text = "\n".join(
            str(item.get("content", ""))
            for item in output
            if isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("content")
        )
        reasoning_text = "\n".join(
            str(item.get("content", ""))
            for item in output
            if isinstance(item, dict)
            and item.get("type") == "reasoning"
            and item.get("content")
        )

        try:
            return parse_reply(message_text, reasoning_text)
        except ValueError as error:
            stats = response_data.get("stats", {})
            total_tokens = stats.get("total_output_tokens") if isinstance(stats, dict) else None
            reasoning_tokens = (
                stats.get("reasoning_output_tokens") if isinstance(stats, dict) else None
            )
            if (
                not message_text.strip()
                and isinstance(total_tokens, (int, float))
                and isinstance(reasoning_tokens, (int, float))
                and total_tokens > 0
                and reasoning_tokens >= total_tokens * 0.9
            ):
                raise RuntimeError(
                    "The selected model spent its entire output budget on reasoning and "
                    "did not return a crop rectangle. Qwen3-VL has separate Thinking and "
                    "Instruct checkpoints; the Thinking checkpoint is not a switchable "
                    "reasoning mode. Download the Qwen3-VL-8B Instruct model in LM Studio "
                    "and rerun Inlay with its default model."
                ) from error

            usage = (
                f" Output tokens: {total_tokens if total_tokens is not None else 'unknown'}; "
                f"reasoning tokens: {reasoning_tokens if reasoning_tokens is not None else 'unknown'}."
            )
            raise RuntimeError(
                "Qwen did not return a valid crop rectangle. "
                f"Message: {message_text!r}; reasoning: {reasoning_text!r}.{usage}"
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
