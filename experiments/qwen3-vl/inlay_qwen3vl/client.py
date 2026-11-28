from __future__ import annotations

import base64
import io
import os
import time
from typing import Any, Callable

import requests
from PIL import Image

from .geometry import box_1000_to_pixels
from .protocol import CandidateSelection, parse_candidate_selection
from .vision import (
    RectangleCandidate,
    draw_boundary_options_overview,
    draw_candidate_overview,
    draw_sharpness_overview,
    rank_sharpness_boundaries,
)


MAX_OUTPUT_TOKENS = 384

SYSTEM_PROMPT = """You are Inlay's foreground-photo boundary detector.

Every input screenshot contains a sharp foreground photograph or image panel
surrounded by a blurred background. The background may be an enlarged, blurred
copy of the same photograph, so similar colors and content do NOT mean it is
part of the foreground. The important distinction is sharp detail inside the
photo versus soft, out-of-focus detail outside it.

You receive four images:
1. The untouched screenshot.
2. The screenshot with numbered OpenCV rectangle proposals.
3. A sharpness map with the same numbered proposals. Brighter areas contain
   more fine detail and are sharper; darker areas are smoother and blurrier.
4. A four-panel boundary atlas. Each panel isolates ONE edge direction and
   draws only the possible positions for that edge. Labels show the candidate
   IDs and pixel coordinates. Use this atlas to map each boundary to its ID.

Find the four outer boundaries of the sharp foreground photo. Inspect each
boundary independently. A correct boundary follows the transition from the
sharp photo to the blurred surround. Reject a line that cuts into sharp photo
content or includes a substantial strip of blurred background. Include the
complete photo panel, including its rounded corners. Do not select a person or
object inside the photograph.

Return four candidate IDs, not one rectangle ID:
- left_candidate_id: choose a vertical line in the LEFT EDGE panel.
- top_candidate_id: choose a horizontal line in the TOP EDGE panel.
- right_candidate_id: choose a vertical line in the RIGHT EDGE panel.
- bottom_candidate_id: choose a horizontal line in the BOTTOM EDGE panel.

The ID labels are repeated in the matching edge panel. Do not copy an ID from
one panel to another unless that ID is also shown in the other panel.

The four IDs may be different. Python will combine the four independently
selected edges into one rectangle. Compare the original image with the
sharpness map; do not choose boundaries merely because a candidate has a
higher geometric score or forms a larger, neater rectangle.

Return exactly one JSON object and nothing else:
{"left_candidate_id": 1, "top_candidate_id": 2, "right_candidate_id": 1, "bottom_candidate_id": 2, "reason": "brief explanation"}

Use 1-based IDs matching the visible labels. Keep the reason to one short
sentence."""


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

    def select_candidate(
        self,
        image: Image.Image,
        candidates: list[RectangleCandidate],
    ) -> CandidateSelection:
        if not candidates:
            raise ValueError("OpenCV did not produce any rectangle candidates.")

        overview = draw_candidate_overview(image, candidates)
        sharpness_overview = draw_sharpness_overview(image, candidates)
        boundary_rankings = rank_sharpness_boundaries(image, candidates)
        boundary_overview = draw_boundary_options_overview(
            image,
            boundary_rankings,
        )
        candidate_lines = []
        for index, candidate in enumerate(candidates, start=1):
            left, top, right, bottom = box_1000_to_pixels(
                candidate.box,
                image.width,
                image.height,
            )
            candidate_lines.append(
                f"{index}: left_x={left}, top_y={top}, right_x={right}, "
                f"bottom_y={bottom}, geometry_score={candidate.score:.3f}, "
                f"source={candidate.source}"
            )

        boundary_lines = []
        coordinate_names = {
            "left": "x",
            "top": "y",
            "right": "x",
            "bottom": "y",
        }
        for side in ("left", "top", "right", "bottom"):
            choices = ", ".join(
                f"ID {int(option['candidate_id'])}={int(option['coordinate'])}px"
                for option in boundary_rankings[side]
            )
            boundary_lines.append(
                f"{side.upper()} ({coordinate_names[side]} coordinates): {choices}"
            )

        instruction = (
            f"Original screenshot dimensions: {image.width} x {image.height} pixels. "
            "Choose each boundary independently, using the matching panel in the "
            "fourth image. The sharp photo is the high-detail region; the surrounding "
            "blurred enlargement is outside the crop. Candidate rectangles and their "
            "coordinates:\n"
            + "\n".join(candidate_lines)
            + "\nIndependent edge choices and their coordinates:\n"
            + "\n".join(boundary_lines)
            + "\nImage 2 shows complete rectangles. Image 3 is the sharpness map. "
            "Image 4 isolates left, right, top, and bottom choices into separate "
            "panels. Select the line at the sharp-to-blurry transition in each panel. "
            "Use the exact ID printed next to that line. The four IDs may differ."
        )

        request_input: list[dict[str, Any]] = [
            {"type": "text", "content": instruction},
            {"type": "image", "data_url": self._encode_image(self._resize(image))},
            {"type": "image", "data_url": self._encode_image(self._resize(overview))},
            {
                "type": "image",
                "data_url": self._encode_image(self._resize(sharpness_overview)),
            },
            {
                "type": "image",
                "data_url": self._encode_image(self._resize(boundary_overview)),
            },
        ]

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
        if not message_text.strip():
            stats = response_data.get("stats", {})
            total_tokens = stats.get("total_output_tokens") if isinstance(stats, dict) else None
            reasoning_tokens = (
                stats.get("reasoning_output_tokens") if isinstance(stats, dict) else None
            )
            raise RuntimeError(
                "The vision model did not return independent boundary selections. "
                f"Message: {message_text!r}. Output tokens: "
                f"{total_tokens if total_tokens is not None else 'unknown'}; "
                f"reasoning tokens: "
                f"{reasoning_tokens if reasoning_tokens is not None else 'unknown'}."
            )

        try:
            return parse_candidate_selection(message_text, len(candidates))
        except ValueError as error:
            raise RuntimeError(
                "The vision model did not return four valid boundary candidate IDs. "
                f"Response: {message_text!r}. {error}"
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
