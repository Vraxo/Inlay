from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateSelection:
    candidate_id: int
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


def parse_candidate_selection(content: str, candidate_count: int) -> CandidateSelection:
    if candidate_count < 1:
        raise ValueError("At least one candidate is required.")

    data = _find_json_object(content)
    candidate_id = data.get("candidate_id")
    if isinstance(candidate_id, str) and candidate_id.isdecimal():
        candidate_id = int(candidate_id)
    if isinstance(candidate_id, bool) or not isinstance(candidate_id, int):
        raise ValueError("'candidate_id' must be an integer.")
    if not 1 <= candidate_id <= candidate_count:
        raise ValueError(
            f"'candidate_id' must be between 1 and {candidate_count}."
        )

    reason = data.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("'reason' must be a string.")

    return CandidateSelection(candidate_id, reason.strip(), content)
