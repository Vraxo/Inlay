from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateSelection:
    candidate_id: int | None
    reason: str
    raw_text: str
    left_candidate_id: int | None = None
    top_candidate_id: int | None = None
    right_candidate_id: int | None = None
    bottom_candidate_id: int | None = None

    @property
    def uses_independent_boundaries(self) -> bool:
        return all(
            value is not None
            for value in (
                self.left_candidate_id,
                self.top_candidate_id,
                self.right_candidate_id,
                self.bottom_candidate_id,
            )
        )


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


def _candidate_id(value: object, key: str, candidate_count: int) -> int:
    if isinstance(value, str) and value.isdecimal():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{key}' must be an integer.")
    if not 1 <= value <= candidate_count:
        raise ValueError(
            f"'{key}' must be between 1 and {candidate_count}."
        )
    return value


def parse_candidate_selection(content: str, candidate_count: int) -> CandidateSelection:
    if candidate_count < 1:
        raise ValueError("At least one candidate is required.")

    data = _find_json_object(content)
    edge_keys = (
        "left_candidate_id",
        "top_candidate_id",
        "right_candidate_id",
        "bottom_candidate_id",
    )
    supplied_edges = [key in data for key in edge_keys]

    if any(supplied_edges):
        if not all(supplied_edges):
            missing = [
                key for key, supplied in zip(edge_keys, supplied_edges, strict=True)
                if not supplied
            ]
            raise ValueError(
                "Independent boundary selection is incomplete; missing: "
                + ", ".join(missing)
            )
        candidate_ids = [
            _candidate_id(data[key], key, candidate_count)
            for key in edge_keys
        ]
        candidate_id = None
    else:
        candidate_id = _candidate_id(
            data.get("candidate_id"),
            "candidate_id",
            candidate_count,
        )
        candidate_ids = [None, None, None, None]

    reason = data.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("'reason' must be a string.")

    return CandidateSelection(
        candidate_id,
        reason.strip(),
        content,
        *candidate_ids,
    )
