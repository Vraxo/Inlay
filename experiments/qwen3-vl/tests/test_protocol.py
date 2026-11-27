import json

import pytest

from inlay_qwen3vl.protocol import parse_candidate_selection


def test_parses_candidate_selection():
    result = parse_candidate_selection(
        '{"candidate_id": 3, "reason": "This box encloses the complete photo."}',
        8,
    )
    assert result.candidate_id == 3
    assert result.reason == "This box encloses the complete photo."


def test_parses_candidate_selection_inside_code_fence():
    result = parse_candidate_selection(
        'Selection:\n```json\n{"candidate_id": "2", "reason": "Best fit"}\n```',
        4,
    )
    assert result.candidate_id == 2


@pytest.mark.parametrize("candidate_id", [0, 5, -1])
def test_rejects_candidate_ids_outside_the_proposal_list(candidate_id):
    with pytest.raises(ValueError, match="between 1 and 4"):
        parse_candidate_selection(
            f'{{"candidate_id": {candidate_id}, "reason": "Invalid"}}',
            4,
        )


@pytest.mark.parametrize("candidate_id", [True, 1.5, None])
def test_rejects_non_integer_candidate_ids(candidate_id):
    with pytest.raises(ValueError, match="'candidate_id' must be an integer"):
        parse_candidate_selection(
            json.dumps({"candidate_id": candidate_id, "reason": "Invalid"}),
            4,
        )


def test_rejects_missing_candidate_selection():
    with pytest.raises(ValueError, match="candidate_id"):
        parse_candidate_selection('{"reason": "I cannot decide."}', 4)


def test_rejects_empty_candidate_list():
    with pytest.raises(ValueError, match="At least one candidate"):
        parse_candidate_selection('{"candidate_id": 1}', 0)
