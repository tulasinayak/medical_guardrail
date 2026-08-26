from unittest.mock import patch

from medical_guardrails.common.schemas import QueryType, StructuredQuery
from medical_guardrails.stage1_slotfill.gate import slot_fill_gate


@patch("medical_guardrails.stage1_slotfill.gate.extract_structured_query")
def test_ready_when_nothing_missing(mock_extract):
    mock_extract.return_value = StructuredQuery(
        raw_text="x",
        query_type=QueryType.GENERAL_INFO,
    )
    result = slot_fill_gate("what is ibuprofen", None)
    assert result.status == "ready"
    assert result.clarifying_question is None
    assert result.missing == []


@patch("medical_guardrails.stage1_slotfill.gate.extract_structured_query")
def test_needs_clarification_asks_about_each_missing_field(mock_extract):
    mock_extract.return_value = StructuredQuery(
        raw_text="x",
        query_type=QueryType.DRUG_INTERACTION,
        drug_names=["ibuprofen", "warfarin"],
        allergies=None,
        age_bracket=None,
    )
    result = slot_fill_gate("can I take ibuprofen with warfarin", None)
    assert result.status == "needs_clarification"
    assert set(result.missing) == {"allergies", "age_bracket"}
    assert "allerg" in result.clarifying_question.lower()
    assert "age" in result.clarifying_question.lower()
