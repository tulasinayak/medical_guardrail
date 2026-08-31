from unittest.mock import patch

from medical_guardrails.common.schemas import DomainQuery
from medical_guardrails.context_guardrail.gate import slot_fill_gate


@patch("medical_guardrails.context_guardrail.gate.extract_structured_query")
def test_ready_when_nothing_missing(mock_extract):
    mock_extract.return_value = DomainQuery(
        raw_text="x", query_type="general_info", answer_scope="general", fields={}
    )
    result = slot_fill_gate("what is ibuprofen", None)
    assert result.status == "ready"
    assert result.clarifying_question is None
    assert result.missing == []


@patch("medical_guardrails.context_guardrail.gate.extract_structured_query")
def test_needs_clarification_asks_about_each_missing_field(mock_extract):
    mock_extract.return_value = DomainQuery(
        raw_text="x",
        query_type="drug_interaction",
        answer_scope="personal",
        fields={
            "drug_names": ["ibuprofen", "warfarin"],
            "allergies": None,
            "age_bracket": None,
        },
    )
    result = slot_fill_gate("can I take ibuprofen with warfarin", None)
    assert result.status == "needs_clarification"
    assert set(result.missing) == {"allergies", "age_bracket"}
    assert "allerg" in result.clarifying_question.lower()
    assert "age" in result.clarifying_question.lower()


@patch("medical_guardrails.context_guardrail.gate.extract_structured_query")
def test_general_scope_is_ready_even_with_type_requiring_fields(mock_extract):
    # drug_interaction normally requires allergies/age_bracket, but a
    # "general" question shouldn't be gated on personal fields at all.
    mock_extract.return_value = DomainQuery(
        raw_text="x",
        query_type="drug_interaction",
        answer_scope="general",
        fields={"drug_names": ["ibuprofen", "warfarin"], "allergies": None, "age_bracket": None},
    )
    result = slot_fill_gate("is combining ibuprofen and warfarin generally risky", None)
    assert result.status == "ready"
    assert result.missing == []
