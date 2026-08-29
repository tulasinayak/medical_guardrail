from unittest.mock import MagicMock, patch

from medical_guardrails.common.schemas import (
    Claim,
    ClaimVerdict,
    DomainQuery,
    EvidenceChunk,
)
from medical_guardrails.orchestrator import MedicalGuardrailPipeline
from medical_guardrails.stage1_slotfill.gate import GateResult
from medical_guardrails.stage3_verify.verification import VerificationResult


def _pipeline():
    return MedicalGuardrailPipeline(settings=MagicMock(), llm_client=MagicMock())


@patch("medical_guardrails.orchestrator.slot_fill_gate")
def test_stops_at_clarification_without_calling_generation_or_verification(mock_gate):
    query = DomainQuery(
        raw_text="x",
        query_type="drug_interaction",
        fields={"drug_names": ["ibuprofen"], "allergies": None},
    )
    mock_gate.return_value = GateResult(
        "needs_clarification", query, ["allergies"], "Do you have any allergies?"
    )

    pipeline = _pipeline()
    with (
        patch("medical_guardrails.orchestrator.retrieve_evidence") as mock_retrieve,
        patch("medical_guardrails.orchestrator.generate_grounded_response") as mock_generate,
        patch("medical_guardrails.orchestrator.verify_response") as mock_verify,
    ):
        result = pipeline.process_query("can I take ibuprofen")

    assert result.status == "needs_clarification"
    assert result.missing_fields == ["allergies"]
    assert result.clarifying_question == "Do you have any allergies?"
    mock_retrieve.assert_not_called()
    mock_generate.assert_not_called()
    mock_verify.assert_not_called()


@patch("medical_guardrails.orchestrator.verify_response")
@patch("medical_guardrails.orchestrator.generate_grounded_response")
@patch("medical_guardrails.orchestrator.retrieve_evidence")
@patch("medical_guardrails.orchestrator.slot_fill_gate")
def test_ready_query_runs_full_generation_and_verification(
    mock_gate, mock_retrieve, mock_generate, mock_verify
):
    query = DomainQuery(
        raw_text="Can I take ibuprofen with warfarin?",
        query_type="drug_interaction",
        fields={
            "drug_names": ["ibuprofen", "warfarin"],
            "allergies": ["lactose"],
            "age_bracket": "adult",
        },
    )
    mock_gate.return_value = GateResult("ready", query, [], None)

    evidence = [
        EvidenceChunk(
            source="ddinter",
            authority="curated_secondary",
            drug_names=["ibuprofen", "warfarin"],
            field_name="interaction_severity",
            text="major",
        )
    ]
    mock_retrieve.return_value = evidence
    mock_generate.return_value = "draft response text"
    mock_verify.return_value = VerificationResult(
        claims=[Claim(claim_text="a", verdict=ClaimVerdict.SUPPORTED)],
        ingredients_found=[],
        ingredient_conflicts=[],
        action="pass",
        final_response="draft response text",
    )

    pipeline = _pipeline()
    result = pipeline.process_query(query.raw_text)

    mock_retrieve.assert_called_once()
    assert mock_retrieve.call_args.kwargs["drug_names"] == ["ibuprofen", "warfarin"]
    mock_generate.assert_called_once_with(query.raw_text, evidence, pipeline.llm_client)
    mock_verify.assert_called_once_with("draft response text", evidence, ["lactose"], pipeline.llm_client)

    assert result.status == "answered"
    assert result.evidence == evidence
    assert result.final_response == "draft response text"


@patch("medical_guardrails.orchestrator.verify_response")
@patch("medical_guardrails.orchestrator.generate_grounded_response")
@patch("medical_guardrails.orchestrator.retrieve_evidence")
@patch("medical_guardrails.orchestrator.slot_fill_gate")
def test_none_allergies_passed_as_empty_list_to_verification(
    mock_gate, mock_retrieve, mock_generate, mock_verify
):
    # GENERAL_INFO doesn't require allergies, so a "ready" query can still have allergies=None
    query = DomainQuery(raw_text="What is ibuprofen?", query_type="general_info", fields={"allergies": None})
    mock_gate.return_value = GateResult("ready", query, [], None)
    mock_retrieve.return_value = []
    mock_generate.return_value = "draft"
    mock_verify.return_value = VerificationResult([], [], [], "pass", "draft")

    pipeline = _pipeline()
    pipeline.process_query(query.raw_text)

    mock_verify.assert_called_once_with("draft", [], [], pipeline.llm_client)
