from unittest.mock import MagicMock, patch

from medical_guardrails.common.schemas import Claim, ClaimVerdict, EvidenceChunk
from medical_guardrails.stage3_verify.verification import verify_response


def _ddinter_evidence():
    return [
        EvidenceChunk(
            source="ddinter",
            authority="curated_secondary",
            drug_names=["ibuprofen", "warfarin"],
            field_name="interaction_severity",
            text="ibuprofen and warfarin have a documented major interaction.",
        )
    ]


@patch("medical_guardrails.stage3_verify.verification.decompose_claims")
@patch("medical_guardrails.stage3_verify.verification.verify_claims")
def test_passes_through_when_all_claims_supported_and_no_ingredient_conflict(
    mock_verify_claims, mock_decompose
):
    mock_decompose.return_value = ["claim a"]
    mock_verify_claims.return_value = [Claim(claim_text="claim a", verdict=ClaimVerdict.SUPPORTED)]

    result = verify_response("draft text", _ddinter_evidence(), [], MagicMock())

    assert result.action == "pass"
    assert result.final_response.startswith("draft text")
    assert result.ingredient_conflicts == []


@patch("medical_guardrails.stage3_verify.verification.decompose_claims")
@patch("medical_guardrails.stage3_verify.verification.verify_claims")
def test_blocks_when_a_claim_is_contradicted(mock_verify_claims, mock_decompose):
    mock_decompose.return_value = ["claim a"]
    mock_verify_claims.return_value = [Claim(claim_text="claim a", verdict=ClaimVerdict.CONTRADICTED)]

    result = verify_response("draft text", _ddinter_evidence(), [], MagicMock())

    assert result.action == "block"
    assert "claim a" in result.final_response
    assert "draft text" not in result.final_response


@patch("medical_guardrails.stage3_verify.verification.decompose_claims")
@patch("medical_guardrails.stage3_verify.verification.verify_claims")
def test_blocks_on_ingredient_conflict_even_if_claims_are_all_supported(
    mock_verify_claims, mock_decompose
):
    mock_decompose.return_value = ["claim a"]
    mock_verify_claims.return_value = [Claim(claim_text="claim a", verdict=ClaimVerdict.SUPPORTED)]

    evidence = _ddinter_evidence() + [
        EvidenceChunk(
            source="openfda",
            authority="regulatory",
            drug_names=["ibuprofen"],
            field_name="inactive_ingredient",
            text="lactose anhydrous",
        )
    ]

    result = verify_response("draft text", evidence, ["lactose"], MagicMock())

    assert result.action == "block"
    assert "lactose" in result.final_response


@patch("medical_guardrails.stage3_verify.verification.decompose_claims")
@patch("medical_guardrails.stage3_verify.verification.verify_claims")
def test_ingredients_section_always_rendered_on_pass(mock_verify_claims, mock_decompose):
    mock_decompose.return_value = []
    mock_verify_claims.return_value = []

    evidence = [
        EvidenceChunk(
            source="openfda",
            authority="regulatory",
            drug_names=["ibuprofen"],
            field_name="active_ingredient",
            text="ibuprofen",
        )
    ]

    result = verify_response("draft text", evidence, [], MagicMock())

    assert result.action == "pass"
    assert "Ingredients found" in result.final_response
    assert "ibuprofen" in result.final_response
