from unittest.mock import MagicMock

from medical_guardrails.common.schemas import ClaimVerdict, EvidenceChunk
from medical_guardrails.stage3_verify.entailment import verify_claims


def _evidence():
    return [
        EvidenceChunk(
            source="ddinter",
            drug_names=["ibuprofen", "warfarin"],
            field_name="interaction_severity",
            text="ibuprofen and warfarin have a documented major interaction.",
        )
    ]


def test_parses_verdicts_in_order():
    llm_client = MagicMock()
    llm_client.chat.return_value = "1: SUPPORTED\n2: CONTRADICTED\n3: UNSUPPORTED"
    claims = verify_claims(["a", "b", "c"], _evidence(), llm_client)
    assert [c.verdict for c in claims] == [
        ClaimVerdict.SUPPORTED,
        ClaimVerdict.CONTRADICTED,
        ClaimVerdict.UNSUPPORTED,
    ]
    assert [c.claim_text for c in claims] == ["a", "b", "c"]


def test_missing_verdict_line_fails_closed_to_unsupported():
    llm_client = MagicMock()
    llm_client.chat.return_value = "1: SUPPORTED"  # no line for claim 2
    claims = verify_claims(["a", "b"], _evidence(), llm_client)
    assert claims[0].verdict == ClaimVerdict.SUPPORTED
    assert claims[1].verdict == ClaimVerdict.UNSUPPORTED


def test_malformed_response_fails_closed_to_unsupported():
    llm_client = MagicMock()
    llm_client.chat.return_value = "I think claim 1 seems fine actually"
    claims = verify_claims(["a"], _evidence(), llm_client)
    assert claims[0].verdict == ClaimVerdict.UNSUPPORTED


def test_empty_claims_list_short_circuits_without_calling_llm():
    llm_client = MagicMock()
    assert verify_claims([], _evidence(), llm_client) == []
    llm_client.chat.assert_not_called()


def test_is_case_insensitive_to_verdict_text():
    llm_client = MagicMock()
    llm_client.chat.return_value = "1: supported"
    claims = verify_claims(["a"], _evidence(), llm_client)
    assert claims[0].verdict == ClaimVerdict.SUPPORTED
