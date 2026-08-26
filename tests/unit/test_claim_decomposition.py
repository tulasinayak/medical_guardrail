from unittest.mock import MagicMock

from medical_guardrails.stage3_verify.claim_decomposition import decompose_claims


def test_splits_response_into_lines():
    llm_client = MagicMock()
    llm_client.chat.return_value = (
        "Ibuprofen should be avoided with warfarin.\nWarfarin increases bleeding risk."
    )
    claims = decompose_claims("some draft response", llm_client)
    assert claims == [
        "Ibuprofen should be avoided with warfarin.",
        "Warfarin increases bleeding risk.",
    ]


def test_strips_blank_lines():
    llm_client = MagicMock()
    llm_client.chat.return_value = "Claim one.\n\n\nClaim two.\n"
    claims = decompose_claims("draft", llm_client)
    assert claims == ["Claim one.", "Claim two."]


def test_returns_empty_list_for_no_claims():
    llm_client = MagicMock()
    llm_client.chat.return_value = ""
    assert decompose_claims("Please consult a doctor.", llm_client) == []
