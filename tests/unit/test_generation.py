from unittest.mock import MagicMock

from medical_guardrails.common.schemas import EvidenceChunk
from medical_guardrails.stage2_generate.generation import NOT_IN_SOURCES_FALLBACK, generate_grounded_response


def test_passes_system_prompt_and_formatted_evidence_to_llm():
    llm_client = MagicMock()
    llm_client.chat.return_value = "Yes, there is a major interaction."

    evidence = [
        EvidenceChunk(
            source="ddinter",
            authority="curated_secondary",
            drug_names=["ibuprofen", "warfarin"],
            field_name="interaction_severity",
            text="ibuprofen and warfarin have a documented major interaction.",
        )
    ]

    reply = generate_grounded_response("Any interaction?", evidence, llm_client)

    assert reply == "Yes, there is a major interaction."
    messages = llm_client.chat.call_args[0][0]
    assert messages[0]["role"] == "system"
    assert "ONLY" in messages[0]["content"]
    assert "ddinter" in messages[1]["content"]
    assert "Any interaction?" in messages[1]["content"]


def test_empty_evidence_returns_fallback_without_calling_llm():
    llm_client = MagicMock()

    reply = generate_grounded_response("Any interaction?", [], llm_client)

    assert NOT_IN_SOURCES_FALLBACK in reply
    llm_client.chat.assert_not_called()
