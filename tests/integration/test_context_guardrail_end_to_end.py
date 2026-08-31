"""Real local Ollama server. Run with: pytest tests/integration -m integration"""

import pytest

from medical_guardrails.config import Settings
from medical_guardrails.llm.ollama_client import OllamaClient
from medical_guardrails.context_guardrail.gate import slot_fill_gate

pytestmark = pytest.mark.integration


@pytest.fixture
def llm_client():
    settings = Settings()
    client = OllamaClient(
        host=settings.ollama_host, model=settings.ollama_model, timeout=settings.ollama_timeout_seconds
    )
    assert client.health_check(), "Ollama must be running locally with the configured model pulled"
    return client


def test_missing_allergies_triggers_clarification(llm_client):
    result = slot_fill_gate("Is it safe to take ibuprofen with warfarin?", llm_client)
    assert result.status == "needs_clarification"
    assert "allergies" in result.missing


def test_explicit_no_allergies_is_recognized(llm_client):
    result = slot_fill_gate(
        "Can I take ibuprofen with warfarin? I have no allergies.", llm_client
    )
    assert result.structured_query.fields["allergies"] == []
