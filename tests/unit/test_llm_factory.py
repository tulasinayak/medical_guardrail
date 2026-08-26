import pytest

from medical_guardrails.config import Settings
from medical_guardrails.llm.factory import build_llm_client
from medical_guardrails.llm.ollama_client import OllamaClient
from medical_guardrails.llm.openai_client import OpenAIClient


def test_defaults_to_ollama():
    client = build_llm_client(Settings())
    assert isinstance(client, OllamaClient)


def test_builds_openai_client_when_selected():
    settings = Settings(llm_provider="openai", openai_api_key="sk-test", openai_model="gpt-4o-mini")
    client = build_llm_client(settings)
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4o-mini"
    assert client.api_key == "sk-test"


def test_openai_without_api_key_raises():
    settings = Settings(llm_provider="openai", openai_api_key=None)
    with pytest.raises(ValueError, match="API key"):
        build_llm_client(settings)
