"""Builds whichever LLM backend Settings.llm_provider selects, so callers
(the orchestrator, each stage's standalone CLI) don't hardcode a specific
client. Ollama is the default (free, local, no API key); OpenAI is an
opt-in alternative for anyone with an API key who wants to compare
reliability against a hosted model.
"""

from __future__ import annotations

from medical_guardrails.config import Settings
from medical_guardrails.llm.base import LLMClient
from medical_guardrails.llm.ollama_client import OllamaClient
from medical_guardrails.llm.openai_client import OpenAIClient


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "MEDICAL_GUARDRAILS_LLM_PROVIDER=openai requires an API key -- set "
                "OPENAI_API_KEY or MEDICAL_GUARDRAILS_OPENAI_API_KEY."
            )
        return OpenAIClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout=settings.openai_timeout_seconds,
        )

    return OllamaClient(
        host=settings.ollama_host,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout_seconds,
    )
