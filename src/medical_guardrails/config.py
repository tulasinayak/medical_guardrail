"""Runtime settings, loaded from environment variables (prefix
MEDICAL_GUARDRAILS_) with sensible free/no-auth defaults for every external
service this project talks to.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEDICAL_GUARDRAILS_")

    llm_provider: Literal["ollama", "openai"] = "ollama"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "mistral:latest"
    ollama_timeout_seconds: float = 60.0

    # openai_api_key falls back to the standard OPENAI_API_KEY env var if the
    # project-prefixed one isn't set, so an existing key in your environment
    # just works without renaming it.
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MEDICAL_GUARDRAILS_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = 60.0

    # Used only by the medical demo's narrow ingredient/allergy safety check
    # (medical/openfda_client.py) -- the core Context Guardrail -> Main LLM
    # pipeline makes no external evidence calls at all.
    openfda_base_url: str = "https://api.fda.gov/drug/label.json"
    http_timeout_seconds: float = 15.0
