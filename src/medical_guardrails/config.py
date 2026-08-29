"""Runtime settings, loaded from environment variables (prefix
MEDICAL_GUARDRAILS_) with sensible free/no-auth defaults for every external
service this project talks to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DDINTER_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "ddinter" / "ddinter.sqlite"


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

    rxnorm_base_url: str = "https://rxnav.nlm.nih.gov/REST"
    openfda_base_url: str = "https://api.fda.gov/drug/label.json"
    medlineplus_base_url: str = "https://wsearch.nlm.nih.gov/ws/query"
    http_timeout_seconds: float = 15.0

    ddinter_db_path: Path = DEFAULT_DDINTER_DB_PATH
