"""Thin wrapper over OpenAI's /v1/chat/completions endpoint. Same
deliberately-minimal, raw-httpx philosophy as OllamaClient (no `openai` SDK
dependency), and the same chat(messages, format=None) -> str interface, so
every caller in this project works unchanged regardless of which backend
`llm/factory.py` selects.

`format` (a JSON schema, as used by Stage 1's classifier) is translated to
OpenAI's structured-outputs `response_format` -- which additionally
requires `additionalProperties: false` on the schema, unlike Ollama's
grammar-based constraint. That requirement is handled here rather than in
the caller, so the schema classifier.py defines stays backend-agnostic.
Structured outputs need a reasonably modern model (gpt-4o-mini or later);
older models will reject the request.
"""

from __future__ import annotations

import httpx


def _as_strict_schema(schema: dict) -> dict:
    if "additionalProperties" in schema:
        return schema
    return {**schema, "additionalProperties": False}


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]], format: dict | None = None) -> str:
        payload: dict = {"model": self.model, "messages": messages}
        if format is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_extraction",
                    "schema": _as_strict_schema(format),
                    "strict": True,
                },
            }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def health_check(self) -> bool:
        try:
            response = httpx.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5.0,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False
