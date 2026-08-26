"""Thin wrapper over a local Ollama server's /api/chat endpoint.

Adapted from the pii_guardrails sibling project's client of the same name:
deliberately minimal, raw httpx rather than the `ollama` pip package, so the
integration stays readable in one glance.
"""

from __future__ import annotations

import httpx


class OllamaClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "mistral:latest",
        timeout: float = 60.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]], format: dict | None = None) -> str:
        """messages: list of {"role": "user"|"assistant"|"system", "content": str}.
        `format`: an optional JSON schema -- when given, Ollama grammar-constrains
        decoding so the reply structurally cannot deviate from it (see
        docs.ollama.com/capabilities/structured-outputs). Returns the assistant's
        reply text (a JSON string when `format` is set)."""
        payload = {"model": self.model, "messages": messages, "stream": False}
        if format is not None:
            payload["format"] = format
        response = httpx.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def health_check(self) -> bool:
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=5.0)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            return self.model in models
        except (httpx.HTTPError, KeyError, ValueError):
            return False
