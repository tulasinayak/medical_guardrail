"""The interface every LLM backend in this project implements, so callers
(classifier.py, generation.py, claim_decomposition.py, entailment.py,
gate.py, the orchestrator) work unchanged regardless of which concrete
client is configured -- see llm/factory.py for how one gets selected.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def chat(self, messages: list[dict[str, str]], format: dict | None = None) -> str:
        """messages: list of {"role": ..., "content": str}. `format`, when given, is a
        JSON schema the reply must conform to (see each backend for how it's enforced)."""
        ...

    def health_check(self) -> bool: ...
