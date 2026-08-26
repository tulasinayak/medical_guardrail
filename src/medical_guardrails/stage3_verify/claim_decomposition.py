"""Splits a draft response into atomic, independently-checkable factual
claims via the local LLM. Uses a plain one-claim-per-line output format
rather than JSON -- more robust to parse reliably out of a small local
model than well-formed JSON tends to be.
"""

from __future__ import annotations

from medical_guardrails.llm.ollama_client import OllamaClient

SYSTEM_PROMPT = """You decompose a medical assistant's draft response into atomic factual claims.

Rules:
- Each claim must be a single, independently-checkable factual assertion (e.g. \
"Ibuprofen should be avoided with warfarin" is one claim; do not combine multiple facts into one line).
- Do not include hedges, recommendations to "consult a doctor", or questions -- only checkable \
factual assertions.
- Output ONLY the claims, one per line, with no numbering, bullets, or extra commentary.
- If the response contains no checkable factual claims at all, output nothing."""


def decompose_claims(draft_response: str, llm_client: OllamaClient) -> list[str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": draft_response},
    ]
    raw = llm_client.chat(messages)
    return [line.strip() for line in raw.splitlines() if line.strip()]
