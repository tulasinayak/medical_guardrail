"""Checks each decomposed claim against the retrieved evidence via LLM-as-
judge, in a single batched call rather than one call per claim (this
project's local model is CPU-bound and slow, so minimizing round trips
matters). A v2 could swap this for a dedicated NLI classifier (e.g.
DeBERTa-MNLI) without changing the interface.

Fails closed: any claim whose verdict line is missing or unparseable is
treated as UNSUPPORTED rather than silently dropped or assumed safe.
"""

from __future__ import annotations

import re

from medical_guardrails.common.schemas import Claim, ClaimVerdict, EvidenceChunk
from medical_guardrails.llm.ollama_client import OllamaClient

SYSTEM_PROMPT = """You are a strict fact-checker. Given EVIDENCE and a numbered list of CLAIMS, \
decide for each claim whether the evidence SUPPORTS it, CONTRADICTS it, or is UNSUPPORTED \
(the evidence says nothing about it or doesn't cover it).

Respond with exactly one line per claim, in the same order, in the exact format:
<N>: <SUPPORTED|CONTRADICTED|UNSUPPORTED>

No explanations, no extra lines."""

_VERDICT_LINE = re.compile(r"^\s*(\d+)\s*[:.\-]\s*(SUPPORTED|CONTRADICTED|UNSUPPORTED)", re.IGNORECASE)


def _format_evidence(evidence: list[EvidenceChunk]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    lines = []
    for chunk in evidence:
        drugs = ", ".join(chunk.drug_names)
        lines.append(f"[{chunk.source} | {drugs} | {chunk.field_name}] {chunk.text}")
    return "\n".join(lines)


def verify_claims(
    claims: list[str], evidence: list[EvidenceChunk], llm_client: OllamaClient
) -> list[Claim]:
    if not claims:
        return []

    numbered_claims = "\n".join(f"{i + 1}. {claim}" for i, claim in enumerate(claims))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"EVIDENCE:\n{_format_evidence(evidence)}\n\nCLAIMS:\n{numbered_claims}",
        },
    ]
    raw = llm_client.chat(messages)

    verdicts: dict[int, ClaimVerdict] = {}
    for line in raw.splitlines():
        match = _VERDICT_LINE.match(line)
        if match:
            index, verdict_text = int(match.group(1)), match.group(2).upper()
            verdicts[index] = ClaimVerdict(verdict_text.lower())

    return [
        Claim(
            claim_text=claim,
            verdict=verdicts.get(i + 1, ClaimVerdict.UNSUPPORTED),
        )
        for i, claim in enumerate(claims)
    ]
