"""Grounded generation: answers a user's question using only the retrieved
evidence, via the local Ollama model. The system prompt is the guardrail
here -- it forbids answering from the model's own memorized knowledge and
requires an explicit fallback when the evidence doesn't cover the question.
"""

from __future__ import annotations

from medical_guardrails.common.schemas import EvidenceChunk
from medical_guardrails.llm.ollama_client import OllamaClient

NOT_IN_SOURCES_FALLBACK = "I don't have reliable information on this in my sources."

SYSTEM_PROMPT = f"""You are a medical information assistant. You must answer ONLY using \
the EVIDENCE block below -- never from your own training or general knowledge, even if you \
believe you know the answer. Each evidence item is tagged with its source and the drug(s) it \
concerns.

If the evidence does not cover part or all of the question, say so explicitly using this exact \
phrase for the uncovered part: "{NOT_IN_SOURCES_FALLBACK}" Do not guess, infer beyond what the \
evidence states, or fill gaps from outside knowledge.

Always end your answer with a line recommending the user confirm with a pharmacist or doctor."""


def _format_evidence(evidence: list[EvidenceChunk]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    lines = []
    for chunk in evidence:
        drugs = ", ".join(chunk.drug_names)
        lines.append(f"[{chunk.source} | {chunk.authority} | {drugs} | {chunk.field_name}] {chunk.text}")
    return "\n".join(lines)


def generate_grounded_response(
    user_query: str, evidence: list[EvidenceChunk], llm_client: OllamaClient
) -> str:
    evidence_block = _format_evidence(evidence)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"EVIDENCE:\n{evidence_block}\n\nQUESTION: {user_query}",
        },
    ]
    return llm_client.chat(messages)
