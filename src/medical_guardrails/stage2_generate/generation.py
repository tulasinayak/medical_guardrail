"""Grounded generation: answers a user's question using only the retrieved
evidence. The system prompt is the guardrail for *partial* coverage -- it
forbids answering from the model's own memorized knowledge and requires an
explicit fallback when the evidence doesn't cover part of the question.

For *no* coverage at all (evidence is empty), that guarantee is enforced in
code instead of by prompt: generate_grounded_response returns the fallback
directly without calling the LLM. A live eval run found the prompt-only
version occasionally failed silently -- given zero evidence, the model
sometimes answered from its own general knowledge anyway rather than
declining, which is exactly the failure mode this guardrail exists to
prevent. There's nothing a retrieved-evidence-grounded prompt can add when
there's no evidence to ground it in, so skipping the call entirely closes
that gap rather than just asking the model more firmly.
"""

from __future__ import annotations

from medical_guardrails.common.schemas import EvidenceChunk
from medical_guardrails.llm.base import LLMClient

NOT_IN_SOURCES_FALLBACK = "I don't have reliable information on this in my sources."
NO_EVIDENCE_RESPONSE = f"{NOT_IN_SOURCES_FALLBACK} Please consult a pharmacist or doctor for guidance specific to your situation."

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


def build_generation_messages(user_query: str, evidence: list[EvidenceChunk]) -> list[dict[str, str]]:
    """The exact message list Stage 2 would send to the LLM for this query
    and evidence -- exposed separately from generate_grounded_response so
    callers (e.g. the interactive prompt builder) can inspect/save it
    without actually invoking the model."""
    evidence_block = _format_evidence(evidence)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"EVIDENCE:\n{evidence_block}\n\nQUESTION: {user_query}",
        },
    ]


def generate_grounded_response(
    user_query: str, evidence: list[EvidenceChunk], llm_client: LLMClient
) -> str:
    if not evidence:
        return NO_EVIDENCE_RESPONSE

    messages = build_generation_messages(user_query, evidence)
    return llm_client.chat(messages)
