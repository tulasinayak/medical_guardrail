"""The pre-generation gate: classify + extract, then either hand back a
ready StructuredQuery or a clarifying question -- never both, and
generation must not proceed on "needs_clarification".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from medical_guardrails.common.schemas import StructuredQuery
from medical_guardrails.llm.base import LLMClient
from medical_guardrails.stage1_slotfill.classifier import extract_structured_query
from medical_guardrails.stage1_slotfill.required_fields import CLARIFYING_QUESTIONS, missing_fields

GateStatus = Literal["ready", "needs_clarification"]


@dataclass
class GateResult:
    status: GateStatus
    structured_query: StructuredQuery
    missing: list[str]
    clarifying_question: str | None


def slot_fill_gate(raw_text: str, llm_client: LLMClient) -> GateResult:
    query = extract_structured_query(raw_text, llm_client)
    missing = missing_fields(query)

    if not missing:
        return GateResult("ready", query, [], None)

    question = " ".join(CLARIFYING_QUESTIONS[field] for field in missing)
    return GateResult("needs_clarification", query, missing, question)
