"""The Context Guardrail: classify + extract, then either hand back a
ready DomainQuery or a clarifying question -- never both, and Main LLM
must not run on "needs_clarification". Domain-parametrized (defaults to
the medical domain) -- see context_guardrail/domain.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from medical_guardrails.common.schemas import DomainQuery
from medical_guardrails.context_guardrail.classifier import extract_structured_query
from medical_guardrails.context_guardrail.domain import DomainSchema
from medical_guardrails.context_guardrail.domains.medical import MEDICAL_DOMAIN
from medical_guardrails.context_guardrail.required_fields import missing_fields
from medical_guardrails.llm.base import LLMClient

GateStatus = Literal["ready", "needs_clarification"]


@dataclass
class GateResult:
    status: GateStatus
    structured_query: DomainQuery
    missing: list[str]
    clarifying_question: str | None


def slot_fill_gate(
    raw_text: str, llm_client: LLMClient, domain: DomainSchema = MEDICAL_DOMAIN
) -> GateResult:
    query = extract_structured_query(raw_text, llm_client, domain)
    missing = missing_fields(query, domain)

    if not missing:
        return GateResult("ready", query, [], None)

    question = " ".join(domain.fields[name].clarifying_question for name in missing)
    return GateResult("needs_clarification", query, missing, question)
