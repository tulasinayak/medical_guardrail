"""Multi-turn slot-filling: repeatedly runs the single-shot gate
(gate.py's slot_fill_gate), feeding each answer back into the accumulated
conversation text, until every required field is resolved or a question
budget runs out.

Stateless-conversation design: rather than trying to merge extracted
fields incrementally (a DomainQuery diff/merge that doesn't exist),
each round re-runs classification on the *whole* accumulated conversation
text. Simpler and more robust than partial-merge logic -- the model sees
full context every round instead of juggling partial state -- at the cost
of a few extra tokens re-reading earlier turns each time.

Pure logic, no I/O: `ask_fn` is injected so this is testable without a real
terminal, and reusable from a future GUI without change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from medical_guardrails.common.schemas import DomainQuery
from medical_guardrails.llm.base import LLMClient
from medical_guardrails.stage1_slotfill.domain import DomainSchema
from medical_guardrails.stage1_slotfill.domains.medical import MEDICAL_DOMAIN
from medical_guardrails.stage1_slotfill.gate import slot_fill_gate

DEFAULT_MAX_QUESTIONS = 5

TranscriptEntry = tuple[str, str]  # (kind, text) where kind is "query" | "question" | "answer"


@dataclass
class InteractiveSlotFillResult:
    resolved: bool
    structured_query: DomainQuery
    conversation_text: str
    transcript: list[TranscriptEntry] = field(default_factory=list)
    questions_asked: int = 0


def run_interactive_slot_fill(
    initial_query: str,
    llm_client: LLMClient,
    ask_fn: Callable[[str], str],
    max_questions: int = DEFAULT_MAX_QUESTIONS,
    domain: DomainSchema = MEDICAL_DOMAIN,
) -> InteractiveSlotFillResult:
    conversation_text = initial_query
    transcript: list[TranscriptEntry] = [("query", initial_query)]
    questions_asked = 0

    gate_result = slot_fill_gate(conversation_text, llm_client, domain)
    while gate_result.status == "needs_clarification" and questions_asked < max_questions:
        questions_asked += 1
        transcript.append(("question", gate_result.clarifying_question))
        answer = ask_fn(gate_result.clarifying_question)
        transcript.append(("answer", answer))
        conversation_text = f"{conversation_text}\n{answer}"
        gate_result = slot_fill_gate(conversation_text, llm_client, domain)

    return InteractiveSlotFillResult(
        resolved=gate_result.status == "ready",
        structured_query=gate_result.structured_query,
        conversation_text=conversation_text,
        transcript=transcript,
        questions_asked=questions_asked,
    )
