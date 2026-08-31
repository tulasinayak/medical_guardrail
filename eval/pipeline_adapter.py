"""Adapts MedicalGuardrailPipeline.process_query() to a flat
{"action": ..., ...} shape, collapsing its actual outcomes into a closed
set of action labels functional_cases.jsonl scores against exact-string:
ask_clarification, answered, blocked_ingredient_match.

Much smaller taxonomy than before the retrieval/verification removal --
there's no "no evidence" state anymore (Main LLM always has something to
say) and no separate claim-verdict-driven block (that stage is gone).

Shared by eval/score.py so the mapping lives in exactly one place.
"""

from __future__ import annotations

from medical_guardrails.orchestrator import MedicalGuardrailPipeline, PipelineResult

ASK_CLARIFICATION = "ask_clarification"
ANSWERED = "answered"
BLOCKED_INGREDIENT_MATCH = "blocked_ingredient_match"


def classify_action(result: PipelineResult) -> str:
    if result.status == "needs_clarification":
        return ASK_CLARIFICATION
    if result.ingredient_check is not None and result.ingredient_check.conflicts:
        return BLOCKED_INGREDIENT_MATCH
    return ANSWERED


def run_pipeline(input_text: str, pipeline: MedicalGuardrailPipeline) -> dict:
    result = pipeline.process_query(input_text)
    return {
        "action": classify_action(result),
        "response": result.final_response,
        "status": result.status,
        "missing_fields": result.missing_fields,
        "answer_scope": result.structured_query.answer_scope,
        "result": result,
    }
