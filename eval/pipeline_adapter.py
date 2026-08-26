"""Adapts MedicalGuardrailPipeline.process_query() to a flat
{"action": ..., ...} shape, collapsing its actual outcomes into the closed
set of action labels functional_cases.jsonl scores against exact-string:
ask_clarification, answer_grounded, block_ingredient_match,
block_unsupported_claim, fallback_not_in_sources.

Shared by eval/score.py and eval/targets/guardrail_target.py so the mapping
lives in exactly one place.
"""

from __future__ import annotations

from medical_guardrails.orchestrator import MedicalGuardrailPipeline, PipelineResult
from medical_guardrails.stage2_generate.generation import NOT_IN_SOURCES_FALLBACK

ASK_CLARIFICATION = "ask_clarification"
ANSWER_GROUNDED = "answer_grounded"
BLOCK_INGREDIENT_MATCH = "block_ingredient_match"
BLOCK_UNSUPPORTED_CLAIM = "block_unsupported_claim"
FALLBACK_NOT_IN_SOURCES = "fallback_not_in_sources"


def classify_action(result: PipelineResult) -> str:
    if result.status == "needs_clarification":
        return ASK_CLARIFICATION

    verification = result.verification
    if verification is not None and verification.ingredient_conflicts:
        return BLOCK_INGREDIENT_MATCH
    if verification is not None and verification.action == "block":
        # Blocked, but not on an ingredient match -> a claim came back
        # contradicted or unsupported. The task's closed action set has no
        # separate "contradicted" label, so both map here.
        return BLOCK_UNSUPPORTED_CLAIM
    if NOT_IN_SOURCES_FALLBACK in (result.final_response or ""):
        return FALLBACK_NOT_IN_SOURCES
    return ANSWER_GROUNDED


def run_pipeline(input_text: str, pipeline: MedicalGuardrailPipeline) -> dict:
    result = pipeline.process_query(input_text)
    return {
        "action": classify_action(result),
        "response": result.final_response,
        "status": result.status,
        "missing_fields": result.missing_fields,
        "result": result,
    }
