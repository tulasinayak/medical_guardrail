"""End-to-end pipeline entry point: a single raw request goes through the
Context Guardrail (decide whether there's enough context) then Main LLM
(generate the answer), with no separate retrieval or verification stage.
The Context Guardrail is responsible for extracting drug names/allergies/
etc. from the text itself -- nothing is passed in separately.

Usage:
    python -m medical_guardrails.cli.pipeline_once "Can I take ibuprofen with warfarin?"
"""

from __future__ import annotations

import sys

from medical_guardrails.orchestrator import MedicalGuardrailPipeline


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print('usage: python -m medical_guardrails.cli.pipeline_once "query text"', file=sys.stderr)
        return 2

    query = " ".join(argv)
    pipeline = MedicalGuardrailPipeline()

    if not pipeline.llm_client.health_check():
        provider = pipeline.settings.llm_provider
        detail = (
            f"{pipeline.settings.ollama_host} with model {pipeline.settings.ollama_model}"
            if provider == "ollama"
            else f"OpenAI model {pipeline.settings.openai_model} (check API key and network)"
        )
        print(f"[!!] LLM backend ({provider}) not reachable: {detail}", file=sys.stderr)
        return 1

    result = pipeline.process_query(query)

    print(f"Query type: {result.structured_query.query_type}", file=sys.stderr)
    print(f"Answer scope: {result.structured_query.answer_scope}", file=sys.stderr)

    if result.status == "needs_clarification":
        print(f"Missing fields: {result.missing_fields}", file=sys.stderr)
        print(f"\n{result.clarifying_question}")
        return 0

    if result.ingredient_check is not None:
        print(f"Ingredient conflicts: {result.ingredient_check.conflicts}", file=sys.stderr)

    print(f"\n{result.final_response}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
