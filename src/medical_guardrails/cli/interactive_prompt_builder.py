"""Interactive front end covering both stages combined: type an initial
request, answer whatever clarifying questions the Context Guardrail asks
(up to --max-questions, default 5) until it's satisfied or the budget runs
out, then Main LLM generates an answer for real -- not just a saved
prompt. The medical domain's ingredient/allergy safety check also runs
for real. Everything (the conversation, the exact Main LLM prompt, the
answer, and any ingredient-conflict result) is saved to one file for
inspection.

Main LLM runs even if the Context Guardrail's question budget ran out
without every field resolving, so you can see what it does with a partial
context -- this is a deliberate diagnostic choice specific to this tool:
the answer will explicitly say what it couldn't personalize, which is
also exactly what the real pipeline (MedicalGuardrailPipeline.
process_query(), used by pipeline_once.py and eval/score.py) does once
its own question budget runs out via the multi-turn path.

Pass --no-generate to only build and save the Main LLM prompt without
actually calling the LLM.

Usage:
    python -m medical_guardrails.cli.interactive_prompt_builder
    python -m medical_guardrails.cli.interactive_prompt_builder "Can I take ibuprofen with warfarin?"
    python -m medical_guardrails.cli.interactive_prompt_builder --max-questions 3 --output my_run.txt
    python -m medical_guardrails.cli.interactive_prompt_builder --no-generate
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from medical_guardrails.common.schemas import DomainQuery
from medical_guardrails.config import Settings
from medical_guardrails.context_guardrail.interactive import (
    DEFAULT_MAX_QUESTIONS,
    InteractiveSlotFillResult,
    run_interactive_slot_fill,
)
from medical_guardrails.llm.factory import build_llm_client
from medical_guardrails.main_llm.generation import build_messages, generate_answer
from medical_guardrails.medical.ingredient_safety import IngredientCheckResult, check_drug_allergy_conflicts
from medical_guardrails.medical.openfda_client import OpenFDAClient

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "eval" / "results" / "interactive_prompts"

_TRANSCRIPT_LABELS = {
    "query": "USER (initial request)",
    "question": "GUARDRAIL asked",
    "answer": "USER answered",
}


def _terminal_ask(question: str) -> str:
    print(f"\n{question}")
    return input("> ")


def render_run_file(
    messages: list[dict[str, str]],
    result: InteractiveSlotFillResult,
    answer: str | None,
    ingredient_check: IngredientCheckResult | None,
) -> str:
    lines: list[str] = ["=== CONTEXT GUARDRAIL: CONVERSATION ==="]
    for kind, text in result.transcript:
        lines.append(f"[{_TRANSCRIPT_LABELS[kind]}] {text}")

    lines.append("")
    lines.append(
        f"=== CONTEXT GUARDRAIL: RESOLVED: {result.resolved} (questions asked: {result.questions_asked}) ==="
    )
    lines.append(f"Extracted fields: {result.structured_query.model_dump(exclude={'raw_text'})}")
    lines.append("")

    if not result.resolved:
        lines.append(
            f"NOTE: question budget ran out with these still unresolved: {result.missing}. "
            "Main LLM runs anyway, told explicitly which fields it doesn't have."
        )
        lines.append("")

    lines.append("=== PROMPT SENT TO MAIN LLM ===")
    for message in messages:
        lines.append(f"--- {message['role'].upper()} ---")
        lines.append(message["content"])
        lines.append("")

    if answer is not None:
        lines.append("=== MAIN LLM ANSWER ===")
        lines.append(answer)
        lines.append("")

    if ingredient_check is not None:
        lines.append("=== MEDICAL INGREDIENT/ALLERGY CHECK ===")
        lines.append(f"Ingredients found: {ingredient_check.ingredients_found}")
        lines.append(f"Conflicts: {ingredient_check.conflicts}")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="*", help="Initial request (omit to be prompted for one)")
    parser.add_argument(
        "--max-questions", type=int, default=DEFAULT_MAX_QUESTIONS, help=f"default {DEFAULT_MAX_QUESTIONS}"
    )
    parser.add_argument(
        "--output", default=None, help="Output file path (default: a timestamped file under eval/results/interactive_prompts/)"
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Only build and save the Main LLM prompt -- do not actually call the LLM.",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    llm_client = build_llm_client(settings)
    if not llm_client.health_check():
        print(f"[!!] LLM backend ({settings.llm_provider}) not reachable", file=sys.stderr)
        return 1

    initial_query = " ".join(args.query) if args.query else input("Enter your medical question: ")

    print(
        f"\n(Using {settings.llm_provider} -- each question asked and each answer you give triggers "
        "one classification call to re-check what's still missing, so this may take a while.)"
    )

    try:
        result = run_interactive_slot_fill(
            initial_query, llm_client, ask_fn=_terminal_ask, max_questions=args.max_questions
        )
    except (EOFError, KeyboardInterrupt):
        print("\n\nInput ended before the guardrail was satisfied -- exiting without saving anything.", file=sys.stderr)
        return 1

    status = "Resolved" if result.resolved else "NOT fully resolved"
    print(f"\n--- Context Guardrail: {status} after {result.questions_asked} question(s) ---")
    print(f"Extracted: {result.structured_query.model_dump(exclude={'raw_text'})}")

    query: DomainQuery = result.structured_query
    messages = build_messages(result.conversation_text, query.fields, result.missing)

    answer: str | None = None
    ingredient_check: IngredientCheckResult | None = None
    if args.no_generate:
        print("\n--no-generate set -- not calling the LLM. Saving the prompt only.")
    else:
        print("\nRunning Main LLM...")
        answer = generate_answer(result.conversation_text, query.fields, result.missing, llm_client)
        print(f"Answer:\n{answer}")

        drug_names = query.fields.get("drug_names") or []
        allergies = query.fields.get("allergies") or []
        if drug_names and allergies:
            print("\nRunning medical ingredient/allergy check...")
            openfda_client = OpenFDAClient(settings.openfda_base_url, settings.http_timeout_seconds)
            ingredient_check = check_drug_allergy_conflicts(drug_names, allergies, openfda_client)
            print(f"Conflicts: {ingredient_check.conflicts}")

    output_path = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_run_file(messages, result, answer, ingredient_check), encoding="utf-8")

    print(f"\nSaved run -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
