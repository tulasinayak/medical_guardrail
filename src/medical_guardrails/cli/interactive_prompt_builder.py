"""Interactive Stage 1 gate: type an initial medical query, then answer
whatever clarifying questions the guardrail asks (up to --max-questions,
default 5) until it's satisfied or the budget runs out. Once resolved,
retrieves Stage 2 evidence and builds the exact prompt that would be sent
to the generation model -- but does NOT call it. Saves that prompt to a
file for inspection instead.

Usage:
    python -m medical_guardrails.cli.interactive_prompt_builder
    python -m medical_guardrails.cli.interactive_prompt_builder "Can I take ibuprofen with warfarin?"
    python -m medical_guardrails.cli.interactive_prompt_builder --max-questions 3 --output my_prompt.txt
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from medical_guardrails.common.schemas import StructuredQuery
from medical_guardrails.config import Settings
from medical_guardrails.llm.factory import build_llm_client
from medical_guardrails.stage1_slotfill.interactive import (
    DEFAULT_MAX_QUESTIONS,
    InteractiveSlotFillResult,
    run_interactive_slot_fill,
)
from medical_guardrails.stage2_generate.ddinter_lookup import DDInterLookup
from medical_guardrails.stage2_generate.generation import build_generation_messages
from medical_guardrails.stage2_generate.openfda_client import OpenFDAClient
from medical_guardrails.stage2_generate.retrieval import retrieve_evidence
from medical_guardrails.stage2_generate.rxnorm_client import RxNormClient

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "eval" / "results" / "interactive_prompts"

_TRANSCRIPT_LABELS = {
    "query": "USER (initial query)",
    "question": "GUARDRAIL asked",
    "answer": "USER answered",
}


def _terminal_ask(question: str) -> str:
    print(f"\n{question}")
    return input("> ")


def render_prompt_file(
    messages: list[dict[str, str]],
    result: InteractiveSlotFillResult,
    evidence_count: int,
) -> str:
    lines: list[str] = ["=== CONVERSATION ==="]
    for kind, text in result.transcript:
        lines.append(f"[{_TRANSCRIPT_LABELS[kind]}] {text}")

    lines.append("")
    lines.append(f"=== RESOLVED: {result.resolved} (questions asked: {result.questions_asked}) ===")
    lines.append(f"Extracted fields: {result.structured_query.model_dump(exclude={'raw_text'})}")
    lines.append(f"Evidence chunks retrieved for Stage 2: {evidence_count}")
    lines.append("")

    if not result.resolved:
        lines.append(
            "NOTE: question budget was exhausted before every required field was resolved. "
            "The real pipeline would not proceed to Stage 2 in this state -- the prompt below "
            "is shown anyway, for inspection, using whatever was extracted."
        )
        lines.append("")

    lines.append("=== PROMPT THAT WOULD BE SENT TO THE MAIN MODEL (Stage 2) -- NOT SENT ===")
    for message in messages:
        lines.append(f"--- {message['role'].upper()} ---")
        lines.append(message["content"])
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="*", help="Initial query (omit to be prompted for one)")
    parser.add_argument(
        "--max-questions", type=int, default=DEFAULT_MAX_QUESTIONS, help=f"default {DEFAULT_MAX_QUESTIONS}"
    )
    parser.add_argument(
        "--output", default=None, help="Output file path (default: a timestamped file under eval/results/interactive_prompts/)"
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
        print("\n\nInput ended before the guardrail was satisfied -- exiting without saving a prompt.", file=sys.stderr)
        return 1

    status = "Resolved" if result.resolved else "NOT fully resolved"
    print(f"\n--- {status} after {result.questions_asked} question(s) ---")
    print(f"Extracted: {result.structured_query.model_dump(exclude={'raw_text'})}")

    query: StructuredQuery = result.structured_query
    evidence = retrieve_evidence(
        drug_names=query.drug_names,
        rxnorm_client=RxNormClient(settings.rxnorm_base_url, settings.http_timeout_seconds),
        openfda_client=OpenFDAClient(settings.openfda_base_url, settings.http_timeout_seconds),
        ddinter_lookup=DDInterLookup(settings.ddinter_db_path),
    )
    print(f"Retrieved {len(evidence)} evidence chunks for Stage 2")

    messages = build_generation_messages(result.conversation_text, evidence)

    output_path = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"prompt_{datetime.now():%Y%m%d_%H%M%S}.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_prompt_file(messages, result, len(evidence)), encoding="utf-8")

    print(f"\nSaved the prompt that would be sent to Stage 2 -> {output_path}")
    print("Not sent to the model -- this only builds and saves it, per request.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
