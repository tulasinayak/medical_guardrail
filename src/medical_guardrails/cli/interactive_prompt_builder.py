"""Interactive front end covering all three guardrail stages combined: type
an initial medical query, answer whatever clarifying questions Stage 1's
gate asks (up to --max-questions, default 5) until it's satisfied or the
budget runs out, then Stage 2 (retrieval + grounded generation) and Stage 3
(claim + ingredient verification) run for real -- not just a saved prompt.
Everything (the conversation, the exact Stage 2 prompt, the draft
response, and the Stage 3 verdict) is saved to one file for inspection.

Stage 2/3 run even if Stage 1's question budget ran out without every
field resolving, so you can see what the rest of the pipeline would have
done from a partial state -- this is a *diagnostic* choice specific to
this tool, not how the real guardrail behaves. The production path
(MedicalGuardrailPipeline.process_query(), used by pipeline_once.py,
eval/score.py, and the spikee target) correctly refuses to proceed past
Stage 1 when required fields are missing; this tool deliberately does not
mirror that restriction, so don't treat a "resolved: False" run here as
representative of what a real user would get.

Pass --no-generate to fall back to the original behavior: build and save
the Stage 2 prompt without actually calling the LLM for generation/
verification.

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
from medical_guardrails.llm.factory import build_llm_client
from medical_guardrails.stage1_slotfill.interactive import (
    DEFAULT_MAX_QUESTIONS,
    InteractiveSlotFillResult,
    run_interactive_slot_fill,
)
from medical_guardrails.stage2_generate.ddinter_lookup import DDInterLookup
from medical_guardrails.stage2_generate.generation import build_generation_messages, generate_grounded_response
from medical_guardrails.stage2_generate.medlineplus_client import MedlinePlusClient
from medical_guardrails.stage2_generate.openfda_client import OpenFDAClient
from medical_guardrails.stage2_generate.retrieval import retrieve_evidence
from medical_guardrails.stage2_generate.rxnorm_client import RxNormClient
from medical_guardrails.stage3_verify.verification import VerificationResult, verify_response

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "eval" / "results" / "interactive_prompts"

_TRANSCRIPT_LABELS = {
    "query": "USER (initial query)",
    "question": "GUARDRAIL asked",
    "answer": "USER answered",
}


def _terminal_ask(question: str) -> str:
    print(f"\n{question}")
    return input("> ")


def render_run_file(
    messages: list[dict[str, str]],
    result: InteractiveSlotFillResult,
    evidence_count: int,
    draft_response: str | None,
    verification: VerificationResult | None,
) -> str:
    lines: list[str] = ["=== STAGE 1: CONVERSATION ==="]
    for kind, text in result.transcript:
        lines.append(f"[{_TRANSCRIPT_LABELS[kind]}] {text}")

    lines.append("")
    lines.append(f"=== STAGE 1: RESOLVED: {result.resolved} (questions asked: {result.questions_asked}) ===")
    lines.append(f"Extracted fields: {result.structured_query.model_dump(exclude={'raw_text'})}")
    lines.append(f"Evidence chunks retrieved for Stage 2: {evidence_count}")
    lines.append("")

    if not result.resolved:
        lines.append(
            "NOTE: question budget was exhausted before every required field was resolved. "
            "The real pipeline (MedicalGuardrailPipeline) would stop here and never reach Stage "
            "2/3 -- this tool continues anyway, for inspection, using whatever was extracted."
        )
        lines.append("")

    lines.append("=== STAGE 2: PROMPT SENT TO THE GENERATION MODEL ===")
    for message in messages:
        lines.append(f"--- {message['role'].upper()} ---")
        lines.append(message["content"])
        lines.append("")

    if draft_response is not None:
        lines.append("=== STAGE 2: DRAFT RESPONSE ===")
        lines.append(draft_response)
        lines.append("")

    if verification is not None:
        lines.append(f"=== STAGE 3: VERIFICATION -- action: {verification.action} ===")
        for claim in verification.claims:
            lines.append(f"  [{claim.verdict.value}] {claim.claim_text}")
        if verification.ingredient_conflicts:
            lines.append(f"  Ingredient conflicts: {verification.ingredient_conflicts}")
        lines.append("")
        lines.append("=== FINAL RESPONSE (after Stage 3) ===")
        lines.append(verification.final_response)
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
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Only build and save the Stage 2 prompt -- do not actually call the LLM for generation/verification.",
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
    print(f"\n--- Stage 1: {status} after {result.questions_asked} question(s) ---")
    print(f"Extracted: {result.structured_query.model_dump(exclude={'raw_text'})}")

    query: DomainQuery = result.structured_query
    evidence = retrieve_evidence(
        drug_names=query.fields.get("drug_names") or [],
        rxnorm_client=RxNormClient(settings.rxnorm_base_url, settings.http_timeout_seconds),
        openfda_client=OpenFDAClient(settings.openfda_base_url, settings.http_timeout_seconds),
        ddinter_lookup=DDInterLookup(settings.ddinter_db_path),
        symptom_query_text=result.conversation_text,
        medlineplus_client=MedlinePlusClient(settings.medlineplus_base_url, settings.http_timeout_seconds),
        llm_client=llm_client,
    )
    print(f"Retrieved {len(evidence)} evidence chunks for Stage 2")

    messages = build_generation_messages(result.conversation_text, evidence)

    draft_response: str | None = None
    verification: VerificationResult | None = None
    if args.no_generate:
        print("\n--no-generate set -- not calling the LLM for Stage 2/3. Saving the prompt only.")
    else:
        print("\nRunning Stage 2 (generation)...")
        draft_response = generate_grounded_response(result.conversation_text, evidence, llm_client)
        print(f"Stage 2 draft:\n{draft_response}")

        print("\nRunning Stage 3 (verification)...")
        verification = verify_response(
            draft_response, evidence, query.fields.get("allergies") or [], llm_client
        )
        print(f"Stage 3 action: {verification.action}")

    output_path = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_run_file(messages, result, len(evidence), draft_response, verification), encoding="utf-8"
    )

    print(f"\nSaved combined 3-stage run -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
