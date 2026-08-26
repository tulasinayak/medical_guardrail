"""Manual test entry point: runs Stage 2 (retrieval + grounded generation)
and then Stage 3 (claim verification + ingredient check) end to end.
Stage 1 doesn't exist yet, so allergies are passed directly via --allergy.

Usage:
    python -m medical_guardrails.stage3_verify.cli ibuprofen warfarin
    python -m medical_guardrails.stage3_verify.cli ibuprofen --allergy aspirin
"""

from __future__ import annotations

import argparse
import sys

from medical_guardrails.config import Settings
from medical_guardrails.llm.ollama_client import OllamaClient
from medical_guardrails.stage2_generate.ddinter_lookup import DDInterLookup
from medical_guardrails.stage2_generate.generation import generate_grounded_response
from medical_guardrails.stage2_generate.openfda_client import OpenFDAClient
from medical_guardrails.stage2_generate.retrieval import retrieve_evidence
from medical_guardrails.stage2_generate.rxnorm_client import RxNormClient
from medical_guardrails.stage3_verify.verification import verify_response


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drugs", nargs="+", help="Drug names to look up and check for interactions")
    parser.add_argument("--question", default=None, help="Question to ask about these drugs")
    parser.add_argument(
        "--allergy", action="append", default=[], help="A known allergy (repeatable)"
    )
    args = parser.parse_args(argv)

    settings = Settings()
    question = args.question or f"Are there any interactions between {', '.join(args.drugs)}?"

    evidence = retrieve_evidence(
        drug_names=args.drugs,
        rxnorm_client=RxNormClient(settings.rxnorm_base_url, settings.http_timeout_seconds),
        openfda_client=OpenFDAClient(settings.openfda_base_url, settings.http_timeout_seconds),
        ddinter_lookup=DDInterLookup(settings.ddinter_db_path),
    )
    print(f"--- Retrieved {len(evidence)} evidence chunks ---", file=sys.stderr)

    llm_client = OllamaClient(
        host=settings.ollama_host,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout_seconds,
    )
    draft = generate_grounded_response(question, evidence, llm_client)
    print(f"\n--- Draft response ---\n{draft}", file=sys.stderr)

    result = verify_response(draft, evidence, args.allergy, llm_client)

    print("\n--- Claims ---", file=sys.stderr)
    for claim in result.claims:
        print(f"  [{claim.verdict.value}] {claim.claim_text}", file=sys.stderr)

    print(f"\n--- Action: {result.action} ---")
    print(result.final_response)


if __name__ == "__main__":
    main()
