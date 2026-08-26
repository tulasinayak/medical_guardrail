"""Manual test entry point for Stage 2 alone (Stage 1 doesn't exist yet, so
this takes drug names directly rather than a full StructuredQuery).

Usage:
    python -m medical_guardrails.stage2_generate.cli ibuprofen warfarin
    python -m medical_guardrails.stage2_generate.cli --question "Can I take this with food?" aspirin lisinopril
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drugs", nargs="+", help="Drug names to look up and check for interactions")
    parser.add_argument(
        "--question",
        default=None,
        help="Question to ask about these drugs (default: are there interactions between them?)",
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
    for chunk in evidence:
        print(f"  [{chunk.source}] {chunk.field_name} ({', '.join(chunk.drug_names)})", file=sys.stderr)

    llm_client = OllamaClient(
        host=settings.ollama_host,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout_seconds,
    )
    reply = generate_grounded_response(question, evidence, llm_client)

    print("\n--- Response ---")
    print(reply)


if __name__ == "__main__":
    main()
