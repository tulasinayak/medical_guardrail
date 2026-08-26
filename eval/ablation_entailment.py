"""Ablation: does batching all claims into one entailment call (the current
default, for speed) actually produce different verdicts than checking each
claim in its own call? Small models are more prone to attention dilution
and positional bias with several claims sharing a large evidence block in
one context -- this is cheap to check empirically rather than assume either
way.

Runs live Stage 2 retrieval + generation once to get a real draft response
and evidence, decomposes it into claims, then verifies those same claims
both ways and reports whether the verdicts match, plus wall-clock time for
each approach.

Usage: python -m eval.ablation_entailment
"""

from __future__ import annotations

import sys
import time

from medical_guardrails.config import Settings
from medical_guardrails.llm.factory import build_llm_client
from medical_guardrails.stage2_generate.ddinter_lookup import DDInterLookup
from medical_guardrails.stage2_generate.generation import generate_grounded_response
from medical_guardrails.stage2_generate.openfda_client import OpenFDAClient
from medical_guardrails.stage2_generate.retrieval import retrieve_evidence
from medical_guardrails.stage2_generate.rxnorm_client import RxNormClient
from medical_guardrails.stage3_verify.claim_decomposition import decompose_claims
from medical_guardrails.stage3_verify.entailment import verify_claim_single, verify_claims

QUERY = "Are there any interactions between ibuprofen and warfarin?"
DRUGS = ["ibuprofen", "warfarin"]


def main() -> None:
    settings = Settings()
    llm_client = build_llm_client(settings)
    if not llm_client.health_check():
        print(f"[!!] LLM backend ({settings.llm_provider}) not reachable", file=sys.stderr)
        raise SystemExit(1)

    evidence = retrieve_evidence(
        drug_names=DRUGS,
        rxnorm_client=RxNormClient(settings.rxnorm_base_url, settings.http_timeout_seconds),
        openfda_client=OpenFDAClient(settings.openfda_base_url, settings.http_timeout_seconds),
        ddinter_lookup=DDInterLookup(settings.ddinter_db_path),
    )
    print(f"Retrieved {len(evidence)} evidence chunks", file=sys.stderr)

    draft = generate_grounded_response(QUERY, evidence, llm_client)
    print(f"Draft response:\n{draft}\n", file=sys.stderr)

    claims = decompose_claims(draft, llm_client)
    print(f"Decomposed into {len(claims)} claims: {claims}\n", file=sys.stderr)

    if not claims:
        print("No claims decomposed -- nothing to compare.", file=sys.stderr)
        return

    start = time.monotonic()
    batched = verify_claims(claims, evidence, llm_client)
    batched_seconds = time.monotonic() - start

    start = time.monotonic()
    single = [verify_claim_single(claim, evidence, llm_client) for claim in claims]
    single_seconds = time.monotonic() - start

    print(f"{'Claim':<70} {'Batched':<14} {'Single':<14} Match?")
    all_match = True
    for b, s in zip(batched, single):
        match = b.verdict == s.verdict
        all_match &= match
        print(f"{b.claim_text[:68]:<70} {b.verdict.value:<14} {s.verdict.value:<14} {'yes' if match else 'NO'}")

    print(f"\nBatched: {batched_seconds:.1f}s total. Single-claim: {single_seconds:.1f}s total ({len(claims)} calls).")
    print("All verdicts match." if all_match else "Verdicts DIVERGE -- see table above.")


if __name__ == "__main__":
    main()
