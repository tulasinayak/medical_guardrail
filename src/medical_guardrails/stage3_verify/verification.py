"""Ties decomposition, entailment checking, and the ingredient cross-check
into one verification pass over a Stage 2 draft response. Any contradicted
or unsupported claim, or any ingredient/allergy conflict, downgrades the
whole response to a refusal-with-referral rather than letting the
unverified part through silently -- see the module docstrings in
entailment.py and ingredient_check.py for why each check fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from medical_guardrails.common.schemas import Claim, ClaimVerdict, EvidenceChunk
from medical_guardrails.llm.base import LLMClient
from medical_guardrails.stage3_verify.claim_decomposition import decompose_claims
from medical_guardrails.stage3_verify.entailment import verify_claims
from medical_guardrails.stage3_verify.ingredient_check import check_ingredient_conflicts

BLOCKED_CLAIM_MESSAGE = (
    "I can't confirm part of this response against my sources, so I'm not able to provide it as "
    "stated. Please consult a pharmacist or doctor for reliable guidance."
)
BLOCKED_INGREDIENT_MESSAGE = (
    "This response has been blocked: it involves an ingredient that matches one of your stated "
    "allergies. Please consult a pharmacist or doctor before taking this medication."
)

Action = Literal["pass", "block"]


@dataclass
class VerificationResult:
    claims: list[Claim]
    ingredients_found: list[str]
    ingredient_conflicts: list[str]
    action: Action
    final_response: str


def _render_ingredients_section(ingredients: list[str]) -> str:
    if not ingredients:
        return ""
    listed = ", ".join(ingredients)
    return f"\n\n---\nIngredients found in the sources checked: {listed}"


def verify_response(
    draft_response: str,
    evidence: list[EvidenceChunk],
    allergies: list[str],
    llm_client: LLMClient,
) -> VerificationResult:
    if not evidence:
        # Nothing was retrieved, so generate_grounded_response() already
        # returned its fixed not-in-sources fallback without calling the
        # LLM. Decomposing/verifying it here would be pointless at best
        # (there's nothing to check it against) and actively wrong at
        # worst: any claim checked against empty evidence trivially reads
        # as unsupported, which would incorrectly block an honest "I don't
        # know" as if it were an unverified assertion.
        return VerificationResult(
            claims=[], ingredients_found=[], ingredient_conflicts=[], action="pass", final_response=draft_response
        )

    claims = verify_claims(decompose_claims(draft_response, llm_client), evidence, llm_client)
    ingredient_result = check_ingredient_conflicts(evidence, allergies)

    bad_claims = [c for c in claims if c.verdict != ClaimVerdict.SUPPORTED]
    ingredients_section = _render_ingredients_section(ingredient_result.ingredients_found)

    if ingredient_result.conflicts:
        conflict_list = "; ".join(ingredient_result.conflicts)
        final_response = f"{BLOCKED_INGREDIENT_MESSAGE} ({conflict_list}){ingredients_section}"
        action: Action = "block"
    elif bad_claims:
        unverified = "; ".join(c.claim_text for c in bad_claims)
        final_response = f"{BLOCKED_CLAIM_MESSAGE} (Unverified: {unverified}){ingredients_section}"
        action = "block"
    else:
        final_response = f"{draft_response}{ingredients_section}"
        action = "pass"

    return VerificationResult(
        claims=claims,
        ingredients_found=ingredient_result.ingredients_found,
        ingredient_conflicts=ingredient_result.conflicts,
        action=action,
        final_response=final_response,
    )
