"""Cross-checks the active/inactive ingredients pulled from openFDA label
evidence against the user's stated allergies. A match is treated as an
automatic block, not a soft warning, since allergic reactions are often to
excipients/dyes/fillers rather than the active compound -- the whole point
is to catch the case an interaction-only check would miss.

The ingredient list is also returned unconditionally (not just used
internally for pass/fail) so the caller can always render it to the user --
something to verify against allergies the pipeline didn't think to ask
about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from medical_guardrails.common.schemas import EvidenceChunk

INGREDIENT_FIELDS = {"active_ingredient", "inactive_ingredient"}

# openFDA ingredient text is often "Ibuprofen USP, 200 mg (NSAID)" or a
# comma-separated excipient list -- strip trailing parenthetical/dosage
# noise and split on commas to approximate individual ingredient names.
_DOSAGE_OR_PARENTHETICAL = re.compile(r"\(.*?\)|\b\d[\d.]*\s*(mg|mcg|g|ml|%)\b", re.IGNORECASE)


def _split_ingredient_names(text: str) -> list[str]:
    cleaned = _DOSAGE_OR_PARENTHETICAL.sub("", text)
    return [part.strip().lower() for part in cleaned.split(",") if part.strip()]


@dataclass
class IngredientCheckResult:
    ingredients_found: list[str]
    conflicts: list[str]  # e.g. "peanut oil (matches stated allergy: peanut)"


def extract_ingredients(evidence: list[EvidenceChunk]) -> list[str]:
    names: list[str] = []
    for chunk in evidence:
        if chunk.field_name in INGREDIENT_FIELDS:
            names.extend(_split_ingredient_names(chunk.text))
    # de-duplicate while preserving order
    seen: set[str] = set()
    unique = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def check_ingredient_conflicts(evidence: list[EvidenceChunk], allergies: list[str]) -> IngredientCheckResult:
    ingredients = extract_ingredients(evidence)
    conflicts = []
    for allergy in allergies:
        allergy_norm = allergy.strip().lower()
        if not allergy_norm:
            continue
        for ingredient in ingredients:
            if allergy_norm in ingredient or ingredient in allergy_norm:
                conflicts.append(f"{ingredient} (matches stated allergy: {allergy})")
    return IngredientCheckResult(ingredients_found=ingredients, conflicts=conflicts)
