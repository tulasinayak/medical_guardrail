"""Deterministic allergy/ingredient safety check for the medical demo --
the one place this project still calls an external medical data source
after removing retrieval/RAG from the core architecture. This isn't
grounding the answer (Main LLM does that from its own knowledge); it's a
plain, verifiable lookup against a public label database, not a matter of
LLM judgment, so it stays a hard check rather than something the model is
merely asked to consider.

A match is treated as an automatic block, not a soft warning, since
allergic reactions are often to excipients/dyes/fillers rather than the
active compound -- the whole point is to catch something the model
answering "is X safe with Y" from memory would have no reason to check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from medical_guardrails.medical.openfda_client import INGREDIENT_FIELDS, OpenFDAClient

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


def _ingredients_for_drug(drug_name: str, openfda_client: OpenFDAClient) -> list[str]:
    label_fields = openfda_client.get_ingredient_fields(drug_name)
    names: list[str] = []
    for field_name in INGREDIENT_FIELDS:
        for value in label_fields.get(field_name, []):
            names.extend(_split_ingredient_names(value))
    return names


def check_drug_allergy_conflicts(
    drug_names: list[str], allergies: list[str], openfda_client: OpenFDAClient
) -> IngredientCheckResult:
    ingredients: list[str] = []
    seen: set[str] = set()
    for drug_name in drug_names:
        for name in _ingredients_for_drug(drug_name, openfda_client):
            if name not in seen:
                seen.add(name)
                ingredients.append(name)

    conflicts = []
    for allergy in allergies:
        allergy_norm = allergy.strip().lower()
        if not allergy_norm:
            continue
        for ingredient in ingredients:
            if allergy_norm in ingredient or ingredient in allergy_norm:
                conflicts.append(f"{ingredient} (matches stated allergy: {allergy})")
    return IngredientCheckResult(ingredients_found=ingredients, conflicts=conflicts)
