"""Classifies a raw query into a QueryType and extracts whatever
StructuredQuery fields are already present in the text, via the local LLM.

Uses a plain line-based output format (KEY: value) rather than JSON, for
the same reliability reasons as Stage 3's claim decomposition/entailment
modules: small local models produce malformed JSON often enough that a
flat, per-line format is more robust to parse.

For each optional field the model is asked to distinguish "not mentioned in
the query" from "mentioned, and explicitly none" -- see the NONE_MENTIONED
vs NONE_STATED distinction below -- since collapsing that distinction is
exactly the failure mode Stage 1 exists to prevent.

Fails closed: if the TYPE line is missing or unrecognized, defaults to
DRUG_INTERACTION -- the query type with the most required fields -- rather
than GENERAL_INFO, which requires none. An unclassifiable query should
trigger *more* scrutiny, not less.
"""

from __future__ import annotations

import re

from medical_guardrails.common.schemas import QueryType, StructuredQuery
from medical_guardrails.llm.ollama_client import OllamaClient

FAIL_CLOSED_TYPE = QueryType.DRUG_INTERACTION

_LIST_FIELDS = {"drug_names", "allergies", "current_medications", "existing_conditions"}
_SCALAR_FIELDS = {"age_bracket", "pregnancy_status", "symptom_duration", "symptom_severity"}

SYSTEM_PROMPT = f"""You extract structured information from a health-related question, to decide \
what a downstream safety check still needs to ask about.

Output EXACTLY these lines, in this order, with no extra commentary:

TYPE: <one of: drug_interaction, dosage, symptom, home_remedy, general_info>
DRUG_NAMES: <comma-separated drug names mentioned, or NONE_MENTIONED>
ALLERGIES: <comma-separated allergies, or NONE_STATED if the user said they have none, or \
NONE_MENTIONED if allergies were never brought up>
CURRENT_MEDICATIONS: <comma-separated medications, or NONE_STATED, or NONE_MENTIONED>
EXISTING_CONDITIONS: <comma-separated conditions, or NONE_STATED, or NONE_MENTIONED>
AGE_BRACKET: <e.g. infant/child/adult/senior/a specific age, or NONE_MENTIONED>
PREGNANCY_STATUS: <if stated, or NONE_MENTIONED>
SYMPTOM_DURATION: <if stated, or NONE_MENTIONED>
SYMPTOM_SEVERITY: <if stated, or NONE_MENTIONED>

NONE_STATED means the user explicitly said they have none. NONE_MENTIONED means the topic never \
came up at all. Do not guess NONE_STATED unless the user actually said so, and attribute each \
"none"/"no" statement to the SPECIFIC field it refers to -- do not apply it to a different field \
than the one the user actually named.

Example: for "I have no drug allergies, can I take ibuprofen with warfarin?" the correct output \
has ALLERGIES: NONE_STATED (because the user named allergies specifically) and \
CURRENT_MEDICATIONS: NONE_MENTIONED (because medications were never brought up at all -- do NOT \
mark them NONE_STATED just because some other field was negated)."""

_LINE = re.compile(r"^\s*([A-Z_]+)\s*:\s*(.*)$")


def _parse_list_value(value: str) -> list[str] | None:
    value = value.strip()
    if value.upper() == "NONE_MENTIONED":
        return None
    if value.upper() == "NONE_STATED":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_scalar_value(value: str) -> str | None:
    value = value.strip()
    if not value or value.upper() == "NONE_MENTIONED":
        return None
    return value


def extract_structured_query(raw_text: str, llm_client: OllamaClient) -> StructuredQuery:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw_text},
    ]
    raw = llm_client.chat(messages)

    fields: dict[str, str] = {}
    for line in raw.splitlines():
        match = _LINE.match(line)
        if match:
            fields[match.group(1).upper()] = match.group(2)

    type_text = fields.get("TYPE", "").strip().lower()
    try:
        query_type = QueryType(type_text)
    except ValueError:
        query_type = FAIL_CLOSED_TYPE

    drug_names = _parse_list_value(fields.get("DRUG_NAMES", "NONE_MENTIONED")) or []

    return StructuredQuery(
        raw_text=raw_text,
        query_type=query_type,
        drug_names=drug_names,
        allergies=_parse_list_value(fields.get("ALLERGIES", "NONE_MENTIONED")),
        current_medications=_parse_list_value(fields.get("CURRENT_MEDICATIONS", "NONE_MENTIONED")),
        existing_conditions=_parse_list_value(fields.get("EXISTING_CONDITIONS", "NONE_MENTIONED")),
        age_bracket=_parse_scalar_value(fields.get("AGE_BRACKET", "")),
        pregnancy_status=_parse_scalar_value(fields.get("PREGNANCY_STATUS", "")),
        symptom_duration=_parse_scalar_value(fields.get("SYMPTOM_DURATION", "")),
        symptom_severity=_parse_scalar_value(fields.get("SYMPTOM_SEVERITY", "")),
    )
