"""Classifies a raw query into a QueryType and extracts whatever
StructuredQuery fields are already present in the text, via the local LLM.

Uses Ollama's structured-output support (a JSON schema passed as `format`)
rather than a hand-rolled text format: grammar-constrained decoding makes a
malformed/unparseable reply structurally impossible, rather than merely
less likely the way a "please output exactly these lines" prompt is. This
replaced an earlier plain-text line format for that reason.

Caveat: constrained decoding guarantees the *shape* of the output conforms
to the schema, not that the model places each value under the *semantically
correct* key -- it can still misattribute a "no allergies" statement to the
wrong field. The schema helps by giving each field (allergies, current
medications, existing conditions) its own explicit status enum rather than
relying on the model to disambiguate one shared free-text negation, but
this needs to be verified empirically, not assumed fixed.

Fails closed: if the reply isn't valid JSON matching the expected shape (a
model/version that ignores `format`, or a malformed edge case), defaults to
DRUG_INTERACTION -- the query type with the most required fields -- rather
than GENERAL_INFO, which requires none. An unclassifiable query should
trigger *more* scrutiny, not less.
"""

from __future__ import annotations

import json

from medical_guardrails.common.schemas import QueryType, StructuredQuery
from medical_guardrails.llm.base import LLMClient

FAIL_CLOSED_TYPE = QueryType.DRUG_INTERACTION

_STATUS_NOT_MENTIONED = "NOT_MENTIONED"
_STATUS_STATED_NONE = "STATED_NONE"
_STATUS_STATED_PRESENT = "STATED_PRESENT"
_STATUS_VALUES = [_STATUS_NOT_MENTIONED, _STATUS_STATED_NONE, _STATUS_STATED_PRESENT]

# Each ambiguous field gets its own status enum + a values array, rather than
# a single field that has to encode both "was this mentioned" and "what was
# said" -- a oneOf/union of string-or-array is the more compact schema but
# not all constrained-decoding backends support it reliably, and splitting
# the two concerns also makes the model's job simpler: one enum choice, then
# (if applicable) a plain list.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "query_type": {"type": "string", "enum": [t.value for t in QueryType]},
        "drug_names": {"type": "array", "items": {"type": "string"}},
        "allergies_status": {"type": "string", "enum": _STATUS_VALUES},
        "allergies": {"type": "array", "items": {"type": "string"}},
        "current_medications_status": {"type": "string", "enum": _STATUS_VALUES},
        "current_medications": {"type": "array", "items": {"type": "string"}},
        "existing_conditions_status": {"type": "string", "enum": _STATUS_VALUES},
        "existing_conditions": {"type": "array", "items": {"type": "string"}},
        "age_bracket": {"type": ["string", "null"]},
        "pregnancy_status": {"type": ["string", "null"]},
        "symptom_duration": {"type": ["string", "null"]},
        "symptom_severity": {"type": ["string", "null"]},
    },
    "required": [
        "query_type",
        "drug_names",
        "allergies_status",
        "allergies",
        "current_medications_status",
        "current_medications",
        "existing_conditions_status",
        "existing_conditions",
        "age_bracket",
        "pregnancy_status",
        "symptom_duration",
        "symptom_severity",
    ],
}

SYSTEM_PROMPT = """You extract structured information from a health-related question, to decide \
what a downstream safety check still needs to ask about.

For allergies, current medications, and existing conditions: set the matching "_status" field to
STATED_PRESENT and list the values if the user named any; to STATED_NONE if the user explicitly
said they have none of that specific thing; or to NOT_MENTIONED if that topic never came up at
all. Attribute each "none"/"no" statement to the SPECIFIC field it refers to -- do not apply it
to a different field than the one the user actually named. For example, "I have no drug
allergies" sets allergies_status to STATED_NONE, and current_medications_status stays
NOT_MENTIONED since medications were never brought up.

For age_bracket, pregnancy_status, symptom_duration, and symptom_severity: use the stated value,
or null if not mentioned."""


def _list_for_status(status: str, values: list[str]) -> list[str] | None:
    if status == _STATUS_STATED_NONE:
        return []
    if status == _STATUS_STATED_PRESENT:
        return [v.strip() for v in values if v.strip()]
    return None


def extract_structured_query(raw_text: str, llm_client: LLMClient) -> StructuredQuery:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw_text},
    ]
    raw = llm_client.chat(messages, format=RESPONSE_SCHEMA)

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        data = {}

    try:
        query_type = QueryType(data.get("query_type"))
    except ValueError:
        query_type = FAIL_CLOSED_TYPE

    return StructuredQuery(
        raw_text=raw_text,
        query_type=query_type,
        drug_names=[d.strip() for d in data.get("drug_names", []) if d and d.strip()],
        allergies=_list_for_status(data.get("allergies_status", ""), data.get("allergies", [])),
        current_medications=_list_for_status(
            data.get("current_medications_status", ""), data.get("current_medications", [])
        ),
        existing_conditions=_list_for_status(
            data.get("existing_conditions_status", ""), data.get("existing_conditions", [])
        ),
        age_bracket=data.get("age_bracket") or None,
        pregnancy_status=data.get("pregnancy_status") or None,
        symptom_duration=data.get("symptom_duration") or None,
        symptom_severity=data.get("symptom_severity") or None,
    )
