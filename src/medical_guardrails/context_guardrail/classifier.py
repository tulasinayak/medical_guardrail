"""Classifies a raw query into a domain query type and extracts whatever
fields the domain's schema defines, via the local LLM.

Uses the LLM backend's structured-output support (a JSON schema passed as `format`,
translated to whatever the active backend needs -- see llm/factory.py)
rather than a hand-rolled text format: grammar-constrained decoding makes a
malformed/unparseable reply structurally impossible, rather than merely
less likely the way a "please output exactly these lines" prompt is. This
replaced an earlier plain-text line format for that reason.

Caveat: constrained decoding guarantees the *shape* of the output conforms
to the schema, not that the model places each value under the *semantically
correct* key -- it can still misattribute a "no allergies" statement to the
wrong field. The schema helps by giving each `list_with_status` field
(allergies, current medications, existing conditions, in the medical
domain) its own explicit status enum rather than relying on the model to
disambiguate one shared free-text negation, but this needs to be verified
empirically, not assumed fixed.

Fails closed: if the reply isn't valid JSON matching the expected shape (a
model/version that ignores `format`, or a malformed edge case), or names a
query type the domain doesn't recognize, defaults to the domain's
`fail_closed_query_type` (for the medical domain: DRUG_INTERACTION, the
type with the most required fields) rather than one requiring nothing.
Same principle for `answer_scope`: an unparseable reply defaults to
"personal" (more questions asked), never "general" (fewer). An
unclassifiable query should trigger *more* scrutiny, not less.
"""

from __future__ import annotations

import json

from medical_guardrails.common.schemas import DomainQuery
from medical_guardrails.llm.base import LLMClient
from medical_guardrails.context_guardrail.domain import DomainSchema
from medical_guardrails.context_guardrail.domains.medical import MEDICAL_DOMAIN

_STATUS_NOT_MENTIONED = "NOT_MENTIONED"
_STATUS_STATED_NONE = "STATED_NONE"
_STATUS_STATED_PRESENT = "STATED_PRESENT"
_STATUS_VALUES = [_STATUS_NOT_MENTIONED, _STATUS_STATED_NONE, _STATUS_STATED_PRESENT]

_ANSWER_SCOPES = ["general", "personal"]
_FAIL_CLOSED_ANSWER_SCOPE = "personal"  # more scrutiny, not less, on an unparseable reply


def build_response_schema(domain: DomainSchema) -> dict:
    """Builds the JSON schema passed as `format` for this domain. Each
    `list_with_status` field gets its own status enum + a values array,
    rather than a single field that has to encode both "was this
    mentioned" and "what was said" -- a oneOf/union of string-or-array is
    the more compact schema but not all constrained-decoding backends
    support it reliably, and splitting the two concerns also makes the
    model's job simpler: one enum choice, then (if applicable) a plain
    list.

    `answer_scope` is domain-agnostic (every domain gets it, not just
    fields the domain declares) since it's about whether personal context
    is needed at all, not a fact any particular domain extracts."""
    properties: dict = {
        "query_type": {"type": "string", "enum": list(domain.query_types)},
        "answer_scope": {"type": "string", "enum": _ANSWER_SCOPES},
    }
    required = ["query_type", "answer_scope"]

    for name, spec in domain.fields.items():
        if spec.kind == "list_plain":
            properties[name] = {"type": "array", "items": {"type": "string"}}
            required.append(name)
        elif spec.kind == "list_with_status":
            properties[f"{name}_status"] = {"type": "string", "enum": _STATUS_VALUES}
            properties[name] = {"type": "array", "items": {"type": "string"}}
            required.extend([f"{name}_status", name])
        else:  # scalar
            properties[name] = {"type": ["string", "null"]}
            required.append(name)

    return {"type": "object", "properties": properties, "required": required}


def _list_for_status(status: str, values: list[str]) -> list[str] | None:
    if status == _STATUS_STATED_NONE:
        return []
    if status == _STATUS_STATED_PRESENT:
        return [v.strip() for v in values if v.strip()]
    return None


def _extract_fields(data: dict, domain: DomainSchema) -> dict[str, list[str] | str | None]:
    fields: dict[str, list[str] | str | None] = {}
    for name, spec in domain.fields.items():
        if spec.kind == "list_plain":
            fields[name] = [v.strip() for v in data.get(name, []) if v and v.strip()]
        elif spec.kind == "list_with_status":
            fields[name] = _list_for_status(data.get(f"{name}_status", ""), data.get(name, []))
        else:  # scalar
            fields[name] = data.get(name) or None
    return fields


def extract_structured_query(
    raw_text: str, llm_client: LLMClient, domain: DomainSchema = MEDICAL_DOMAIN
) -> DomainQuery:
    messages = [
        {"role": "system", "content": domain.extraction_system_prompt},
        {"role": "user", "content": raw_text},
    ]
    raw = llm_client.chat(messages, format=build_response_schema(domain))

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        data = {}

    query_type = data.get("query_type")
    if query_type not in domain.query_types:
        query_type = domain.fail_closed_query_type

    answer_scope = data.get("answer_scope")
    if answer_scope not in _ANSWER_SCOPES:
        answer_scope = _FAIL_CLOSED_ANSWER_SCOPE

    return DomainQuery(
        raw_text=raw_text,
        query_type=query_type,
        answer_scope=answer_scope,
        fields=_extract_fields(data, domain),
    )
