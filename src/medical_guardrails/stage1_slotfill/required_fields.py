"""Generic missing-fields check: given a `DomainSchema`'s required-fields
mapping and a `DomainQuery`, returns which required fields are still
unresolved. Domain-agnostic -- all the domain-specific content (which
fields exist, which query types require which fields) lives in the
`DomainSchema` itself (see domains/medical.py for the shipped example).
"""

from __future__ import annotations

from medical_guardrails.common.schemas import DomainQuery
from medical_guardrails.stage1_slotfill.domain import DomainSchema
from medical_guardrails.stage1_slotfill.domains.medical import MEDICAL_DOMAIN


def missing_fields(query: DomainQuery, domain: DomainSchema = MEDICAL_DOMAIN) -> list[str]:
    required = domain.required_fields[query.query_type]
    missing = []
    for name in required:
        spec = domain.fields[name]
        value = query.fields.get(name)
        is_missing = not value if spec.kind == "list_plain" else value is None
        if is_missing:
            missing.append(name)
    return missing
