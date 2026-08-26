"""Per-query-type required-fields schema: which StructuredQuery fields must
be resolved (non-None, or non-empty for drug_names) before generation is
allowed to proceed.

Scope note: pregnancy_status/conditional requirements (e.g. only required
for certain age brackets) are deliberately left out of this first pass --
that conditional logic adds real complexity for a case this project isn't
yet equipped to reason about (age extraction is a free-text bracket, not a
structured value), so drug_interaction requires only the fields listed
below. Documented here rather than silently handled.
"""

from __future__ import annotations

from medical_guardrails.common.schemas import QueryType, StructuredQuery

REQUIRED_FIELDS: dict[QueryType, list[str]] = {
    QueryType.DRUG_INTERACTION: ["drug_names", "allergies", "age_bracket"],
    QueryType.DOSAGE: ["drug_names", "age_bracket", "existing_conditions"],
    QueryType.SYMPTOM: [
        "symptom_duration",
        "symptom_severity",
        "existing_conditions",
        "current_medications",
    ],
    QueryType.HOME_REMEDY: ["allergies"],
    QueryType.GENERAL_INFO: [],
}

CLARIFYING_QUESTIONS: dict[str, str] = {
    "drug_names": "Which medication(s) are you asking about?",
    "allergies": "Do you have any known drug allergies? (If none, just say so.)",
    "current_medications": "Are you currently taking any other medications? (If none, just say so.)",
    "age_bracket": "Could you tell me the age range involved (e.g. infant, child, adult, senior)?",
    "existing_conditions": (
        "Do you have any existing medical conditions I should know about, such as kidney or liver "
        "issues? (If none, just say so.)"
    ),
    "symptom_duration": "How long have you had these symptoms?",
    "symptom_severity": "How severe would you say the symptoms are (mild, moderate, or severe)?",
}


def missing_fields(query: StructuredQuery) -> list[str]:
    required = REQUIRED_FIELDS[query.query_type]
    missing = []
    for field in required:
        value = getattr(query, field)
        is_missing = not value if field == "drug_names" else value is None
        if is_missing:
            missing.append(field)
    return missing
