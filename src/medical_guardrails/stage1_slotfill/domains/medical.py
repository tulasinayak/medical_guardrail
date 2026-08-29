"""The medical domain: this project's original (and so far only) concrete
`DomainSchema`. Reconstructs the exact query types, required fields,
clarifying questions, and extraction guidance that used to live directly
in classifier.py/required_fields.py, now expressed as data instead of
hardcoded logic so the mechanism in those two files works for any domain.

Scope note (unchanged from before this refactor): pregnancy_status has no
query type that requires it -- conditional requirements (e.g. only
required for certain age brackets) are deliberately out of scope, so it's
still extracted but never gates on. `fail_closed_query_type` is
`drug_interaction`, the type with the most required fields, so an
unclassifiable query gets *more* scrutiny, not less.
"""

from __future__ import annotations

from medical_guardrails.stage1_slotfill.domain import DomainSchema, FieldSpec

QUERY_TYPES = ["drug_interaction", "dosage", "symptom", "home_remedy", "general_info"]

_FIELDS = {
    "drug_names": FieldSpec(
        "drug_names", "list_plain", "Which medication(s) are you asking about?"
    ),
    "allergies": FieldSpec(
        "allergies",
        "list_with_status",
        "Do you have any known drug allergies? (If none, just say so.)",
    ),
    "current_medications": FieldSpec(
        "current_medications",
        "list_with_status",
        "Are you currently taking any other medications? (If none, just say so.)",
    ),
    "existing_conditions": FieldSpec(
        "existing_conditions",
        "list_with_status",
        "Do you have any existing medical conditions I should know about, such as kidney or "
        "liver issues? (If none, just say so.)",
    ),
    "age_bracket": FieldSpec(
        "age_bracket",
        "scalar",
        "Could you tell me the age range involved (e.g. infant, child, adult, senior)?",
    ),
    "pregnancy_status": FieldSpec(
        "pregnancy_status",
        "scalar",
        "Are you currently pregnant or breastfeeding, if relevant?",
    ),
    "symptom_duration": FieldSpec(
        "symptom_duration", "scalar", "How long have you had these symptoms?"
    ),
    "symptom_severity": FieldSpec(
        "symptom_severity",
        "scalar",
        "How severe would you say the symptoms are (mild, moderate, or severe)?",
    ),
}

_REQUIRED_FIELDS = {
    "drug_interaction": ["drug_names", "allergies", "age_bracket"],
    "dosage": ["drug_names", "age_bracket", "existing_conditions"],
    "symptom": ["symptom_duration", "symptom_severity", "existing_conditions", "current_medications"],
    "home_remedy": ["allergies"],
    "general_info": [],
}

_EXTRACTION_SYSTEM_PROMPT = """You extract structured information from a health-related question, to decide \
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

MEDICAL_DOMAIN = DomainSchema(
    name="medical",
    query_types=QUERY_TYPES,
    fields=_FIELDS,
    required_fields=_REQUIRED_FIELDS,
    fail_closed_query_type="drug_interaction",
    extraction_system_prompt=_EXTRACTION_SYSTEM_PROMPT,
)
