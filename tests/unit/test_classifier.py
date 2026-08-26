from unittest.mock import MagicMock

from medical_guardrails.common.schemas import QueryType
from medical_guardrails.stage1_slotfill.classifier import extract_structured_query


def _client(response: str) -> MagicMock:
    client = MagicMock()
    client.chat.return_value = response
    return client


def test_parses_full_response():
    response = """TYPE: drug_interaction
DRUG_NAMES: ibuprofen, warfarin
ALLERGIES: penicillin
CURRENT_MEDICATIONS: NONE_STATED
EXISTING_CONDITIONS: NONE_MENTIONED
AGE_BRACKET: adult
PREGNANCY_STATUS: NONE_MENTIONED
SYMPTOM_DURATION: NONE_MENTIONED
SYMPTOM_SEVERITY: NONE_MENTIONED"""
    query = extract_structured_query("Can I take ibuprofen with warfarin?", _client(response))

    assert query.query_type == QueryType.DRUG_INTERACTION
    assert query.drug_names == ["ibuprofen", "warfarin"]
    assert query.allergies == ["penicillin"]
    assert query.current_medications == []  # NONE_STATED -> explicitly empty
    assert query.existing_conditions is None  # NONE_MENTIONED -> not asked
    assert query.age_bracket == "adult"


def test_none_stated_vs_none_mentioned_distinction():
    response = """TYPE: drug_interaction
DRUG_NAMES: aspirin
ALLERGIES: NONE_STATED
CURRENT_MEDICATIONS: NONE_MENTIONED
EXISTING_CONDITIONS: NONE_MENTIONED
AGE_BRACKET: NONE_MENTIONED
PREGNANCY_STATUS: NONE_MENTIONED
SYMPTOM_DURATION: NONE_MENTIONED
SYMPTOM_SEVERITY: NONE_MENTIONED"""
    query = extract_structured_query("x", _client(response))

    assert query.allergies == []
    assert query.current_medications is None


def test_unparseable_type_fails_closed_to_drug_interaction():
    response = "TYPE: not_a_real_type\nDRUG_NAMES: NONE_MENTIONED"
    query = extract_structured_query("x", _client(response))
    assert query.query_type == QueryType.DRUG_INTERACTION


def test_missing_type_line_fails_closed_to_drug_interaction():
    query = extract_structured_query("x", _client("garbage response with no structure"))
    assert query.query_type == QueryType.DRUG_INTERACTION


def test_no_drug_names_mentioned_yields_empty_list():
    response = "TYPE: general_info\nDRUG_NAMES: NONE_MENTIONED"
    query = extract_structured_query("x", _client(response))
    assert query.drug_names == []
