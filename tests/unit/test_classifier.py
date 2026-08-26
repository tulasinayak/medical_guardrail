import json
from unittest.mock import MagicMock

from medical_guardrails.common.schemas import QueryType
from medical_guardrails.stage1_slotfill.classifier import RESPONSE_SCHEMA, extract_structured_query


def _client(response: dict) -> MagicMock:
    client = MagicMock()
    client.chat.return_value = json.dumps(response)
    return client


def _base_response(**overrides) -> dict:
    base = {
        "query_type": "drug_interaction",
        "drug_names": [],
        "allergies_status": "NOT_MENTIONED",
        "allergies": [],
        "current_medications_status": "NOT_MENTIONED",
        "current_medications": [],
        "existing_conditions_status": "NOT_MENTIONED",
        "existing_conditions": [],
        "age_bracket": None,
        "pregnancy_status": None,
        "symptom_duration": None,
        "symptom_severity": None,
    }
    base.update(overrides)
    return base


def test_passes_json_schema_as_format():
    client = _client(_base_response())
    extract_structured_query("x", client)
    assert client.chat.call_args.kwargs["format"] == RESPONSE_SCHEMA


def test_parses_full_response():
    response = _base_response(
        query_type="drug_interaction",
        drug_names=["ibuprofen", "warfarin"],
        allergies_status="STATED_PRESENT",
        allergies=["penicillin"],
        current_medications_status="STATED_NONE",
        age_bracket="adult",
    )
    query = extract_structured_query("Can I take ibuprofen with warfarin?", _client(response))

    assert query.query_type == QueryType.DRUG_INTERACTION
    assert query.drug_names == ["ibuprofen", "warfarin"]
    assert query.allergies == ["penicillin"]
    assert query.current_medications == []  # STATED_NONE -> explicitly empty
    assert query.existing_conditions is None  # NOT_MENTIONED -> not asked
    assert query.age_bracket == "adult"


def test_stated_none_vs_not_mentioned_distinction():
    response = _base_response(
        drug_names=["aspirin"], allergies_status="STATED_NONE", allergies=[]
    )
    query = extract_structured_query("x", _client(response))

    assert query.allergies == []
    assert query.current_medications is None


def test_unparseable_type_fails_closed_to_drug_interaction():
    response = _base_response(query_type="not_a_real_type")
    query = extract_structured_query("x", _client(response))
    assert query.query_type == QueryType.DRUG_INTERACTION


def test_malformed_json_fails_closed_to_drug_interaction():
    client = MagicMock()
    client.chat.return_value = "not valid json at all"
    query = extract_structured_query("x", client)
    assert query.query_type == QueryType.DRUG_INTERACTION
    assert query.drug_names == []


def test_no_drug_names_mentioned_yields_empty_list():
    response = _base_response(query_type="general_info", drug_names=[])
    query = extract_structured_query("x", _client(response))
    assert query.drug_names == []
