from medical_guardrails.common.schemas import QueryType, StructuredQuery
from medical_guardrails.stage1_slotfill.required_fields import missing_fields


def test_drug_interaction_flags_missing_allergies_and_age():
    query = StructuredQuery(
        raw_text="x",
        query_type=QueryType.DRUG_INTERACTION,
        drug_names=["ibuprofen", "warfarin"],
        allergies=None,
        age_bracket=None,
    )
    assert set(missing_fields(query)) == {"allergies", "age_bracket"}


def test_drug_interaction_with_explicit_no_allergies_is_not_missing():
    query = StructuredQuery(
        raw_text="x",
        query_type=QueryType.DRUG_INTERACTION,
        drug_names=["ibuprofen", "warfarin"],
        allergies=[],  # explicitly stated none
        age_bracket="adult",
    )
    assert missing_fields(query) == []


def test_drug_interaction_missing_drug_names():
    query = StructuredQuery(
        raw_text="x",
        query_type=QueryType.DRUG_INTERACTION,
        drug_names=[],
        allergies=[],
        age_bracket="adult",
    )
    assert missing_fields(query) == ["drug_names"]


def test_general_info_never_requires_anything():
    query = StructuredQuery(raw_text="x", query_type=QueryType.GENERAL_INFO)
    assert missing_fields(query) == []


def test_symptom_requires_duration_severity_conditions_and_medications():
    query = StructuredQuery(raw_text="x", query_type=QueryType.SYMPTOM)
    assert set(missing_fields(query)) == {
        "symptom_duration",
        "symptom_severity",
        "existing_conditions",
        "current_medications",
    }
