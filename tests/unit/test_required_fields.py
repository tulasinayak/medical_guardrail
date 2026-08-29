from medical_guardrails.common.schemas import DomainQuery
from medical_guardrails.stage1_slotfill.required_fields import missing_fields


def test_drug_interaction_flags_missing_allergies_and_age():
    query = DomainQuery(
        raw_text="x",
        query_type="drug_interaction",
        fields={"drug_names": ["ibuprofen", "warfarin"], "allergies": None, "age_bracket": None},
    )
    assert set(missing_fields(query)) == {"allergies", "age_bracket"}


def test_drug_interaction_with_explicit_no_allergies_is_not_missing():
    query = DomainQuery(
        raw_text="x",
        query_type="drug_interaction",
        fields={
            "drug_names": ["ibuprofen", "warfarin"],
            "allergies": [],  # explicitly stated none
            "age_bracket": "adult",
        },
    )
    assert missing_fields(query) == []


def test_drug_interaction_missing_drug_names():
    query = DomainQuery(
        raw_text="x",
        query_type="drug_interaction",
        fields={"drug_names": [], "allergies": [], "age_bracket": "adult"},
    )
    assert missing_fields(query) == ["drug_names"]


def test_general_info_never_requires_anything():
    query = DomainQuery(raw_text="x", query_type="general_info", fields={})
    assert missing_fields(query) == []


def test_symptom_requires_duration_severity_conditions_and_medications():
    query = DomainQuery(raw_text="x", query_type="symptom", fields={})
    assert set(missing_fields(query)) == {
        "symptom_duration",
        "symptom_severity",
        "existing_conditions",
        "current_medications",
    }
