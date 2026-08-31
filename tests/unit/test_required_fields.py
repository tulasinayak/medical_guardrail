from medical_guardrails.common.schemas import DomainQuery
from medical_guardrails.context_guardrail.required_fields import missing_fields


def test_drug_interaction_flags_missing_allergies_and_age():
    query = DomainQuery(
        raw_text="x",
        query_type="drug_interaction",
        answer_scope="personal",
        fields={"drug_names": ["ibuprofen", "warfarin"], "allergies": None, "age_bracket": None},
    )
    assert set(missing_fields(query)) == {"allergies", "age_bracket"}


def test_drug_interaction_with_explicit_no_allergies_is_not_missing():
    query = DomainQuery(
        raw_text="x",
        query_type="drug_interaction",
        answer_scope="personal",
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
        answer_scope="personal",
        fields={"drug_names": [], "allergies": [], "age_bracket": "adult"},
    )
    assert missing_fields(query) == ["drug_names"]


def test_general_info_never_requires_anything():
    query = DomainQuery(raw_text="x", query_type="general_info", answer_scope="personal", fields={})
    assert missing_fields(query) == []


def test_symptom_requires_duration_severity_conditions_and_medications():
    query = DomainQuery(raw_text="x", query_type="symptom", answer_scope="personal", fields={})
    assert set(missing_fields(query)) == {
        "symptom_duration",
        "symptom_severity",
        "existing_conditions",
        "current_medications",
    }


def test_general_scope_never_requires_anything_regardless_of_type():
    query = DomainQuery(
        raw_text="x",
        query_type="drug_interaction",
        answer_scope="general",
        fields={"drug_names": ["ibuprofen"], "allergies": None, "age_bracket": None},
    )
    assert missing_fields(query) == []
