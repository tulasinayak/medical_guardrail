from unittest.mock import MagicMock

from medical_guardrails.medical.ingredient_safety import check_drug_allergy_conflicts


def _client(label_fields: dict[str, list[str]]) -> MagicMock:
    client = MagicMock()
    client.get_ingredient_fields.return_value = label_fields
    return client


def test_extracts_and_cleans_ingredient_names():
    client = _client(
        {
            "active_ingredient": ["Ibuprofen USP, 200 mg (NSAID)"],
            "inactive_ingredient": ["colloidal silicon dioxide, corn starch, lactose anhydrous"],
        }
    )
    result = check_drug_allergy_conflicts(["ibuprofen"], [], client)
    assert "ibuprofen usp" in result.ingredients_found  # dosage/parenthetical stripped, suffix kept
    assert "lactose anhydrous" in result.ingredients_found
    assert "corn starch" in result.ingredients_found
    assert not any("nsaid" in i for i in result.ingredients_found)  # parenthetical stripped
    assert not any("mg" in i for i in result.ingredients_found)  # dosage stripped


def test_ignores_non_ingredient_fields():
    client = _client({"warnings": ["May cause drowsiness"]})
    result = check_drug_allergy_conflicts(["ibuprofen"], [], client)
    assert result.ingredients_found == []


def test_deduplicates_ingredients_within_and_across_drugs():
    client = _client({"active_ingredient": ["Ibuprofen 200 mg"], "inactive_ingredient": ["ibuprofen, corn starch"]})
    result = check_drug_allergy_conflicts(["ibuprofen", "warfarin"], [], client)
    assert result.ingredients_found.count("ibuprofen") == 1


def test_flags_conflict_with_stated_allergy():
    client = _client({"inactive_ingredient": ["lactose anhydrous, corn starch"]})
    result = check_drug_allergy_conflicts(["ibuprofen"], ["lactose"], client)
    assert len(result.conflicts) == 1
    assert "lactose" in result.conflicts[0]


def test_no_conflict_when_allergy_not_present():
    client = _client({"inactive_ingredient": ["corn starch"]})
    result = check_drug_allergy_conflicts(["ibuprofen"], ["peanut"], client)
    assert result.conflicts == []


def test_no_conflict_when_no_allergies_stated():
    client = _client({"inactive_ingredient": ["corn starch"]})
    result = check_drug_allergy_conflicts(["ibuprofen"], [], client)
    assert result.conflicts == []
    assert result.ingredients_found == ["corn starch"]


def test_calls_client_once_per_drug_name():
    client = _client({})
    check_drug_allergy_conflicts(["ibuprofen", "warfarin"], ["lactose"], client)
    assert client.get_ingredient_fields.call_count == 2
