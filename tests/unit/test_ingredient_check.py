from medical_guardrails.common.schemas import EvidenceChunk
from medical_guardrails.stage3_verify.ingredient_check import (
    check_ingredient_conflicts,
    extract_ingredients,
)


def _label_evidence(field_name: str, text: str) -> EvidenceChunk:
    return EvidenceChunk(
        source="openfda", authority="regulatory", drug_names=["ibuprofen"], field_name=field_name, text=text
    )


def test_extracts_and_cleans_ingredient_names():
    evidence = [
        _label_evidence("active_ingredient", "Ibuprofen USP, 200 mg (NSAID)"),
        _label_evidence(
            "inactive_ingredient", "colloidal silicon dioxide, corn starch, lactose anhydrous"
        ),
    ]
    ingredients = extract_ingredients(evidence)
    assert "ibuprofen usp" in ingredients  # dosage/parenthetical stripped, pharmacopeia suffix kept
    assert "lactose anhydrous" in ingredients
    assert "corn starch" in ingredients
    assert not any("nsaid" in i for i in ingredients)  # parenthetical stripped
    assert not any("mg" in i for i in ingredients)  # dosage stripped


def test_ignores_non_ingredient_fields():
    evidence = [_label_evidence("warnings", "May cause drowsiness")]
    assert extract_ingredients(evidence) == []


def test_deduplicates_ingredients():
    evidence = [
        _label_evidence("active_ingredient", "Ibuprofen 200 mg"),
        _label_evidence("inactive_ingredient", "ibuprofen, corn starch"),
    ]
    ingredients = extract_ingredients(evidence)
    assert ingredients.count("ibuprofen") == 1


def test_flags_conflict_with_stated_allergy():
    evidence = [_label_evidence("inactive_ingredient", "lactose anhydrous, corn starch")]
    result = check_ingredient_conflicts(evidence, ["lactose"])
    assert len(result.conflicts) == 1
    assert "lactose" in result.conflicts[0]


def test_no_conflict_when_allergy_not_present():
    evidence = [_label_evidence("inactive_ingredient", "corn starch")]
    result = check_ingredient_conflicts(evidence, ["peanut"])
    assert result.conflicts == []


def test_no_conflict_when_no_allergies_stated():
    evidence = [_label_evidence("inactive_ingredient", "corn starch")]
    result = check_ingredient_conflicts(evidence, [])
    assert result.conflicts == []
    assert result.ingredients_found == ["corn starch"]
