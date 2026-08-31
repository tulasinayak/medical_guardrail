from unittest.mock import MagicMock, patch

from medical_guardrails.common.schemas import DomainQuery
from medical_guardrails.orchestrator import MedicalGuardrailPipeline
from medical_guardrails.context_guardrail.gate import GateResult
from medical_guardrails.medical.ingredient_safety import IngredientCheckResult


def _pipeline():
    return MedicalGuardrailPipeline(settings=MagicMock(), llm_client=MagicMock())


def _query(**field_overrides) -> DomainQuery:
    fields = {"drug_names": ["ibuprofen"], "allergies": None}
    fields.update(field_overrides)
    return DomainQuery(raw_text="x", query_type="drug_interaction", answer_scope="personal", fields=fields)


@patch("medical_guardrails.orchestrator.slot_fill_gate")
def test_stops_at_clarification_without_calling_main_llm(mock_gate):
    query = _query(allergies=None)
    mock_gate.return_value = GateResult(
        "needs_clarification", query, ["allergies"], "Do you have any allergies?"
    )

    pipeline = _pipeline()
    with patch("medical_guardrails.orchestrator.generate_answer") as mock_generate:
        result = pipeline.process_query("can I take ibuprofen")

    assert result.status == "needs_clarification"
    assert result.missing_fields == ["allergies"]
    assert result.clarifying_question == "Do you have any allergies?"
    mock_generate.assert_not_called()


@patch("medical_guardrails.orchestrator.check_drug_allergy_conflicts")
@patch("medical_guardrails.orchestrator.generate_answer")
@patch("medical_guardrails.orchestrator.slot_fill_gate")
def test_ready_query_generates_answer_and_checks_ingredients(mock_gate, mock_generate, mock_check):
    query = _query(drug_names=["ibuprofen"], allergies=["lactose"])
    mock_gate.return_value = GateResult("ready", query, [], None)
    mock_generate.return_value = "draft answer text"
    mock_check.return_value = IngredientCheckResult(ingredients_found=["lactose"], conflicts=[])

    pipeline = _pipeline()
    result = pipeline.process_query(query.raw_text)

    mock_generate.assert_called_once_with(query.raw_text, query.fields, [], pipeline.main_llm_client)
    mock_check.assert_called_once_with(["ibuprofen"], ["lactose"], pipeline.openfda_client)

    assert result.status == "answered"
    assert result.draft_response == "draft answer text"
    assert result.final_response == "draft answer text"


@patch("medical_guardrails.orchestrator.check_drug_allergy_conflicts")
@patch("medical_guardrails.orchestrator.generate_answer")
@patch("medical_guardrails.orchestrator.slot_fill_gate")
def test_ingredient_conflict_blocks_the_answer(mock_gate, mock_generate, mock_check):
    query = _query(drug_names=["ibuprofen"], allergies=["lactose"])
    mock_gate.return_value = GateResult("ready", query, [], None)
    mock_generate.return_value = "draft answer text"
    mock_check.return_value = IngredientCheckResult(
        ingredients_found=["lactose"], conflicts=["lactose (matches stated allergy: lactose)"]
    )

    pipeline = _pipeline()
    result = pipeline.process_query(query.raw_text)

    assert "blocked" in result.final_response.lower()
    assert "lactose" in result.final_response


@patch("medical_guardrails.orchestrator.check_drug_allergy_conflicts")
@patch("medical_guardrails.orchestrator.generate_answer")
@patch("medical_guardrails.orchestrator.slot_fill_gate")
def test_no_ingredient_check_when_no_drug_names(mock_gate, mock_generate, mock_check):
    query = _query(drug_names=[], allergies=["lactose"])
    mock_gate.return_value = GateResult("ready", query, [], None)
    mock_generate.return_value = "draft"

    pipeline = _pipeline()
    pipeline.process_query(query.raw_text)

    mock_check.assert_not_called()


@patch("medical_guardrails.orchestrator.check_drug_allergy_conflicts")
@patch("medical_guardrails.orchestrator.generate_answer")
@patch("medical_guardrails.orchestrator.slot_fill_gate")
def test_no_ingredient_check_when_no_allergies_stated(mock_gate, mock_generate, mock_check):
    query = _query(drug_names=["ibuprofen"], allergies=None)
    mock_gate.return_value = GateResult("ready", query, [], None)
    mock_generate.return_value = "draft"

    pipeline = _pipeline()
    pipeline.process_query(query.raw_text)

    mock_check.assert_not_called()


def test_guardrail_and_main_llm_clients_default_to_the_same_client():
    client = MagicMock()
    pipeline = MedicalGuardrailPipeline(settings=MagicMock(), llm_client=client)
    assert pipeline.guardrail_llm_client is client
    assert pipeline.main_llm_client is client


def test_guardrail_and_main_llm_clients_can_differ():
    guardrail_client, main_client = MagicMock(), MagicMock()
    pipeline = MedicalGuardrailPipeline(
        settings=MagicMock(), guardrail_llm_client=guardrail_client, main_llm_client=main_client
    )
    assert pipeline.guardrail_llm_client is guardrail_client
    assert pipeline.main_llm_client is main_client
