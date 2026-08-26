"""Real network calls + a real local Ollama server, chaining Stage 2 into
Stage 3. Run with: pytest tests/integration -m integration
"""

import pytest

from medical_guardrails.config import Settings
from medical_guardrails.llm.ollama_client import OllamaClient
from medical_guardrails.stage2_generate.ddinter_lookup import DDInterLookup
from medical_guardrails.stage2_generate.generation import generate_grounded_response
from medical_guardrails.stage2_generate.openfda_client import OpenFDAClient
from medical_guardrails.stage2_generate.retrieval import retrieve_evidence
from medical_guardrails.stage2_generate.rxnorm_client import RxNormClient
from medical_guardrails.stage3_verify.verification import verify_response

pytestmark = pytest.mark.integration


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def llm_client(settings):
    client = OllamaClient(
        host=settings.ollama_host, model=settings.ollama_model, timeout=settings.ollama_timeout_seconds
    )
    assert client.health_check(), "Ollama must be running locally with the configured model pulled"
    return client


def test_ingredient_allergy_conflict_blocks_even_when_claims_are_supported(settings, llm_client):
    evidence = retrieve_evidence(
        drug_names=["ibuprofen", "warfarin"],
        rxnorm_client=RxNormClient(settings.rxnorm_base_url, settings.http_timeout_seconds),
        openfda_client=OpenFDAClient(settings.openfda_base_url, settings.http_timeout_seconds),
        ddinter_lookup=DDInterLookup(settings.ddinter_db_path),
    )
    draft = generate_grounded_response(
        "Are there any interactions between ibuprofen and warfarin?", evidence, llm_client
    )
    result = verify_response(draft, evidence, allergies=["lactose"], llm_client=llm_client)

    assert result.action == "block"
    assert any("lactose" in c for c in result.ingredient_conflicts)


def test_no_conflict_passes_through_with_ingredients_rendered(settings, llm_client):
    evidence = retrieve_evidence(
        drug_names=["ibuprofen", "warfarin"],
        rxnorm_client=RxNormClient(settings.rxnorm_base_url, settings.http_timeout_seconds),
        openfda_client=OpenFDAClient(settings.openfda_base_url, settings.http_timeout_seconds),
        ddinter_lookup=DDInterLookup(settings.ddinter_db_path),
    )
    draft = generate_grounded_response(
        "Are there any interactions between ibuprofen and warfarin?", evidence, llm_client
    )
    result = verify_response(draft, evidence, allergies=["peanuts"], llm_client=llm_client)

    assert result.ingredient_conflicts == []
    assert "Ingredients found" in result.final_response
