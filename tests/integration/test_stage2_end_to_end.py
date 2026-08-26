"""Real network calls to RxNorm/openFDA + a real local Ollama server.
Run with: pytest tests/integration -m integration
"""

import pytest

from medical_guardrails.config import Settings
from medical_guardrails.llm.ollama_client import OllamaClient
from medical_guardrails.stage2_generate.ddinter_lookup import DDInterLookup
from medical_guardrails.stage2_generate.generation import generate_grounded_response
from medical_guardrails.stage2_generate.openfda_client import OpenFDAClient
from medical_guardrails.stage2_generate.retrieval import retrieve_evidence
from medical_guardrails.stage2_generate.rxnorm_client import RxNormClient

pytestmark = pytest.mark.integration


@pytest.fixture
def settings():
    return Settings()


def test_retrieves_real_evidence_for_known_interaction(settings):
    evidence = retrieve_evidence(
        drug_names=["ibuprofen", "warfarin"],
        rxnorm_client=RxNormClient(settings.rxnorm_base_url, settings.http_timeout_seconds),
        openfda_client=OpenFDAClient(settings.openfda_base_url, settings.http_timeout_seconds),
        ddinter_lookup=DDInterLookup(settings.ddinter_db_path),
    )
    assert len(evidence) > 0


def test_full_pipeline_grounds_response_in_evidence(settings):
    llm_client = OllamaClient(
        host=settings.ollama_host,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout_seconds,
    )
    assert llm_client.health_check(), "Ollama must be running locally with the configured model pulled"

    evidence = retrieve_evidence(
        drug_names=["ibuprofen", "warfarin"],
        rxnorm_client=RxNormClient(settings.rxnorm_base_url, settings.http_timeout_seconds),
        openfda_client=OpenFDAClient(settings.openfda_base_url, settings.http_timeout_seconds),
        ddinter_lookup=DDInterLookup(settings.ddinter_db_path),
    )
    reply = generate_grounded_response(
        "Are there any interactions between ibuprofen and warfarin?", evidence, llm_client
    )
    assert reply
