from unittest.mock import MagicMock

from medical_guardrails.stage2_generate.retrieval import retrieve_evidence
from medical_guardrails.stage2_generate.rxnorm_client import RxNormNotFoundError


def test_combines_openfda_and_ddinter_evidence():
    rxnorm = MagicMock()
    rxnorm.find_rxcui.side_effect = lambda name: {"ibuprofen": "5640", "warfarin": "11289"}[name]
    rxnorm.get_canonical_name.side_effect = lambda rxcui: {"5640": "ibuprofen", "11289": "warfarin"}[
        rxcui
    ]

    openfda = MagicMock()
    openfda.get_label_fields.side_effect = lambda name: (
        {"contraindications": ["Avoid with anticoagulants."]} if name == "ibuprofen" else {}
    )

    ddinter = MagicMock()
    ddinter.find_interaction.return_value = "Major"

    chunks = retrieve_evidence(["ibuprofen", "warfarin"], rxnorm, openfda, ddinter)

    sources = {chunk.source for chunk in chunks}
    assert sources == {"openfda", "ddinter"}
    assert any(c.field_name == "interaction_severity" and "major" in c.text for c in chunks)
    assert any(c.field_name == "contraindications" for c in chunks)


def test_skips_drugs_that_dont_resolve_in_rxnorm():
    rxnorm = MagicMock()
    rxnorm.find_rxcui.side_effect = RxNormNotFoundError("nope")

    openfda = MagicMock()
    ddinter = MagicMock()
    ddinter.find_interaction.return_value = None

    chunks = retrieve_evidence(["not-a-real-drug"], rxnorm, openfda, ddinter)

    openfda.get_label_fields.assert_not_called()
    assert chunks == []


def test_no_ddinter_chunk_when_pair_not_found():
    rxnorm = MagicMock()
    rxnorm.find_rxcui.return_value = "1"
    openfda = MagicMock()
    openfda.get_label_fields.return_value = {}
    ddinter = MagicMock()
    ddinter.find_interaction.return_value = None

    chunks = retrieve_evidence(["a", "b"], rxnorm, openfda, ddinter)
    assert chunks == []


def test_falls_back_to_medlineplus_when_no_drug_names():
    rxnorm, openfda, ddinter = MagicMock(), MagicMock(), MagicMock()
    llm_client = MagicMock()
    llm_client.chat.return_value = "back pain"
    medlineplus = MagicMock()
    medlineplus.search_health_topics.return_value = [
        {"title": "Back Pain", "summary": "Rest and OTC pain relievers can help.", "url": "https://x"}
    ]

    chunks = retrieve_evidence(
        [],
        rxnorm,
        openfda,
        ddinter,
        symptom_query_text="my back is paining, what to do?",
        medlineplus_client=medlineplus,
        llm_client=llm_client,
    )

    assert len(chunks) == 1
    assert chunks[0].source == "medlineplus"
    assert chunks[0].authority == "regulatory"
    assert "Back Pain" in chunks[0].text
    medlineplus.search_health_topics.assert_called_once_with("back pain")


def test_no_medlineplus_lookup_when_drug_names_present():
    rxnorm = MagicMock()
    rxnorm.find_rxcui.side_effect = RxNormNotFoundError("nope")
    openfda, ddinter = MagicMock(), MagicMock()
    ddinter.find_interaction.return_value = None
    medlineplus = MagicMock()
    llm_client = MagicMock()

    chunks = retrieve_evidence(
        ["not-a-real-drug"],
        rxnorm,
        openfda,
        ddinter,
        symptom_query_text="does not matter",
        medlineplus_client=medlineplus,
        llm_client=llm_client,
    )

    assert chunks == []
    medlineplus.search_health_topics.assert_not_called()
    llm_client.chat.assert_not_called()


def test_no_medlineplus_lookup_when_clients_not_provided():
    rxnorm, openfda, ddinter = MagicMock(), MagicMock(), MagicMock()
    chunks = retrieve_evidence([], rxnorm, openfda, ddinter, symptom_query_text="back pain")
    assert chunks == []
