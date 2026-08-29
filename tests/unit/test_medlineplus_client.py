from unittest.mock import MagicMock

import httpx
import respx

from medical_guardrails.stage2_generate.medlineplus_client import MedlinePlusClient, extract_symptom_topic

BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<nlmSearchResult>
  <term>back pain</term>
  <count>1</count>
  <list num="1" start="0" per="1">
    <document rank="0" url="https://medlineplus.gov/backpain.html">
      <content name="title">&lt;span class="qt0"&gt;Back&lt;/span&gt; Pain</content>
      <content name="FullSummary">&lt;p&gt;Rest and over-the-counter pain relievers can help.&lt;/p&gt;</content>
    </document>
  </list>
</nlmSearchResult>"""

EMPTY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<nlmSearchResult>
  <term>gibberish</term>
  <count>0</count>
</nlmSearchResult>"""


@respx.mock
def test_parses_title_and_summary_stripped_of_html():
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, content=SAMPLE_XML.encode()))
    client = MedlinePlusClient(base_url=BASE_URL)
    results = client.search_health_topics("back pain")
    assert results == [
        {
            "title": "Back Pain",
            "summary": "Rest and over-the-counter pain relievers can help.",
            "url": "https://medlineplus.gov/backpain.html",
        }
    ]


@respx.mock
def test_returns_empty_list_when_no_matches():
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, content=EMPTY_XML.encode()))
    client = MedlinePlusClient(base_url=BASE_URL)
    assert client.search_health_topics("gibberish") == []


def test_returns_empty_list_for_empty_term():
    client = MedlinePlusClient(base_url=BASE_URL)
    assert client.search_health_topics("") == []


@respx.mock
def test_returns_empty_list_on_http_error():
    respx.get(BASE_URL).mock(return_value=httpx.Response(500))
    client = MedlinePlusClient(base_url=BASE_URL)
    assert client.search_health_topics("back pain") == []


@respx.mock
def test_returns_empty_list_on_malformed_xml():
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, content=b"not xml"))
    client = MedlinePlusClient(base_url=BASE_URL)
    assert client.search_health_topics("back pain") == []


def test_extract_symptom_topic_strips_and_lowercases():
    llm_client = MagicMock()
    llm_client.chat.return_value = '"Back Pain."'
    assert extract_symptom_topic("my back is paining, what to do?", llm_client) == "back pain"


def test_extract_symptom_topic_returns_empty_string_on_llm_error():
    llm_client = MagicMock()
    llm_client.chat.side_effect = RuntimeError("boom")
    assert extract_symptom_topic("anything", llm_client) == ""
