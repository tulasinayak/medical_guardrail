import httpx
import pytest
import respx

from medical_guardrails.stage2_generate.rxnorm_client import RxNormClient, RxNormNotFoundError

BASE_URL = "https://rxnav.nlm.nih.gov/REST"


@respx.mock
def test_exact_match_returns_rxcui():
    respx.get(f"{BASE_URL}/rxcui.json").mock(
        return_value=httpx.Response(200, json={"idGroup": {"rxnormId": ["5640"]}})
    )
    client = RxNormClient(base_url=BASE_URL)
    assert client.find_rxcui("ibuprofen") == "5640"


@respx.mock
def test_falls_back_to_approximate_match():
    respx.get(f"{BASE_URL}/rxcui.json").mock(
        return_value=httpx.Response(200, json={"idGroup": {}})
    )
    respx.get(f"{BASE_URL}/approximateTerm.json").mock(
        return_value=httpx.Response(
            200, json={"approximateGroup": {"candidate": [{"rxcui": "11289"}]}}
        )
    )
    client = RxNormClient(base_url=BASE_URL)
    assert client.find_rxcui("warfarn") == "11289"


@respx.mock
def test_raises_when_nothing_matches():
    respx.get(f"{BASE_URL}/rxcui.json").mock(
        return_value=httpx.Response(200, json={"idGroup": {}})
    )
    respx.get(f"{BASE_URL}/approximateTerm.json").mock(
        return_value=httpx.Response(200, json={"approximateGroup": {}})
    )
    client = RxNormClient(base_url=BASE_URL)
    with pytest.raises(RxNormNotFoundError):
        client.find_rxcui("not-a-real-drug")
