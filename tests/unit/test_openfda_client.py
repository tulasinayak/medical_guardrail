import httpx
import respx

from medical_guardrails.medical.openfda_client import OpenFDAClient

BASE_URL = "https://api.fda.gov/drug/label.json"


@respx.mock
def test_extracts_ingredient_fields_via_substance_name():
    respx.get(BASE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "active_ingredient": ["Ibuprofen 200mg"],
                        "inactive_ingredient": ["corn starch"],
                        "warnings": ["ignored -- not an ingredient field"],
                    }
                ]
            },
        )
    )
    client = OpenFDAClient(base_url=BASE_URL)
    fields = client.get_ingredient_fields("ibuprofen")
    assert fields["active_ingredient"] == ["Ibuprofen 200mg"]
    assert fields["inactive_ingredient"] == ["corn starch"]
    assert "warnings" not in fields


@respx.mock
def test_falls_back_to_generic_name_when_substance_name_has_no_match():
    route = respx.get(BASE_URL)
    route.side_effect = [
        httpx.Response(404),
        httpx.Response(200, json={"results": [{"active_ingredient": ["Something 100mg"]}]}),
    ]
    client = OpenFDAClient(base_url=BASE_URL)
    fields = client.get_ingredient_fields("somedrug")
    assert fields["active_ingredient"] == ["Something 100mg"]
    assert route.call_count == 2


@respx.mock
def test_returns_empty_dict_when_no_label_matches_either_field():
    respx.get(BASE_URL).mock(return_value=httpx.Response(404))
    client = OpenFDAClient(base_url=BASE_URL)
    assert client.get_ingredient_fields("not-a-real-drug") == {}


@respx.mock
def test_returns_empty_dict_when_results_empty():
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json={"results": []}))
    client = OpenFDAClient(base_url=BASE_URL)
    assert client.get_ingredient_fields("ibuprofen") == {}
