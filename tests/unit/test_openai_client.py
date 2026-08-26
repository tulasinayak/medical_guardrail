import json

import httpx
import respx

from medical_guardrails.llm.openai_client import OpenAIClient

BASE_URL = "https://api.openai.com/v1"


@respx.mock
def test_chat_returns_message_content():
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "Hello there."}}]}
        )
    )
    client = OpenAIClient(api_key="sk-test", base_url=BASE_URL)
    assert client.chat([{"role": "user", "content": "hi"}]) == "Hello there."


@respx.mock
def test_chat_sends_bearer_auth_header():
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    client = OpenAIClient(api_key="sk-test", base_url=BASE_URL)
    client.chat([{"role": "user", "content": "hi"}])
    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-test"


@respx.mock
def test_chat_with_format_wraps_schema_as_strict_json_schema_response_format():
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    )
    client = OpenAIClient(api_key="sk-test", base_url=BASE_URL)
    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    client.chat([{"role": "user", "content": "hi"}], format=schema)

    sent = json.loads(route.calls.last.request.content)
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    assert sent["response_format"]["json_schema"]["schema"]["properties"] == schema["properties"]


@respx.mock
def test_chat_preserves_existing_additional_properties_value():
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    )
    client = OpenAIClient(api_key="sk-test", base_url=BASE_URL)
    schema = {"type": "object", "properties": {}, "additionalProperties": True}
    client.chat([{"role": "user", "content": "hi"}], format=schema)

    sent = json.loads(route.calls.last.request.content)
    assert sent["response_format"]["json_schema"]["schema"]["additionalProperties"] is True


@respx.mock
def test_health_check_true_on_200():
    respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json={}))
    client = OpenAIClient(api_key="sk-test", base_url=BASE_URL)
    assert client.health_check() is True


@respx.mock
def test_health_check_false_on_401():
    respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(401, json={}))
    client = OpenAIClient(api_key="bad-key", base_url=BASE_URL)
    assert client.health_check() is False


@respx.mock
def test_health_check_false_on_network_error():
    respx.get(f"{BASE_URL}/models").mock(side_effect=httpx.ConnectError("refused"))
    client = OpenAIClient(api_key="sk-test", base_url=BASE_URL)
    assert client.health_check() is False
