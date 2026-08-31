from unittest.mock import MagicMock

from medical_guardrails.main_llm.generation import generate_answer


def test_passes_system_prompt_request_and_context_to_llm():
    llm_client = MagicMock()
    llm_client.chat.return_value = "Yes, that's generally fine for adults."

    reply = generate_answer(
        "Can I take ibuprofen?", {"age_bracket": "adult", "allergies": []}, [], llm_client
    )

    assert reply == "Yes, that's generally fine for adults."
    messages = llm_client.chat.call_args[0][0]
    assert messages[0]["role"] == "system"
    assert "Can I take ibuprofen?" in messages[1]["content"]
    assert "age_bracket: adult" in messages[1]["content"]


def test_empty_and_none_context_values_are_omitted():
    llm_client = MagicMock()
    llm_client.chat.return_value = "ok"

    generate_answer("x", {"allergies": None, "current_medications": [], "age_bracket": "adult"}, [], llm_client)

    user_message = llm_client.chat.call_args[0][0][1]["content"]
    assert "allergies" not in user_message
    assert "current_medications" not in user_message
    assert "age_bracket: adult" in user_message


def test_no_known_context_says_so_explicitly():
    llm_client = MagicMock()
    llm_client.chat.return_value = "ok"

    generate_answer("What is ibuprofen?", {}, [], llm_client)

    user_message = llm_client.chat.call_args[0][0][1]["content"]
    assert "(none provided)" in user_message


def test_unresolved_fields_are_named_in_the_prompt():
    llm_client = MagicMock()
    llm_client.chat.return_value = "ok"

    generate_answer("Can I take ibuprofen?", {}, ["allergies", "age_bracket"], llm_client)

    user_message = llm_client.chat.call_args[0][0][1]["content"]
    assert "allergies" in user_message
    assert "age_bracket" in user_message
    assert "NOT PROVIDED" in user_message
