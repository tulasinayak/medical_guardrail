"""Streamlit GUI: Context Guardrail Q&A -> Main LLM answer, plus one
checkpoint the CLI tool doesn't have -- once the Context Guardrail is
satisfied (or its question budget runs out), the exact prompt that would
be sent to Main LLM is shown to the user, and nothing is sent until the
user clicks "Approve & send to Main LLM". The medical domain's ingredient/
allergy safety check runs after Main LLM answers, if applicable.

Run with:
    python -m streamlit run src/medical_guardrails/gui/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from medical_guardrails.common.schemas import DomainQuery
from medical_guardrails.config import Settings
from medical_guardrails.context_guardrail.gate import slot_fill_gate
from medical_guardrails.llm.factory import build_llm_client
from medical_guardrails.main_llm.generation import build_messages, generate_answer
from medical_guardrails.medical.ingredient_safety import check_drug_allergy_conflicts
from medical_guardrails.medical.openfda_client import OpenFDAClient

MAX_QUESTIONS = 5
TIMEOUT_HINT = (
    "If this looks like a timeout, set MEDICAL_GUARDRAILS_OLLAMA_TIMEOUT_SECONDS to a higher "
    "value (e.g. 300) in the terminal you launched this app from, then restart it."
)

st.set_page_config(page_title="Context Guardrail", page_icon="🛡️", layout="centered")


@st.cache_resource
def _settings() -> Settings:
    return Settings()


@st.cache_resource
def _llm_client():
    return build_llm_client(_settings())


@st.cache_resource
def _openfda_client() -> OpenFDAClient:
    s = _settings()
    return OpenFDAClient(s.openfda_base_url, s.http_timeout_seconds)


def _reset() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def _init_state() -> None:
    st.session_state.setdefault("stage", "input")
    st.session_state.setdefault("conversation_text", "")
    st.session_state.setdefault("transcript", [])
    st.session_state.setdefault("questions_asked", 0)
    st.session_state.setdefault("current_question", None)
    st.session_state.setdefault("structured_query", None)
    st.session_state.setdefault("resolved", False)
    st.session_state.setdefault("missing", [])
    st.session_state.setdefault("messages", None)
    st.session_state.setdefault("answer", None)
    st.session_state.setdefault("ingredient_check", None)
    st.session_state.setdefault("error", None)


def _advance_gate() -> None:
    """Call the Context Guardrail once against the current conversation
    text. On success: either move to the prompt-review checkpoint
    (resolved, or the question budget ran out -- the answer will
    explicitly say what it couldn't personalize) or ask the next
    question. On failure (e.g. an LLM timeout), record the error and
    leave the stage untouched so the caller can offer a retry.
    """
    try:
        gate_result = slot_fill_gate(st.session_state.conversation_text, _llm_client())
    except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
        st.session_state.error = f"Context Guardrail call failed: {exc}\n\n{TIMEOUT_HINT}"
        return

    if gate_result.status == "ready":
        st.session_state.structured_query = gate_result.structured_query
        st.session_state.resolved = True
        st.session_state.missing = []
        st.session_state.stage = "review_prompt"
    elif st.session_state.questions_asked >= MAX_QUESTIONS:
        st.session_state.structured_query = gate_result.structured_query
        st.session_state.resolved = False
        st.session_state.missing = gate_result.missing
        st.session_state.stage = "review_prompt"
    else:
        st.session_state.current_question = gate_result.clarifying_question
        st.session_state.stage = "asking"


def _render_transcript() -> None:
    for kind, text in st.session_state.transcript:
        role = "assistant" if kind == "question" else "user"
        st.chat_message(role).write(text)


def _stage_input() -> None:
    st.caption(
        "The Context Guardrail will ask up to 5 clarifying questions if the request needs them, "
        "then you'll see the exact prompt before it's sent to Main LLM -- nothing is sent without "
        "your approval."
    )
    query = st.chat_input("Your question")
    if query:
        st.session_state.error = None
        st.session_state.conversation_text = query.strip()
        st.session_state.transcript = [("query", query.strip())]
        st.session_state.questions_asked = 0
        with st.spinner("Checking what's needed..."):
            _advance_gate()
        st.rerun()


def _stage_asking() -> None:
    _render_transcript()
    st.chat_message("assistant").write(st.session_state.current_question)
    answer = st.chat_input("Your answer")
    if answer:
        st.session_state.error = None
        st.session_state.questions_asked += 1
        st.session_state.transcript.append(("question", st.session_state.current_question))
        st.session_state.transcript.append(("answer", answer))
        st.session_state.conversation_text = f"{st.session_state.conversation_text}\n{answer}"
        with st.spinner("Checking what's needed..."):
            _advance_gate()
        st.rerun()
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.error and st.button("Retry last check"):
            st.session_state.error = None
            with st.spinner("Checking what's needed..."):
                _advance_gate()
            st.rerun()
    with col2:
        if st.button("Start over"):
            _reset()
            st.rerun()


def _stage_review_prompt() -> None:
    _render_transcript()

    if not st.session_state.resolved:
        st.warning(
            f"The Context Guardrail asked {st.session_state.questions_asked} question(s) but "
            f"these are still unresolved: {st.session_state.missing}. Main LLM will be told this "
            "explicitly and asked to say what it can't personalize -- or start over instead."
        )

    query: DomainQuery = st.session_state.structured_query
    with st.expander("Extracted fields (Context Guardrail output)", expanded=False):
        st.json(query.model_dump(exclude={"raw_text"}))

    if st.session_state.messages is None:
        st.session_state.messages = build_messages(
            st.session_state.conversation_text, query.fields, st.session_state.missing
        )

    st.subheader("Prompt that will be sent to Main LLM")
    st.caption("Review this before it goes to the model. Nothing is sent until you approve.")
    for message in st.session_state.messages:
        st.markdown(f"**{message['role'].upper()}**")
        st.text_area(
            label=message["role"],
            value=message["content"],
            height=200,
            key=f"prompt_view_{message['role']}",
            disabled=True,
            label_visibility="collapsed",
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve & send to Main LLM", type="primary"):
            st.session_state.error = None
            st.session_state.stage = "generating"
            st.rerun()
    with col2:
        if st.button("Reject / start over"):
            _reset()
            st.rerun()


def _stage_generating() -> None:
    query: DomainQuery = st.session_state.structured_query
    try:
        with st.spinner("Running Main LLM..."):
            st.session_state.answer = generate_answer(
                st.session_state.conversation_text, query.fields, st.session_state.missing, _llm_client()
            )

        drug_names = query.fields.get("drug_names") or []
        allergies = query.fields.get("allergies") or []
        if drug_names and allergies:
            with st.spinner("Running medical ingredient/allergy check..."):
                st.session_state.ingredient_check = check_drug_allergy_conflicts(
                    drug_names, allergies, _openfda_client()
                )
        st.session_state.stage = "result"
    except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
        st.session_state.error = f"Main LLM call failed: {exc}\n\n{TIMEOUT_HINT}"
        st.session_state.stage = "review_prompt"
    st.rerun()


def _stage_result() -> None:
    ingredient_check = st.session_state.ingredient_check

    if ingredient_check is not None and ingredient_check.conflicts:
        st.error(
            "Blocked: this response involves an ingredient matching one of your stated "
            f"allergies. ({'; '.join(ingredient_check.conflicts)})"
        )
        with st.expander("Main LLM's answer (blocked)", expanded=False):
            st.write(st.session_state.answer)
    else:
        st.success(st.session_state.answer)

    if ingredient_check is not None:
        with st.expander("Medical ingredient/allergy check detail", expanded=False):
            st.write(f"Ingredients found: {ingredient_check.ingredients_found}")
            st.write(f"Conflicts: {ingredient_check.conflicts}")

    if st.button("Start a new question"):
        _reset()
        st.rerun()


def main() -> None:
    _init_state()
    st.title("🛡️ Context Guardrail")

    if st.session_state.error:
        st.error(st.session_state.error)

    if not _llm_client().health_check():
        st.error(f"LLM backend ({_settings().llm_provider}) is not reachable. Start it and reload the page.")
        st.stop()

    stage = st.session_state.stage
    if stage == "input":
        _stage_input()
    elif stage == "asking":
        _stage_asking()
    elif stage == "review_prompt":
        _stage_review_prompt()
    elif stage == "generating":
        _stage_generating()
    elif stage == "result":
        _stage_result()


main()
