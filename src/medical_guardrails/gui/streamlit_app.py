"""Streamlit GUI for the medical guardrail pipeline.

Same 3-stage flow as cli/interactive_prompt_builder.py (Stage 1 slot-filling
Q&A -> Stage 2 grounded generation -> Stage 3 claim/ingredient verification),
plus one checkpoint the CLI tool doesn't have: once Stage 1 resolves, the
exact prompt that would be sent to the generation model is shown to the
user, and nothing is sent to that model until the user clicks "Approve &
send to model".

Run with:
    python -m streamlit run src/medical_guardrails/gui/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from medical_guardrails.common.schemas import DomainQuery
from medical_guardrails.config import Settings
from medical_guardrails.llm.factory import build_llm_client
from medical_guardrails.stage1_slotfill.gate import slot_fill_gate
from medical_guardrails.stage2_generate.ddinter_lookup import DDInterLookup
from medical_guardrails.stage2_generate.generation import build_generation_messages, generate_grounded_response
from medical_guardrails.stage2_generate.medlineplus_client import MedlinePlusClient
from medical_guardrails.stage2_generate.openfda_client import OpenFDAClient
from medical_guardrails.stage2_generate.retrieval import retrieve_evidence
from medical_guardrails.stage2_generate.rxnorm_client import RxNormClient
from medical_guardrails.stage3_verify.verification import verify_response

MAX_QUESTIONS = 5
TIMEOUT_HINT = (
    "If this looks like a timeout, set MEDICAL_GUARDRAILS_OLLAMA_TIMEOUT_SECONDS to a higher "
    "value (e.g. 300) in the terminal you launched this app from, then restart it."
)

st.set_page_config(page_title="Medical Guardrail", page_icon="🛡️", layout="centered")


@st.cache_resource
def _settings() -> Settings:
    return Settings()


@st.cache_resource
def _llm_client():
    return build_llm_client(_settings())


@st.cache_resource
def _rxnorm_client() -> RxNormClient:
    s = _settings()
    return RxNormClient(s.rxnorm_base_url, s.http_timeout_seconds)


@st.cache_resource
def _openfda_client() -> OpenFDAClient:
    s = _settings()
    return OpenFDAClient(s.openfda_base_url, s.http_timeout_seconds)


@st.cache_resource
def _ddinter_lookup() -> DDInterLookup:
    return DDInterLookup(_settings().ddinter_db_path)


@st.cache_resource
def _medlineplus_client() -> MedlinePlusClient:
    s = _settings()
    return MedlinePlusClient(s.medlineplus_base_url, s.http_timeout_seconds)


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
    st.session_state.setdefault("evidence", None)
    st.session_state.setdefault("messages", None)
    st.session_state.setdefault("draft_response", None)
    st.session_state.setdefault("verification", None)
    st.session_state.setdefault("error", None)


def _advance_gate() -> None:
    """Call the Stage 1 gate once against the current conversation text.
    On success: either move to the prompt-review checkpoint (resolved, or
    the question budget ran out -- mirroring the CLI tool's documented
    "inspect a partial state" behavior) or ask the next question. On
    failure (e.g. an LLM timeout), record the error and leave the stage
    untouched so the caller can offer a retry.
    """
    try:
        gate_result = slot_fill_gate(st.session_state.conversation_text, _llm_client())
    except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
        st.session_state.error = f"Stage 1 call failed: {exc}\n\n{TIMEOUT_HINT}"
        return

    if gate_result.status == "ready":
        st.session_state.structured_query = gate_result.structured_query
        st.session_state.resolved = True
        st.session_state.stage = "review_prompt"
    elif st.session_state.questions_asked >= MAX_QUESTIONS:
        st.session_state.structured_query = gate_result.structured_query
        st.session_state.resolved = False
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
        "Stage 1 will ask up to 5 clarifying questions, then you'll see the exact prompt "
        "before it's sent to the generation model -- nothing is sent without your approval."
    )
    query = st.chat_input("Your medical question")
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
            f"Stage 1 asked {st.session_state.questions_asked} question(s) but some fields are "
            "still unresolved. You can still review and approve the prompt below, or start over."
        )

    query: DomainQuery = st.session_state.structured_query
    with st.expander("Extracted fields (Stage 1 output)", expanded=False):
        st.json(query.model_dump(exclude={"raw_text"}))

    if st.session_state.evidence is None:
        with st.spinner("Retrieving evidence..."):
            st.session_state.evidence = retrieve_evidence(
                drug_names=query.fields.get("drug_names") or [],
                rxnorm_client=_rxnorm_client(),
                openfda_client=_openfda_client(),
                ddinter_lookup=_ddinter_lookup(),
                symptom_query_text=st.session_state.conversation_text,
                medlineplus_client=_medlineplus_client(),
                llm_client=_llm_client(),
            )
            st.session_state.messages = build_generation_messages(
                st.session_state.conversation_text, st.session_state.evidence
            )

    st.info(f"Retrieved {len(st.session_state.evidence)} evidence chunk(s) for this prompt.")

    st.subheader("Prompt that will be sent to the generation model")
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
        if st.button("Approve & send to model", type="primary"):
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
        with st.spinner("Running Stage 2 (generation)..."):
            st.session_state.draft_response = generate_grounded_response(
                st.session_state.conversation_text, st.session_state.evidence, _llm_client()
            )
        with st.spinner("Running Stage 3 (verification)..."):
            st.session_state.verification = verify_response(
                st.session_state.draft_response,
                st.session_state.evidence,
                query.fields.get("allergies") or [],
                _llm_client(),
            )
        st.session_state.stage = "result"
    except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
        st.session_state.error = f"Stage 2/3 call failed: {exc}\n\n{TIMEOUT_HINT}"
        st.session_state.stage = "review_prompt"
    st.rerun()


def _stage_result() -> None:
    verification = st.session_state.verification

    if verification.action == "block":
        st.error(verification.final_response)
    else:
        st.success(verification.final_response)

    with st.expander("Stage 2 draft (before verification)", expanded=False):
        st.write(st.session_state.draft_response)

    with st.expander("Stage 3 verification detail", expanded=False):
        st.write(f"Action: **{verification.action}**")
        for claim in verification.claims:
            st.write(f"- [{claim.verdict.value if claim.verdict else 'unverified'}] {claim.claim_text}")
        if verification.ingredient_conflicts:
            st.write(f"Ingredient conflicts: {verification.ingredient_conflicts}")

    if st.button("Start a new question"):
        _reset()
        st.rerun()


def main() -> None:
    _init_state()
    st.title("🛡️ Medical Guardrail")

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
