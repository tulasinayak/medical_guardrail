from unittest.mock import patch

from medical_guardrails.common.schemas import QueryType, StructuredQuery
from medical_guardrails.stage1_slotfill.gate import GateResult
from medical_guardrails.stage1_slotfill.interactive import run_interactive_slot_fill


def _query(**overrides) -> StructuredQuery:
    base = dict(
        raw_text="x",
        query_type=QueryType.DRUG_INTERACTION,
        drug_names=["ibuprofen", "warfarin"],
        allergies=None,
        age_bracket=None,
    )
    base.update(overrides)
    return StructuredQuery(**base)


@patch("medical_guardrails.stage1_slotfill.interactive.slot_fill_gate")
def test_resolves_after_one_answer(mock_gate):
    mock_gate.side_effect = [
        GateResult(
            "needs_clarification",
            _query(),
            ["allergies", "age_bracket"],
            "Do you have allergies? What's your age?",
        ),
        GateResult("ready", _query(allergies=[], age_bracket="adult"), [], None),
    ]

    result = run_interactive_slot_fill(
        "Can I take ibuprofen with warfarin?",
        llm_client=None,
        ask_fn=lambda q: "No allergies, I'm an adult.",
    )

    assert result.resolved is True
    assert result.questions_asked == 1
    assert result.structured_query.allergies == []
    assert "Can I take ibuprofen with warfarin?" in result.conversation_text
    assert "No allergies, I'm an adult." in result.conversation_text


@patch("medical_guardrails.stage1_slotfill.interactive.slot_fill_gate")
def test_stops_immediately_if_already_ready(mock_gate):
    mock_gate.return_value = GateResult("ready", _query(allergies=[], age_bracket="adult"), [], None)

    result = run_interactive_slot_fill(
        "x", llm_client=None, ask_fn=lambda q: (_ for _ in ()).throw(AssertionError("should not be asked"))
    )

    assert result.resolved is True
    assert result.questions_asked == 0
    assert result.transcript == [("query", "x")]


@patch("medical_guardrails.stage1_slotfill.interactive.slot_fill_gate")
def test_stops_after_max_questions_if_still_unresolved(mock_gate):
    mock_gate.return_value = GateResult(
        "needs_clarification", _query(), ["allergies"], "Do you have allergies?"
    )

    calls = []

    def ask(question: str) -> str:
        calls.append(question)
        return "unhelpful answer"

    result = run_interactive_slot_fill("x", llm_client=None, ask_fn=ask, max_questions=3)

    assert result.resolved is False
    assert result.questions_asked == 3
    assert len(calls) == 3
    assert mock_gate.call_count == 4  # initial check + one re-check per answer


@patch("medical_guardrails.stage1_slotfill.interactive.slot_fill_gate")
def test_transcript_records_query_questions_and_answers_in_order(mock_gate):
    mock_gate.side_effect = [
        GateResult("needs_clarification", _query(), ["allergies"], "Do you have allergies?"),
        GateResult("ready", _query(allergies=[]), [], None),
    ]

    result = run_interactive_slot_fill("initial query", llm_client=None, ask_fn=lambda q: "no allergies")

    assert result.transcript == [
        ("query", "initial query"),
        ("question", "Do you have allergies?"),
        ("answer", "no allergies"),
    ]


@patch("medical_guardrails.stage1_slotfill.interactive.slot_fill_gate")
def test_conversation_text_accumulates_across_multiple_rounds(mock_gate):
    mock_gate.side_effect = [
        GateResult("needs_clarification", _query(), ["allergies"], "Do you have allergies?"),
        GateResult("needs_clarification", _query(allergies=[]), ["age_bracket"], "What's your age?"),
        GateResult("ready", _query(allergies=[], age_bracket="adult"), [], None),
    ]

    answers = iter(["no allergies", "I'm an adult"])
    result = run_interactive_slot_fill("initial query", llm_client=None, ask_fn=lambda q: next(answers))

    assert result.resolved is True
    assert result.questions_asked == 2
    assert result.conversation_text == "initial query\nno allergies\nI'm an adult"
