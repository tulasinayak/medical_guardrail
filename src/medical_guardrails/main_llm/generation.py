"""Main LLM: builds one prompt from the original request, whatever context
the Context Guardrail gathered, and which fields (if any) are still
unresolved, then calls the LLM. Domain-agnostic -- nothing here is
medical-specific; the medical demo's ingredient/allergy safety check runs
separately, after this, in medical/ingredient_safety.py.

No retrieval, no evidence block, no claim-level checking. This project
used to ground answers in retrieved evidence and separately verify claims
against that evidence -- that traded a real, provable safety property
(a code-level rule against answering from parametric memory when no
evidence exists) for the scope and complexity of maintaining a full
retrieval + verification pipeline, and that verification stage had a
documented failure mode: it blocked entire answers over single claims
evidence simply didn't happen to mention, treating "unsupported" the same
as "false." This version's safety story is narrower and more honest: ask
enough questions before answering (the Context Guardrail's job), and be
explicit about what's still uncertain (this module's job) -- rather than
claim the answer is "grounded" or "verified" against a source this
project cannot itself vouch for the correctness of.
"""

from __future__ import annotations

from medical_guardrails.llm.base import LLMClient

SYSTEM_PROMPT = """You are a careful assistant. Answer the user's request directly and \
helpfully using your own knowledge.

If the user did not provide some information that would materially change or personalize your \
answer, say so explicitly -- name what's missing and answer the general case as best you can, \
rather than refusing or pretending the gap doesn't exist. Do not state a guess as if it were an \
established fact; flag genuine uncertainty instead."""


def _format_context(context: dict[str, object]) -> str:
    known = {k: v for k, v in context.items() if v not in (None, [], "")}
    if not known:
        return "(none provided)"
    return "\n".join(f"- {key}: {value}" for key, value in known.items())


def build_messages(
    user_request: str, context: dict[str, object], unresolved_fields: list[str]
) -> list[dict[str, str]]:
    """The exact message list sent to Main LLM -- exposed separately from
    generate_answer so callers (e.g. the interactive prompt builder / GUI)
    can inspect/save it before actually invoking the model."""
    lines = [f"USER REQUEST: {user_request}", "", "KNOWN CONTEXT:", _format_context(context)]
    if unresolved_fields:
        lines += [
            "",
            f"NOT PROVIDED (say explicitly you can't personalize on these): {', '.join(unresolved_fields)}",
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def generate_answer(
    user_request: str,
    context: dict[str, object],
    unresolved_fields: list[str],
    llm_client: LLMClient,
) -> str:
    return llm_client.chat(build_messages(user_request, context, unresolved_fields))
