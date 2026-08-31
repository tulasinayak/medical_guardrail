"""Scores answer *quality* for the same three configurations compared in
baseline_compare.py (raw LLM, LLM + "ask if missing" prompt, Context
Guardrail -> Main LLM), on the same 35-case set -- the piece
baseline_compare.py deliberately left unautomated.

Rubric (0=poor, 1=partial, 2=strong), scored per response by an LLM judge:
  relevance               -- does it address what the user actually asked?
  completeness             -- does it cover what's needed given what was said?
  appropriate_uncertainty  -- does it hedge/flag gaps correctly, without
                              overclaiming or refusing unnecessarily?
  context_use              -- does it use the details the user actually gave
                              (age, allergies, current meds, ...)?
  overall_usefulness       -- everything above, taken together

Ask-case handling: baseline3 sometimes asks a clarifying question instead of
answering (single-shot run here, no follow-up turn). Rather than skip those
cases or invent a fake follow-up, the clarifying question itself is graded
under the same rubric -- it's literally what the user got back in that turn.
The judge prompt tells it explicitly that a well-targeted question can score
well on appropriate_uncertainty/context_use despite scoring low on
completeness, so asking isn't penalized just for not being an answer.

Judge independence: grading uses a *different* model (gpt-4o) than the one
generating the three baselines' replies (gpt-4o-mini, same as
baseline_compare.py/eval.score), to avoid a model preferring its own outputs.
The judge is also never told which baseline produced a given reply (blind
grading) -- only the original question and the reply text.

Usage: python -m eval.quality_judge
"""

from __future__ import annotations

import json
import sys
from statistics import mean

from medical_guardrails.config import Settings
from medical_guardrails.llm.base import LLMClient
from medical_guardrails.llm.factory import build_llm_client
from medical_guardrails.orchestrator import MedicalGuardrailPipeline

from eval.baseline_compare import load_cases, run_baseline1, run_baseline2, run_baseline3

_JUDGE_MODEL = "gpt-4o"

_CRITERIA = [
    "relevance",
    "completeness",
    "appropriate_uncertainty",
    "context_use",
    "overall_usefulness",
]

_JUDGE_SYSTEM_PROMPT = """You are grading a single response to a medical question, using this \
rubric. Score each criterion 0 (poor), 1 (partial), or 2 (strong):

- relevance: does the response address what the user actually asked?
- completeness: does it cover what's needed to be useful, given what the user said?
- appropriate_uncertainty: does it correctly hedge or flag genuine uncertainty/missing \
information, without either overclaiming confidence or refusing unnecessarily? A response that \
asks a well-targeted clarifying question instead of guessing can score well here.
- context_use: does it make use of the details the user actually provided (age, allergies, \
current medications, etc.), or ignore them?
- overall_usefulness: taking all of the above together, how useful is this response to the user \
in this moment?

Some responses are direct answers; others are clarifying questions asked instead of answering. \
Grade a clarifying question on its own terms -- it will typically score low on completeness (it \
doesn't answer yet) but can score well on appropriate_uncertainty and context_use if the question \
is well-targeted at real missing information. Do not penalize a response for asking when asking \
was the right call, and do not reward an answer that guessed past a genuine gap.

Output only the JSON scores."""

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "integer", "enum": [0, 1, 2]} for name in _CRITERIA},
    "required": _CRITERIA,
    "additionalProperties": False,
}


def judge_reply(user_question: str, reply: str, judge_client: LLMClient) -> dict[str, int]:
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"USER QUESTION: {user_question}\n\nRESPONSE TO GRADE: {reply}"},
    ]
    raw = judge_client.chat(messages, format=_JUDGE_SCHEMA)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {name: 0 for name in _CRITERIA}
    return {name: data.get(name, 0) if data.get(name) in (0, 1, 2) else 0 for name in _CRITERIA}


def main() -> int:
    settings = Settings()
    gen_client = build_llm_client(settings)
    if not gen_client.health_check():
        print(f"[!!] Generation LLM backend ({settings.llm_provider}) not reachable", file=sys.stderr)
        return 1

    judge_settings = Settings(openai_model=_JUDGE_MODEL) if settings.llm_provider == "openai" else settings
    judge_client = build_llm_client(judge_settings)
    if judge_client is not gen_client and not judge_client.health_check():
        print(f"[!!] Judge LLM backend not reachable", file=sys.stderr)
        return 1

    pipeline = MedicalGuardrailPipeline(settings=settings, llm_client=gen_client)
    cases = load_cases()

    scores: dict[str, list[dict[str, int]]] = {"baseline1": [], "baseline2": [], "baseline3": []}

    for case in cases:
        replies = {
            "baseline1": run_baseline1(case["input"], gen_client)["reply"],
            "baseline2": run_baseline2(case["input"], gen_client)["reply"],
            "baseline3": run_baseline3(case["input"], pipeline)["reply"],
        }

        print(f"\n=== {case['id']} ===")
        print(f"INPUT: {case['input']}")
        for name, reply in replies.items():
            case_scores = judge_reply(case["input"], reply, judge_client)
            scores[name].append(case_scores)
            print(f"  [{name}] {case_scores}")
            print(f"    {reply[:200]}")

    print(f"\n=== QUALITY SUMMARY (rubric mean, 0-2 scale, judged by {_JUDGE_MODEL}) ===")
    header = "baseline".ljust(12) + "".join(c[:12].ljust(14) for c in _CRITERIA) + "avg"
    print(header)
    for name, case_score_list in scores.items():
        means = {c: mean(s[c] for s in case_score_list) for c in _CRITERIA}
        overall = mean(means.values())
        row = name.ljust(12) + "".join(f"{means[c]:.2f}".ljust(14) for c in _CRITERIA) + f"{overall:.2f}"
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
