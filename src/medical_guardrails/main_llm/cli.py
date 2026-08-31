"""Manual test entry point for Main LLM alone (the Context Guardrail isn't
involved, so context is passed directly as key=value pairs rather than
coming from a real DomainQuery).

Usage:
    python -m medical_guardrails.main_llm.cli "Can I take ibuprofen?"
    python -m medical_guardrails.main_llm.cli "Can I take ibuprofen?" --context age=adult allergies=none
    python -m medical_guardrails.main_llm.cli "Can I take ibuprofen?" --unresolved allergies age_bracket
"""

from __future__ import annotations

import argparse

from medical_guardrails.config import Settings
from medical_guardrails.llm.factory import build_llm_client
from medical_guardrails.main_llm.generation import generate_answer


def _parse_context(pairs: list[str]) -> dict[str, str]:
    context = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        context[key] = value
    return context


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("request", help="The user's request")
    parser.add_argument("--context", nargs="*", default=[], help="key=value pairs of known context")
    parser.add_argument("--unresolved", nargs="*", default=[], help="Field names that weren't provided")
    args = parser.parse_args(argv)

    settings = Settings()
    llm_client = build_llm_client(settings)
    reply = generate_answer(args.request, _parse_context(args.context), args.unresolved, llm_client)

    print(reply)


if __name__ == "__main__":
    main()
