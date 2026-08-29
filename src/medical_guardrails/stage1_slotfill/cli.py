"""Manual test entry point for Stage 1 alone.

Usage:
    python -m medical_guardrails.stage1_slotfill.cli "Can I take ibuprofen with warfarin?"
"""

from __future__ import annotations

import argparse

from medical_guardrails.config import Settings
from medical_guardrails.llm.factory import build_llm_client
from medical_guardrails.stage1_slotfill.gate import slot_fill_gate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="The raw user query to run through the slot-filling gate")
    args = parser.parse_args(argv)

    settings = Settings()
    llm_client = build_llm_client(settings)
    result = slot_fill_gate(args.query, llm_client)

    print(f"Query type: {result.structured_query.query_type}")
    print(f"Extracted: {result.structured_query.model_dump(exclude={'raw_text'})}")
    print(f"Status: {result.status}")
    if result.status == "needs_clarification":
        print(f"Missing: {result.missing}")
        print(f"Clarifying question: {result.clarifying_question}")


if __name__ == "__main__":
    main()
