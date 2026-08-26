"""Runs the hand-labeled regression cases in eval/cases.jsonl against the
real end-to-end pipeline (real Ollama, real RxNorm/openFDA/DDInter) and
reports pass/fail per case.

This is the seed of the hand-labeled eval set from the project's design
doc -- deliberately simple (no scoring/statistics), since its purpose right
now is to freeze specific bugs this system actually produced as permanent
regression checks, not to compute aggregate accuracy metrics.

Requires Ollama running locally and network access -- same requirements as
tests/integration, and for the same reason not run as part of `pytest
tests/unit`.

Usage: python -m eval.run_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from medical_guardrails.orchestrator import MedicalGuardrailPipeline, PipelineResult

CASES_PATH = Path(__file__).resolve().parent / "cases.jsonl"


def _check_status(result: PipelineResult, value: Any) -> str | None:
    return None if result.status == value else f"expected status={value!r}, got {result.status!r}"


def _check_min_evidence_chunks(result: PipelineResult, value: Any) -> str | None:
    return None if len(result.evidence) >= value else f"expected >={value} evidence chunks, got {len(result.evidence)}"


def _check_evidence_source_present(result: PipelineResult, value: Any) -> str | None:
    sources = {chunk.source for chunk in result.evidence}
    return None if value in sources else f"expected an evidence chunk with source={value!r}, got sources={sources}"


def _check_allergies_extracted(result: PipelineResult, value: Any) -> str | None:
    actual = result.structured_query.allergies
    return None if actual == value else f"expected allergies={value!r}, got {actual!r}"


def _check_verification_action(result: PipelineResult, value: Any) -> str | None:
    if result.verification is None:
        return f"expected verification.action={value!r}, but no verification ran (status={result.status})"
    actual = result.verification.action
    return None if actual == value else f"expected verification.action={value!r}, got {actual!r}"


def _check_ingredient_conflict_substring(result: PipelineResult, value: Any) -> str | None:
    if result.verification is None:
        return f"expected an ingredient conflict containing {value!r}, but no verification ran"
    conflicts = result.verification.ingredient_conflicts
    if any(value in c for c in conflicts):
        return None
    return f"expected an ingredient conflict containing {value!r}, got {conflicts}"


def _check_missing_fields_include(result: PipelineResult, value: Any) -> str | None:
    missing = set(result.missing_fields)
    wanted = set(value)
    if wanted <= missing:
        return None
    return f"expected missing_fields to include {sorted(wanted)}, got {sorted(missing)}"


_CHECKS: dict[str, Callable[[PipelineResult, Any], str | None]] = {
    "status": _check_status,
    "min_evidence_chunks": _check_min_evidence_chunks,
    "evidence_source_present": _check_evidence_source_present,
    "allergies_extracted": _check_allergies_extracted,
    "verification_action": _check_verification_action,
    "ingredient_conflict_substring": _check_ingredient_conflict_substring,
    "missing_fields_include": _check_missing_fields_include,
}


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_case(case: dict, pipeline: MedicalGuardrailPipeline) -> list[str]:
    result = pipeline.process_query(case["query"])
    failures = []
    for key, value in case["expected"].items():
        check = _CHECKS.get(key)
        if check is None:
            failures.append(f"unknown expectation key: {key!r}")
            continue
        failure = check(result, value)
        if failure:
            failures.append(failure)
    return failures


def main() -> int:
    pipeline = MedicalGuardrailPipeline()
    if not pipeline.llm_client.health_check():
        print(
            f"[!!] Ollama not reachable at {pipeline.settings.ollama_host} "
            f"with model {pipeline.settings.ollama_model}",
            file=sys.stderr,
        )
        return 1

    cases = load_cases()
    passed = 0
    for case in cases:
        print(f"--- {case['id']} ---")
        failures = run_case(case, pipeline)
        if failures:
            print("FAIL")
            for failure in failures:
                print(f"  - {failure}")
        else:
            print("PASS")
            passed += 1

    print(f"\n{passed}/{len(cases)} cases passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
