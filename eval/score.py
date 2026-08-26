"""Scoring harness for eval/functional_cases.jsonl against the real
end-to-end pipeline (real Ollama, real RxNorm/openFDA/DDInter). Exact-string
comparison against a closed set of expected_action values -- see
eval/pipeline_adapter.py for how the pipeline's actual output collapses
into that set.

Requires Ollama running locally and network access, same as
tests/integration and eval/run_eval.py's successor here.

Usage: python -m eval.score
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from medical_guardrails.orchestrator import MedicalGuardrailPipeline

from eval.pipeline_adapter import run_pipeline

CASES_PATH = Path(__file__).resolve().parent / "functional_cases.jsonl"
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "functional_run.jsonl"


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(cases: list[dict], pipeline: MedicalGuardrailPipeline) -> list[dict]:
    results = []
    for case in cases:
        actual = run_pipeline(case["input"], pipeline)
        passed = actual["action"] == case["expected_action"]
        results.append(
            {
                **{k: v for k, v in case.items()},
                "actual_action": actual["action"],
                "actual_status": actual["status"],
                "actual_missing_fields": actual["missing_fields"],
                "passed": passed,
            }
        )
    return results


def report(results: list[dict]) -> None:
    total = len(results)
    passed = sum(r["passed"] for r in results)
    by_action: dict[str, list[int]] = {}
    for r in results:
        key = r["expected_action"]
        by_action.setdefault(key, [0, 0])
        by_action[key][1] += 1
        by_action[key][0] += r["passed"]

    print(f"Overall: {passed}/{total} ({passed / total:.1%})")
    for action, (p, t) in sorted(by_action.items()):
        print(f"  {action}: {p}/{t} ({p / t:.1%})")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\nFailures:")
        for r in failures:
            print(f"  {r['id']}: expected {r['expected_action']!r}, got {r['actual_action']!r} -- {r['input'][:80]}")


def save_results(results: list[dict], out_path: Path = RESULTS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            # drop the raw PipelineResult-bearing objects; keep the JSON-safe fields only
            f.write(json.dumps({k: v for k, v in r.items()}) + "\n")


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
    results = score(cases, pipeline)
    report(results)
    save_results(results)
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
