# Medical Guardrails

A staged guardrail that wraps an LLM for health-related queries — the sibling
project to [`pii_guardrails`](../pii_guardrails), same "interception layer
between prompt and model" architecture, different check.

```
User prompt
    │
    ▼
[Stage 1: pre-generation slot-filling]
    │  classify query type, require key fields (allergies, meds, age),
    │  ask a clarifying question instead of generating if any are missing
    ▼
[Stage 2: grounded generation]
    │  retrieve evidence (RxNorm identity resolution, openFDA label text,
    │  local DDInter interaction severities) and generate constrained to
    │  ONLY that retrieved evidence
    ▼
[Stage 3: post-generation claim verification]
    │  decompose the draft response into atomic claims, check each against
    │  the retrieved evidence, and cross-check openFDA active/inactive
    │  ingredients against stated allergies -- block/downgrade anything
    │  unsupported, contradicted, or allergy-conflicting
    ▼
Final response
```

All three stages are wired into one end-to-end pipeline
(`orchestrator.MedicalGuardrailPipeline`, mirroring `pii_guardrails`'
`orchestrator.GuardrailedChat`), and each stage also has its own standalone
CLI for testing it in isolation.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate                    # Windows
pip install -e ".[dev]"

# Build the local DDInter interaction database (one-time; downloads DDInter's
# public per-letter CSV export)
python data/ddinter/build_ddinter_db.py

# All manual tests below require Ollama running locally with the configured
# model pulled (default mistral:latest). Prompt eval on a CPU-only Ollama
# instance is slow for anything with a large evidence block -- see Known
# limitations below -- so pass a longer timeout via
# MEDICAL_GUARDRAILS_OLLAMA_TIMEOUT_SECONDS if calls are timing out.

# Full pipeline: Stage 1 extracts drug names/allergies/age from the raw query itself --
# no separate flags needed. Missing required fields short-circuit to a clarifying
# question before anything is generated.
python -m medical_guardrails.cli.pipeline_once "Is it safe to take ibuprofen with warfarin?"
python -m medical_guardrails.cli.pipeline_once "I have a lactose allergy. Is it safe for an adult to take ibuprofen with warfarin?"

# Individual stages, for testing one in isolation:
python -m medical_guardrails.stage1_slotfill.cli "Can I take ibuprofen with warfarin?"
python -m medical_guardrails.stage2_generate.cli ibuprofen warfarin
python -m medical_guardrails.stage3_verify.cli ibuprofen warfarin --allergy lactose

# Tests
pytest tests/unit                          # fast, fully mocked, no network/Ollama needed
pytest tests -m "not integration"          # same as above
pytest tests -m integration                # real RxNorm/openFDA calls + real Ollama

# Regression eval set (real Ollama + network; see Known limitations -- currently 2/4 pass)
python -m eval.run_eval

# Batched vs single-claim entailment ablation (real Ollama + network)
python -m eval.ablation_entailment
```

## Architecture

- `src/medical_guardrails/orchestrator.py` — `MedicalGuardrailPipeline.process_query()`: runs Stage 1's gate first and returns immediately with a clarifying question if anything required is missing; otherwise runs Stage 2 retrieval + generation, then Stage 3 verification, using Stage 1's extracted `StructuredQuery` as the actual input (its `drug_names` feed retrieval, its `allergies` feed the ingredient check) rather than passing those in separately.
- `src/medical_guardrails/cli/pipeline_once.py` — the full end-to-end CLI, taking one raw natural-language query and nothing else.
- `src/medical_guardrails/common/schemas.py` — shared Pydantic models passed between stages: `StructuredQuery` (Stage 1's output), `EvidenceChunk` (Stage 2's output), `Claim` (Stage 3's output).
- `src/medical_guardrails/stage1_slotfill/` — `classifier.py` (LLM-based query-type classification + field extraction, using Ollama's structured-output `format` parameter -- a JSON schema grammar-constrains decoding so the reply structurally cannot deviate from it), `required_fields.py` (per-query-type required-fields table + clarifying questions), `gate.py` (`slot_fill_gate()`: ties the two together into a ready/needs_clarification decision).
- `src/medical_guardrails/stage2_generate/` — `rxnorm_client.py` (name → RxCUI, plus canonical-name lookup), `openfda_client.py` (label text by name: contraindications/warnings/interactions/ingredients), `ddinter_lookup.py` (local offline pairwise interaction severity), `retrieval.py` (combines all three into one evidence list, tagging each chunk's `authority` as `regulatory` (openFDA) or `curated_secondary` (DDInter)), `generation.py` (grounded generation via Ollama, with a system prompt that forbids answering outside the retrieved evidence).
- `src/medical_guardrails/stage3_verify/` — `claim_decomposition.py` (splits a draft response into atomic claims via LLM), `entailment.py` (LLM-as-judge verdict per claim: supported/contradicted/unsupported, fails closed to unsupported on any parse failure; `verify_claims()` batches all claims into one call, `verify_claim_single()` checks one at a time for the ablation in `eval/ablation_entailment.py`), `ingredient_check.py` (extracts active/inactive ingredients from openFDA evidence and cross-checks against stated allergies -- a match is an automatic block), `verification.py` (`verify_response()`: ties all three into a pass/block decision + final response text, always rendering the ingredients list to the user).
- `data/ddinter/build_ddinter_db.py` — one-time script to build the local DDInter SQLite dump from DDInter's public bulk CSV export (not committed; regenerate locally).
- `eval/cases.jsonl` + `eval/run_eval.py` — a hand-labeled regression suite (see Known limitations for current pass rate) seeded from bugs the system actually produced, not hypothetical cases. `eval/ablation_entailment.py` — batched-vs-single-claim entailment comparison tool.

## Known limitations

- **Stage 1 extraction accuracy is the biggest open gap, and it's not just a prompt-format problem.** Switching from a hand-rolled line-based text format to Ollama's structured-output `format` parameter (a JSON-schema-constrained decode) completely eliminated malformed/unparseable output -- every response is now valid JSON matching the schema, with no parsing failures observed. It did **not** reliably fix the deeper semantic problem: live re-testing the same "no allergies" queries that failed before showed one now correct and one still wrong, with clean JSON either way. Constrained decoding guarantees the *shape* of the output, not that the model places each value under the *semantically correct* key. `eval/run_eval.py` currently passes 2/4 seeded cases against `mistral:latest`; the two failures are real: one missed an explicitly-stated age bracket entirely, and the other misclassified a plain drug-interaction question as `SYMPTOM` -- which matters more than a field-attribution slip, since `required_fields.py` keys its whole required-fields table off `query_type`, so a wrong type means the gate can demand the wrong fields outright rather than just ask one redundant question.
- **A same-size, newer model (Qwen3-8B) doesn't cleanly solve this either.** Tested head-to-head against the same failing queries: Qwen3-8B correctly recognized "no allergies" in 3 of 4 relevant cases (better than mistral's roughly 1-of-2), but it hit the *exact same* `SYMPTOM` misclassification on the lactose-allergy query, missed an explicit "I'm an adult" age statement in one case, and introduced a new failure mode mistral didn't have: it fabricated `warfarin` as a "current medication" in 3 of 4 runs, when warfarin was the drug being asked about, not something the patient said they were already taking. It was also markedly slower per call (roughly 100-225s vs. mistral's tighter range) -- likely Qwen3's hidden "thinking" tokens, which don't appear in the final JSON content but plausibly explain the latency given how short the actual output is. Net finding: the model swap traded one accuracy problem for a different one rather than raising the ceiling outright, at a real speed cost -- a genuinely larger model (or fine-tuning for this exact extraction task) looks more promising than lateral model swaps at this size class.
- **Batched vs. single-claim entailment**: ran the ablation live (`eval/ablation_entailment.py`) across two cases (1 claim, then 2 claims) -- verdicts matched between batched and single-claim verification both times, no divergence observed. Sample size is small (3 claims total), so this doesn't rule out attention dilution on responses with more claims, but it doesn't support switching off the batched default either; kept batched for speed pending a larger sample.
- **Evidence authority is tagged but not yet used for anything beyond display.** `EvidenceChunk.authority` (`regulatory` for openFDA, `curated_secondary` for DDInter) is surfaced in every evidence block shown to both the generation and verification prompts, but no weighting/scoring logic reads it yet -- that's intentionally left for a future iteration.
- **Stage 2/DDInter**: DDInter's bulk CSV export only carries interaction severity (Major/Moderate/Minor), not the mechanism/management text DDInter shows on its per-pair detail pages. Many drugs won't resolve to an openFDA label at all (esp. less common generics); `retrieve_evidence` treats that as a normal empty result, not an error.
- **Stage 3 ingredient parsing**: ingredient names are split from openFDA's free-text label fields with a simple comma/parenthetical/dosage heuristic, not a real parser -- footnote markers and pharmacopeia suffixes (e.g. "USP") sometimes survive in the extracted name. This doesn't affect allergy matching (substring matching still catches e.g. "ibuprofen" inside "ibuprofen usp"), but the rendered ingredient list isn't always clean.
- **Generation/verification latency**: a CPU-only local Ollama instance evals prompts at roughly 40ms/token. A 2-drug Stage 2 query's evidence block is ~3,300 tokens, and Stage 3 adds two more LLM round trips (decomposition + batched entailment) over that same evidence -- a full Stage 2+3 run can take several minutes on hardware like this.
- **Evidence scope**: retrieval is drug-name-centric. For a query where Stage 1 extracts no drug names at all (most pure symptom/home-remedy/general-info questions), evidence will be empty and Stage 2 correctly falls back to "not in my sources" -- this project does not implement a symptom- or recipe-specific evidence source, so those query types are gated by Stage 1 but not usefully answered by Stage 2/3 yet. Given the eval results above, quantifying how often this scope gap actually triggers (and running the deferred spikee adversarial slice against Stage 1/3, which haven't been tested against adversarial input at all) are probably higher-value next steps than building a new evidence source.
- **No conversation state across turns**: `process_query()` takes one raw string and reclassifies from scratch every call -- there's no mechanism yet to carry a clarifying question's answer forward into the next turn. Combined with the query-type misclassification risk above, this means a hypothetical multi-turn flow could reclassify differently turn to turn; not exercised or fixed here.
