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
```

## Architecture

- `src/medical_guardrails/orchestrator.py` — `MedicalGuardrailPipeline.process_query()`: runs Stage 1's gate first and returns immediately with a clarifying question if anything required is missing; otherwise runs Stage 2 retrieval + generation, then Stage 3 verification, using Stage 1's extracted `StructuredQuery` as the actual input (its `drug_names` feed retrieval, its `allergies` feed the ingredient check) rather than passing those in separately.
- `src/medical_guardrails/cli/pipeline_once.py` — the full end-to-end CLI, taking one raw natural-language query and nothing else.
- `src/medical_guardrails/common/schemas.py` — shared Pydantic models passed between stages: `StructuredQuery` (Stage 1's output), `EvidenceChunk` (Stage 2's output), `Claim` (Stage 3's output).
- `src/medical_guardrails/stage1_slotfill/` — `classifier.py` (LLM-based query-type classification + field extraction, line-based output format), `required_fields.py` (per-query-type required-fields table + clarifying questions), `gate.py` (`slot_fill_gate()`: ties the two together into a ready/needs_clarification decision).
- `src/medical_guardrails/stage2_generate/` — `rxnorm_client.py` (name → RxCUI, plus canonical-name lookup), `openfda_client.py` (label text by name: contraindications/warnings/interactions/ingredients), `ddinter_lookup.py` (local offline pairwise interaction severity), `retrieval.py` (combines all three into one evidence list), `generation.py` (grounded generation via Ollama, with a system prompt that forbids answering outside the retrieved evidence).
- `src/medical_guardrails/stage3_verify/` — `claim_decomposition.py` (splits a draft response into atomic claims via LLM), `entailment.py` (batched LLM-as-judge verdict per claim: supported/contradicted/unsupported, fails closed to unsupported on any parse failure), `ingredient_check.py` (extracts active/inactive ingredients from openFDA evidence and cross-checks against stated allergies -- a match is an automatic block), `verification.py` (`verify_response()`: ties all three into a pass/block decision + final response text, always rendering the ingredients list to the user).
- `data/ddinter/build_ddinter_db.py` — one-time script to build the local DDInter SQLite dump from DDInter's public bulk CSV export (not committed; regenerate locally).

## Known limitations

- **Stage 1 field-attribution accuracy**: the local 7B model doesn't perfectly attribute a "no X" statement to the right field -- e.g. "I have no drug allergies" is sometimes marked as a statement about current medications instead of allergies (worse with an example added to the prompt, but not fully fixed). This fails in the safe direction: it can cause one redundant clarifying question, but the classifier never falsely marks an unmentioned field as "explicitly none" and skips asking (verified live) -- the required-fields gate never lets a genuinely-missing field through.
- **Stage 2/DDInter**: DDInter's bulk CSV export only carries interaction severity (Major/Moderate/Minor), not the mechanism/management text DDInter shows on its per-pair detail pages. Many drugs won't resolve to an openFDA label at all (esp. less common generics); `retrieve_evidence` treats that as a normal empty result, not an error.
- **Stage 3 ingredient parsing**: ingredient names are split from openFDA's free-text label fields with a simple comma/parenthetical/dosage heuristic, not a real parser -- footnote markers and pharmacopeia suffixes (e.g. "USP") sometimes survive in the extracted name. This doesn't affect allergy matching (substring matching still catches e.g. "ibuprofen" inside "ibuprofen usp"), but the rendered ingredient list isn't always clean.
- **Generation/verification latency**: a CPU-only local Ollama instance evals prompts at roughly 40ms/token. A 2-drug Stage 2 query's evidence block is ~3,300 tokens, and Stage 3 adds two more LLM round trips (decomposition + batched entailment) over that same evidence -- a full Stage 2+3 run can take several minutes on hardware like this.
- **Evidence scope**: retrieval is drug-name-centric. For a query where Stage 1 extracts no drug names at all (most pure symptom/home-remedy/general-info questions), evidence will be empty and Stage 2 correctly falls back to "not in my sources" -- this project does not implement a symptom- or recipe-specific evidence source, so those query types are gated by Stage 1 but not usefully answered by Stage 2/3 yet.
