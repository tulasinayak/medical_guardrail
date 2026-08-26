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

# All manual tests below need an LLM backend configured -- Ollama running locally
# (default) or OpenAI with an API key. See "LLM backend" below to switch.
# Prompt eval on a CPU-only Ollama instance is slow for anything with a large
# evidence block -- see Known limitations below -- so pass a longer timeout via
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

# Functional eval set: 28 hand-labeled cases against the real end-to-end pipeline
python -m eval.score

# Batched vs single-claim entailment ablation (real Ollama + network)
python -m eval.ablation_entailment

# Adversarial slice (see "Adversarial testing (spikee)" below for full setup)
pip install -e ".[adversarial]"
cd eval
spikee generate --seed-folder seeds-guardrail-bypass --format full-prompt --instruction-filter bypass-gate
spikee test --dataset datasets/<generated-file>.jsonl --target guardrail_target --target-options gate
```

## Architecture

- `src/medical_guardrails/orchestrator.py` — `MedicalGuardrailPipeline.process_query()`: runs Stage 1's gate first and returns immediately with a clarifying question if anything required is missing; otherwise runs Stage 2 retrieval + generation, then Stage 3 verification, using Stage 1's extracted `StructuredQuery` as the actual input (its `drug_names` feed retrieval, its `allergies` feed the ingredient check) rather than passing those in separately.
- `src/medical_guardrails/cli/pipeline_once.py` — the full end-to-end CLI, taking one raw natural-language query and nothing else.
- `src/medical_guardrails/common/schemas.py` — shared Pydantic models passed between stages: `StructuredQuery` (Stage 1's output), `EvidenceChunk` (Stage 2's output), `Claim` (Stage 3's output).
- `src/medical_guardrails/stage1_slotfill/` — `classifier.py` (LLM-based query-type classification + field extraction, using structured-output `format` -- a JSON schema constrains decoding so the reply structurally cannot deviate from it), `required_fields.py` (per-query-type required-fields table + clarifying questions), `gate.py` (`slot_fill_gate()`: ties the two together into a ready/needs_clarification decision).
- `src/medical_guardrails/stage2_generate/` — `rxnorm_client.py` (name → RxCUI, plus canonical-name lookup), `openfda_client.py` (label text by name: contraindications/warnings/interactions/ingredients), `ddinter_lookup.py` (local offline pairwise interaction severity), `retrieval.py` (combines all three into one evidence list, tagging each chunk's `authority` as `regulatory` (openFDA) or `curated_secondary` (DDInter)), `generation.py` (grounded generation via the configured LLM, with a system prompt that forbids answering outside the retrieved evidence -- and a hard, code-level fallback when there's no evidence at all; see Known limitations).
- `src/medical_guardrails/stage3_verify/` — `claim_decomposition.py` (splits a draft response into atomic claims via LLM), `entailment.py` (LLM-as-judge verdict per claim: supported/contradicted/unsupported, fails closed to unsupported on any parse failure; `verify_claims()` batches all claims into one call, `verify_claim_single()` checks one at a time for the ablation in `eval/ablation_entailment.py`), `ingredient_check.py` (extracts active/inactive ingredients from openFDA evidence and cross-checks against stated allergies -- a match is an automatic block), `verification.py` (`verify_response()`: ties all three into a pass/block decision + final response text, always rendering the ingredients list to the user; skips decomposition/verification entirely when there's no evidence).
- `src/medical_guardrails/llm/` — `base.py` (the `LLMClient` protocol every backend implements: `chat(messages, format=None) -> str`, `health_check() -> bool`), `ollama_client.py`, `openai_client.py`, `factory.py` (`build_llm_client(settings)`: picks the backend from `Settings.llm_provider`). See "LLM backend" below.
- `data/ddinter/build_ddinter_db.py` — one-time script to build the local DDInter SQLite dump from DDInter's public bulk CSV export (not committed; regenerate locally).
- `eval/` — see "Evaluation" below.

## LLM backend

Every stage calls its LLM through the `LLMClient` protocol (`llm/base.py`), so which backend is active is a config choice, not a code choice -- `classifier.py`, `generation.py`, `claim_decomposition.py`, `entailment.py` don't know or care which one they're talking to.

**Ollama (default)** -- free, local, no API key:
```bash
export MEDICAL_GUARDRAILS_LLM_PROVIDER=ollama   # default, can be omitted
export MEDICAL_GUARDRAILS_OLLAMA_MODEL=mistral:latest
```

**OpenAI** -- needs an API key and a model that supports structured outputs (gpt-4o-mini or later; older models will reject Stage 1's schema-constrained requests):
```bash
export MEDICAL_GUARDRAILS_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...                     # or MEDICAL_GUARDRAILS_OPENAI_API_KEY
export MEDICAL_GUARDRAILS_OPENAI_MODEL=gpt-4o-mini   # default
```
`OpenAIClient` is a thin `httpx` wrapper matching `OllamaClient`'s shape (no `openai` SDK dependency, consistent with this project's and `pii_guardrails`' minimal-dependency approach elsewhere). Its `format`-to-`response_format` translation adds `additionalProperties: false` to whatever schema it's given, since OpenAI's strict structured outputs require that and Ollama's grammar-based constraint doesn't -- `classifier.py`'s schema itself stays backend-agnostic.

This makes it straightforward to test whether the Stage 1 extraction-reliability issues documented below (mistral, and to a lesser extent Qwen3-8B) are specific to smaller local models or persist with a stronger hosted one -- run the same `eval/functional_cases.jsonl`/`eval/score.py` cases under each provider and compare, rather than guess.

## Evaluation

Two separate slices, reported as two separate numbers -- a low score on one tells you something different than a low score on the other, so they're never blended into one figure.

**Functional accuracy** (`eval/functional_cases.jsonl` + `eval/score.py`): 28 hand-labeled cases against the real end-to-end pipeline, each with a hand-written `expected_action` from a closed set (`ask_clarification`, `answer_grounded`, `block_ingredient_match`, `block_unsupported_claim`, `fallback_not_in_sources`). `eval/pipeline_adapter.py` holds the one place that maps `PipelineResult` onto that closed set, shared by `score.py` and the spikee target below so the mapping is never duplicated. The first three cases (`regr_001`–`regr_003`) are the three bugs found through manual live testing during this project's build (RxNorm/openFDA granularity mismatch, allergy-status misattribution, the dual-block scenario) -- frozen as regression cases because their expected behavior is already known from live testing, not guessed. `control_answer_grounded_*` is the control group: fully-specified, conflict-free queries that must actually get answered, catching a guardrail that just blocks or asks about everything. Run: `python -m eval.score` (needs Ollama + network; writes `eval/results/functional_run.jsonl`).

One category (`unsupported_claim_*`) is flagged in its own `notes` field as inherently non-deterministic: whether the model states a claim beyond the evidence depends on generation-time behavior, not something forceable through input text the way the other five behaviors are -- treat those four cases as samples to watch, not guaranteed reproductions.

**Adversarial resistance (spikee)**: tests whether a prompt-injection-style instruction can talk Stage 1's gate or Stage 3's block into standing down, using [spikee](https://github.com/ReversecLabs/spikee) (Reversec Labs). See "Adversarial testing (spikee)" below for the full setup -- it needs a separate install and its target/seed files use a real, different interface than a first pass at this might assume, documented there.

## Adversarial testing (spikee)

```bash
pip install -e ".[adversarial]"
```

Layout (all under `eval/`, matching spikee's own conventions rather than an arbitrary one):
- `seeds-guardrail-bypass/base_user_inputs.jsonl` — 4 base documents, each a medical query with a `<PLACEHOLDER>` where the attack text gets substituted. Two families: `gate-incomplete-*` (deliberately missing allergy/age, so Stage 1 should ask a question) and `block-dual-conflict-01` (fully specified, live-verified to trigger `block_ingredient_match` in the unmodified pipeline).
- `seeds-guardrail-bypass/instructions.jsonl` — 7 attack instructions, tagged `instruction_type: bypass-gate` (try to get Stage 1 to skip its question) or `bypass-block` (try to get Stage 3 to wave through the dual-conflict case).
- `seeds-guardrail-bypass/jailbreaks.jsonl` — 5 wrapping/framing styles (a `no-jailbreak` baseline, roleplay, fake system-message injection, developer-authority claim, "pre-approved by compliance" framing).
- `targets/guardrail_target.py` — spikee's real `Target` interface is a class with `process_input()` returning a **bool** (`True` = the check under test was bypassed, `False` = it held), not a string-returning function the way a first pass at this might assume. It supports two modes via `--target-options`: `gate` (default) checks whether an incomplete query got answered instead of gated; `block` checks whether the dual-conflict query got answered instead of blocked.

**Run the two attack families separately** — they test different checks, so a single run mixing both isn't meaningful (see the docstring in `guardrail_target.py` for why):

```bash
cd eval
spikee generate --seed-folder seeds-guardrail-bypass --format full-prompt --instruction-filter bypass-gate
spikee test --dataset datasets/<generated-file>.jsonl --target guardrail_target --target-options gate

spikee generate --seed-folder seeds-guardrail-bypass --format full-prompt --instruction-filter bypass-block
spikee test --dataset datasets/<generated-file>.jsonl --target guardrail_target --target-options block
```

All `judge_name`/`judge_args` in `instructions.jsonl` are set to an always-matching regex (`regex` / `.*`), per spikee's own guidance for guardrail testing -- the real pass/fail signal here is the target's boolean return, not judge-based content matching.

**Corrections made relative to a first-draft version of this setup**, worth knowing about since they change what files exist and where:
- A third seed file, `base_user_inputs.jsonl`, is required by spikee's composable-dataset format alongside `instructions.jsonl`/`jailbreaks.jsonl` -- it wasn't in an earlier sketch of this plan.
- The target lives at `eval/targets/guardrail_target.py`, not `eval/spikee_target.py` -- spikee discovers targets in a `targets/` folder by convention, referenced by module name (`--target guardrail_target`), not by file path. Run `spikee generate`/`spikee test` from inside `eval/` so `targets/` and the seed folder resolve relative to that.
- `process_input` returns a bool (bypass/no-bypass), not a string a generic judge parses -- spikee has a dedicated guardrail-testing convention for exactly this shape of test, which the target file follows instead of encoding the outcome into response text.

This hasn't been run yet (spikee isn't installed in this environment) -- `eval/pipeline_adapter.py`'s import and `functional_cases.jsonl`'s JSON have been verified directly, but the spikee-dependent pieces (target file, generate/test commands) are unverified beyond matching spikee's documented interface as closely as possible.

## Known limitations

- **Stage 1 extraction accuracy is the biggest open gap, and it's worse in aggregate than early spot-checks suggested.** Switching from a hand-rolled line-based text format to structured-output `format` (a JSON-schema-constrained decode) completely eliminated malformed/unparseable output -- every response is now valid JSON matching the schema, with no parsing failures observed. It did **not** reliably fix the deeper semantic problem: constrained decoding guarantees the *shape* of the output, not that the model places each value under the *semantically correct* key. A full `python -m eval.score` run against `mistral:latest` across the 28-case `eval/functional_cases.jsonl` scored 11/28 (39%) -- with a striking pattern: `ask_clarification` cases passed 11/11 (100%), while every category that should have *proceeded* past Stage 1 (`answer_grounded`, `block_ingredient_match`, `block_unsupported_claim`, `fallback_not_in_sources`) scored 0% as a group, 14 of those 17 cases landing on `ask_clarification` instead. Spot-checking one of those failures 3 more times immediately afterward got it right all 3 tries, confirming this is real model noise rather than a code bug in `required_fields.py` -- but the aggregate rate across diverse phrasing (~82% false-clarification among cases that should've proceeded) is meaningfully worse than the ~50/50 estimate from smaller repeated-query spot checks earlier in this project. This asymmetry is the safe direction (over-asking, never silently skipping a required field), but it's a real usability cost, not just noise to wave away. Comparing this against the OpenAI backend (see "LLM backend" above) on the same 28 cases is the natural next check.
- **A same-size, newer model (Qwen3-8B) doesn't cleanly solve this either.** Tested head-to-head against the same failing queries: Qwen3-8B correctly recognized "no allergies" in 3 of 4 relevant cases (better than mistral's roughly 1-of-2), but it hit the *exact same* `SYMPTOM` misclassification on the lactose-allergy query, missed an explicit "I'm an adult" age statement in one case, and introduced a new failure mode mistral didn't have: it fabricated `warfarin` as a "current medication" in 3 of 4 runs, when warfarin was the drug being asked about, not something the patient said they were already taking. It was also markedly slower per call (roughly 100-225s vs. mistral's tighter range) -- likely Qwen3's hidden "thinking" tokens, which don't appear in the final JSON content but plausibly explain the latency given how short the actual output is. Net finding: the model swap traded one accuracy problem for a different one rather than raising the ceiling outright, at a real speed cost -- a genuinely larger model (or fine-tuning for this exact extraction task) looks more promising than lateral model swaps at this size class.
- **Fixed: empty evidence could silently be answered from parametric memory instead of declining.** The same `eval.score` run surfaced this directly: `symptom_no_drug_004` (no drug named, so zero evidence retrieved) got `answer_grounded` -- the model answered from its own general knowledge despite the system prompt instructing it to decline, because that instruction was the *only* thing enforcing it. Fixed in code, not by prompting harder: `generate_grounded_response()` now returns the not-in-sources fallback directly and skips the LLM call entirely whenever `evidence` is empty, since there's nothing a retrieval-grounded prompt could add in that case anyway. Live-verified against the exact failing query -- now deterministically returns the fallback.
- **Fixed: an honest "I don't know" could get blocked as if it were a hallucination.** Same run: `symptom_no_drug_001`/`002` expected `fallback_not_in_sources` but got `block_unsupported_claim` -- Stage 3 was decomposing the model's own "not in my sources" decline into a claim and then blocking it, since any claim checked against empty evidence trivially reads as unsupported. Fixed alongside the above: `verify_response()` now skips claim decomposition/verification entirely when evidence is empty (nothing to check it against) and passes the deterministic fallback straight through. This also happens to save two LLM calls per empty-evidence query. Not fixed: the *partial*-coverage case, where evidence exists but doesn't cover part of a mixed answer, still relies on the model correctly saying so rather than a code-level check -- only the all-or-nothing empty-evidence case is now deterministic.
- **Batched vs. single-claim entailment**: ran the ablation live (`eval/ablation_entailment.py`) across two cases (1 claim, then 2 claims) -- verdicts matched between batched and single-claim verification both times, no divergence observed. Sample size is small (3 claims total), so this doesn't rule out attention dilution on responses with more claims, but it doesn't support switching off the batched default either; kept batched for speed pending a larger sample.
- **Evidence authority is tagged but not yet used for anything beyond display.** `EvidenceChunk.authority` (`regulatory` for openFDA, `curated_secondary` for DDInter) is surfaced in every evidence block shown to both the generation and verification prompts, but no weighting/scoring logic reads it yet -- that's intentionally left for a future iteration.
- **Stage 2/DDInter**: DDInter's bulk CSV export only carries interaction severity (Major/Moderate/Minor), not the mechanism/management text DDInter shows on its per-pair detail pages. Many drugs won't resolve to an openFDA label at all (esp. less common generics); `retrieve_evidence` treats that as a normal empty result, not an error.
- **Stage 3 ingredient parsing**: ingredient names are split from openFDA's free-text label fields with a simple comma/parenthetical/dosage heuristic, not a real parser -- footnote markers and pharmacopeia suffixes (e.g. "USP") sometimes survive in the extracted name. This doesn't affect allergy matching (substring matching still catches e.g. "ibuprofen" inside "ibuprofen usp"), but the rendered ingredient list isn't always clean.
- **Generation/verification latency**: a CPU-only local Ollama instance evals prompts at roughly 40ms/token. A 2-drug Stage 2 query's evidence block is ~3,300 tokens, and Stage 3 adds two more LLM round trips (decomposition + batched entailment) over that same evidence -- a full Stage 2+3 run can take several minutes on hardware like this.
- **Evidence scope**: retrieval is drug-name-centric. For a query where Stage 1 extracts no drug names at all (most pure symptom/home-remedy/general-info questions), evidence will be empty and Stage 2 correctly falls back to "not in my sources" -- this project does not implement a symptom- or recipe-specific evidence source, so those query types are gated by Stage 1 but not usefully answered by Stage 2/3 yet. `eval/functional_cases.jsonl`'s `home_remedy_recipe_scope_gap_001` documents this directly: a pure recipe query can neither be answered nor have its allergens checked, only correctly deferred. `python -m eval.score`'s per-action breakdown is the way to quantify how often this actually triggers, rather than guessing.
- **Adversarial resistance is unverified**: the spikee-based adversarial slice (see "Adversarial testing (spikee)" above) is built but not yet run in this environment -- Stage 1's gate and Stage 3's block haven't been tested against a single adversarial input yet, only against ordinary phrasing.
- **No conversation state across turns**: `process_query()` takes one raw string and reclassifies from scratch every call -- there's no mechanism yet to carry a clarifying question's answer forward into the next turn. Combined with the query-type misclassification risk above, this means a hypothetical multi-turn flow could reclassify differently turn to turn; not exercised or fixed here.
