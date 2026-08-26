# Medical Guardrails

A staged guardrail that wraps an LLM for health-related queries — the sibling
project to [`pii_guardrails`](../pii_guardrails), same "interception layer
between prompt and model" architecture, different check.

```
User prompt
    │
    ▼
[Stage 1: pre-generation slot-filling]   -- not yet built
    │  classify query type, require key fields (allergies, meds, age),
    │  ask a clarifying question instead of generating if any are missing
    ▼
[Stage 2: grounded generation]           -- this repo currently implements this
    │  retrieve evidence (RxNorm identity resolution, openFDA label text,
    │  local DDInter interaction severities) and generate constrained to
    │  ONLY that retrieved evidence
    ▼
[Stage 3: post-generation claim verification]  -- not yet built
    │  decompose the draft response into atomic claims, check each against
    │  the retrieved evidence, block/rewrite anything unsupported
    ▼
Final response
```

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate                    # Windows
pip install -e ".[dev]"

# Build the local DDInter interaction database (one-time; downloads DDInter's
# public per-letter CSV export)
python data/ddinter/build_ddinter_db.py

# Stage 2 manual test (requires Ollama running locally with the configured
# model pulled, default mistral:latest)
python -m medical_guardrails.stage2_generate.cli ibuprofen warfarin

# Tests
pytest tests/unit                          # fast, fully mocked, no network/Ollama needed
pytest tests -m "not integration"          # same as above
pytest tests -m integration                # real RxNorm/openFDA calls + real Ollama
```

## Architecture

- `src/medical_guardrails/common/schemas.py` — shared Pydantic models passed between stages: `StructuredQuery` (Stage 1's output), `EvidenceChunk` (Stage 2's output, fully used today), `Claim` (Stage 3's output).
- `src/medical_guardrails/stage2_generate/` — `rxnorm_client.py` (name → RxCUI), `openfda_client.py` (label text: contraindications/warnings/interactions/ingredients), `ddinter_lookup.py` (local offline pairwise interaction severity), `retrieval.py` (combines all three into one evidence list), `generation.py` (grounded generation via Ollama, with a system prompt that forbids answering outside the retrieved evidence).
- `data/ddinter/build_ddinter_db.py` — one-time script to build the local DDInter SQLite dump from DDInter's public bulk CSV export (not committed; regenerate locally).
- `src/medical_guardrails/stage1_slotfill/`, `src/medical_guardrails/stage3_verify/` — empty package stubs, to be filled in in later sessions.

## Known limitations of Stage 2 as it stands

- DDInter's bulk CSV export only carries interaction severity (Major/Moderate/Minor), not the mechanism/management text DDInter shows on its per-pair detail pages.
- Many drugs won't resolve to an openFDA label at all (esp. less common generics); `retrieve_evidence` treats that as a normal empty result, not an error.
- Stage 1 doesn't exist yet, so there's no allergy/current-medication gating — the CLI takes drug names directly and has no awareness of the user's allergies.
