# Context Guardrail

A lightweight, domain-agnostic guardrail: before a user's request goes to a
"main" LLM, a separate **Context Guardrail** decides whether there's
enough information to answer well and safely, asking a targeted
clarifying question if not. The medical use case that motivated this
project is kept as the flagship demo, but the core mechanism doesn't know
anything about medicine — it's parameterized by a `DomainSchema`, and
medical is one concrete example of that, not the architecture itself.

```
USER REQUEST
     │
     ▼
CONTEXT GUARDRAIL
  classify intent + a coarse "general vs. personal" scope, extract
  whatever's already stated, ask only about what's genuinely missing
  for *this* request (up to 5 questions, never a fixed number) --
  proceeds anyway once the question budget runs out, so refusing to
  answer isn't the fallback for an uncooperative user
     │
     ▼
MAIN LLM
  one prompt: the original request + whatever context was gathered +
  an explicit note of anything still unresolved -- answers directly,
  no retrieval, no separate verification step
     │
     ▼
[medical demo only] one narrow, deterministic ingredient/allergy
check against a public drug-label database, if drugs and allergies
were both mentioned
     │
     ▼
FINAL RESPONSE
```

## Why this shape (and what it used to be)

Earlier versions of this project ran three stages: a slot-filling gate,
retrieval-grounded generation (RxNorm + openFDA + DDInter + MedlinePlus),
and a separate LLM-based claim verifier that decomposed the draft answer
and blocked it if any claim wasn't explicitly supported by the retrieved
evidence. That verifier had a real, observed failure mode: a back-pain
answer that was mostly correct (OTC pain relief, avoid prolonged bed
rest, see a doctor if severe) got entirely blocked because one incidental
claim — "back pain can persist for five days" — wasn't explicitly in the
retrieved evidence. Evidence not mentioning something is not the same as
evidence contradicting it, and treating them the same produced exactly
the over-blocking behavior a guardrail should avoid.

Retrieval and the separate verifier were both removed. What's left is a
narrower, more honest, and more testable claim:

> **Does an external, code-enforced context-sufficiency gate produce
> fewer premature or unpersonalized answers than trusting a
> system-prompted LLM to decide for itself whether it has enough
> information?**

This project has direct evidence the underlying mechanism matters even
when the *judgment* behind it doesn't improve: the classifier that
decides what's missing is measurably unreliable on its own (see Known
limitations) — mistral has been observed misattributing an explicit "no
allergies" statement roughly 30–50% of the time. The Context Guardrail's
value isn't that it judges *better*; it's that `missing_fields()`
returning non-empty is a hard, code-level stop, whereas a model deciding
mid-generation that it should ask is a soft, optional behavior it can
skip in a single turn. `eval/baseline_compare.py` tests this directly
against two baselines (see Evaluation).

**Named trade-off**: removing retrieval means this project can no longer
claim any answer is "grounded in evidence" — that property doesn't exist
anymore. What survives is a narrower claim: the system asks enough
questions before answering, and is explicit about what it couldn't
personalize, rather than claiming certainty it doesn't have. Nothing here
claims medical correctness, and it isn't a production medical product.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate                    # Windows
pip install -e ".[dev]"

# Needs an LLM backend configured -- Ollama running locally (default) or
# OpenAI with an API key. See "LLM backend" below to switch.
# Prompt eval on a CPU-only Ollama instance can be slow; pass a longer
# timeout via MEDICAL_GUARDRAILS_OLLAMA_TIMEOUT_SECONDS if calls time out.

# Full pipeline: the Context Guardrail extracts drug names/allergies/age
# from the raw request itself -- no separate flags needed.
python -m medical_guardrails.cli.pipeline_once "What is ibuprofen used for?"
python -m medical_guardrails.cli.pipeline_once "Can I take ibuprofen with warfarin?"

# Individual pieces, for testing one in isolation:
python -m medical_guardrails.context_guardrail.cli "Can I take ibuprofen with warfarin?"
python -m medical_guardrails.main_llm.cli "Can I take ibuprofen?" --context age_bracket=adult

# Interactive: type a request, answer the guardrail's clarifying question(s)
# at a real terminal prompt (up to --max-questions, default 5), then Main
# LLM actually runs and the whole thing is saved to a file -- see
# "Interactive prompt builder" below.
python -m medical_guardrails.cli.interactive_prompt_builder

# GUI: same flow, plus a checkpoint to review the exact Main LLM prompt
# before approving it.
pip install -e ".[gui]"
python -m streamlit run src/medical_guardrails/gui/streamlit_app.py

# Tests
pytest tests/unit                          # fast, fully mocked, no network/Ollama needed
pytest tests -m "not integration"          # same as above
pytest tests -m integration                # real Ollama + one real openFDA call

# Functional eval set against the real end-to-end pipeline
python -m eval.score

# Three-way comparison: Main LLM alone vs. Main LLM + a system prompt
# telling it to ask vs. the real Context Guardrail -> Main LLM pipeline
python -m eval.baseline_compare
```

## Architecture

- `src/medical_guardrails/orchestrator.py` — `MedicalGuardrailPipeline.process_query()`: runs the Context Guardrail first and returns immediately with a clarifying question if anything required is missing; otherwise calls Main LLM directly, then (medical domain only, and only if drug names *and* allergies were both mentioned) runs the ingredient/allergy safety check. Supports independent `guardrail_llm_client`/`main_llm_client` — e.g. a cheap local model gating access to a stronger hosted one — for exactly the kind of configuration comparison `eval/baseline_compare.py` is built around.
- `src/medical_guardrails/context_guardrail/` — the domain-agnostic mechanism: `domain.py` (`DomainSchema`/`FieldSpec` — the generic shape any domain plugs into), `domains/medical.py` (the one concrete domain shipped: 5 query types, 8 fields, required-fields table), `classifier.py` (LLM-based classification + extraction via structured-output `format`, including the `answer_scope: general | personal` field that decides whether the per-type required-fields table even applies), `required_fields.py` (`missing_fields()` — returns `[]` unconditionally for `general` scope), `gate.py` (`slot_fill_gate()` ties it together into a ready/needs_clarification decision), `interactive.py` (multi-turn loop with an injected `ask_fn`, reused by both the CLI and GUI).
- `src/medical_guardrails/main_llm/generation.py` — `generate_answer(user_request, context, unresolved_fields, llm_client)`: one prompt, no retrieval, no evidence block. If fields are still unresolved (question budget exhausted without full resolution), the prompt explicitly tells the model to say what it can't personalize rather than silently ignoring the gap.
- `src/medical_guardrails/medical/` — the medical demo's own narrow extension, not part of the general mechanism: `openfda_client.py` (drug name → active/inactive ingredients only, via a public label lookup), `ingredient_safety.py` (`check_drug_allergy_conflicts()` — deterministic substring match against stated allergies; a match is an automatic block, since allergic reactions are often to excipients/fillers rather than the active compound).
- `src/medical_guardrails/common/schemas.py` — `DomainQuery`: the Context Guardrail's output (`raw_text`, `query_type`, `answer_scope`, `fields`), domain-agnostic and shared by everything downstream.
- `src/medical_guardrails/llm/` — `base.py` (the `LLMClient` protocol: `chat(messages, format=None) -> str`, `health_check() -> bool`), `ollama_client.py`, `openai_client.py`, `factory.py` (`build_llm_client(settings)`). See "LLM backend" below.
- `src/medical_guardrails/gui/streamlit_app.py` — see "GUI" below.
- `eval/` — see "Evaluation" below.

## LLM backend

Everything calls its LLM through the `LLMClient` protocol (`llm/base.py`), so which backend is active is a config choice, not a code choice.

**Ollama (default)** — free, local, no API key:
```bash
export MEDICAL_GUARDRAILS_LLM_PROVIDER=ollama   # default, can be omitted
export MEDICAL_GUARDRAILS_OLLAMA_MODEL=mistral:latest
```

**OpenAI** — needs an API key and a model that supports structured outputs (gpt-4o-mini or later):
```bash
export MEDICAL_GUARDRAILS_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...                     # or MEDICAL_GUARDRAILS_OPENAI_API_KEY
export MEDICAL_GUARDRAILS_OPENAI_MODEL=gpt-4o-mini   # default
```
`OpenAIClient` is a thin `httpx` wrapper matching `OllamaClient`'s shape (no `openai` SDK dependency). Its `format`-to-`response_format` translation adds `additionalProperties: false` since OpenAI's strict structured outputs require that and Ollama's grammar-based constraint doesn't.

**Alternative OpenAI model — GPT-5.6 Luna**: set `MEDICAL_GUARDRAILS_OPENAI_MODEL=gpt-5.6-luna` for OpenAI's cost-efficient GPT-5.6-family model instead of the `gpt-4o-mini` default — supports Chat Completions and structured outputs, so it's a drop-in swap. Opt-in only, not benchmarked against the default yet.

`MedicalGuardrailPipeline` accepts `guardrail_llm_client`/`main_llm_client` independently, so the Context Guardrail and Main LLM don't have to be the same model/provider — useful for testing whether a cheap local gate in front of a stronger hosted model changes the picture.

## GUI

```bash
pip install -e ".[gui]"
python -m streamlit run src/medical_guardrails/gui/streamlit_app.py
```

Same flow as the interactive CLI, plus a checkpoint the CLI doesn't have:
once the Context Guardrail is satisfied (or its question budget runs
out), the exact prompt that would go to Main LLM is shown before it's
sent — nothing happens until you click "Approve & send to Main LLM". For
the medical domain, the ingredient/allergy check runs after Main LLM
answers and is shown separately.

## Interactive prompt builder

A terminal front end for watching the multi-turn clarification loop directly and running the real pipeline end to end.

```bash
python -m medical_guardrails.cli.interactive_prompt_builder
python -m medical_guardrails.cli.interactive_prompt_builder "Can I take ibuprofen with warfarin?" --max-questions 3
python -m medical_guardrails.cli.interactive_prompt_builder --no-generate   # save the prompt only, don't call the LLM
```

You type (or pass) an initial request; if anything required is missing,
the guardrail's clarifying question is printed and you answer it; your
answer is appended to the conversation and the gate re-checks the *whole*
accumulated text from scratch (no incremental field-merging — simpler
and more robust than patching a partial `DomainQuery`, at the cost of a
full reclassification call each round). This repeats until every
required field resolves or `--max-questions` (default 5) is used up.
Either way, Main LLM then runs for real (told explicitly which fields
are still unresolved if the budget ran out), the medical ingredient
check runs if applicable, and everything — the conversation, the exact
prompt, the answer, any ingredient-conflict result — is saved to a
timestamped file under `eval/results/interactive_prompts/`.

The multi-turn loop itself (`context_guardrail/interactive.py`'s `run_interactive_slot_fill()`) is pure logic with an injected `ask_fn`, decoupled from the terminal — shared as-is by the GUI.

Note: `MedicalGuardrailPipeline.process_query()` (the one-shot path used by `pipeline_once.py` and `eval/score.py`) still reclassifies fresh on every call with no memory between calls — the multi-turn conversation state lives only in the interactive tools built on top of it.

## Results at a glance

All three experiments below use the same 35-case set (`eval/functional_cases.jsonl`) and GPT-4o-mini as the generating model; see "Evaluation" for methodology and "On trusting these numbers" for how much weight each figure can bear.

**Functional accuracy** (`python -m eval.score`):

| | pass rate |
|---|---|
| Overall | 32/35 (91.4%) |
| `answered` | 17/17 |
| `ask_clarification` | 14/16 |
| `blocked_ingredient_match` | 1/2 |

**Does enforcement beat prompting?** (`python -m eval.baseline_compare`) — same model throughout, only the mechanism differs:

| configuration | correct | unnecessary questions | premature answers |
|---|---|---|---|
| Main LLM directly | 19/35 | 0 | 16 |
| Main LLM + "ask if missing" system prompt | 20/35 | 0 | 15 |
| Context Guardrail → Main LLM | 31/35 | 2 | 2 |

The prompted model asked exactly **once** across all 35 cases, despite 16 needing it — a soft instruction doesn't reliably produce the behavior; a code-level gate does.

**Answer quality rubric**, 0–2 scale (`python -m eval.quality_judge`), judged by GPT-4o and cross-checked by an independent fresh Claude agent with no project context:

| baseline | GPT-4o judge (avg) | Claude judge (avg) |
|---|---|---|
| Main LLM directly | 1.90 | 1.77 |
| Main LLM + system prompt | 1.87 | 1.81 |
| Context Guardrail | 1.58 | 1.50 |

Both judges rank the three configurations identically. The Guardrail's lower average is almost entirely the mechanical cost of grading a clarifying question as an incomplete answer: restricted to the 19/35 cases it actually answers, its quality score is 1.95 — on par with the other two.

## Evaluation

**Functional accuracy** (`eval/functional_cases.jsonl` + `eval/score.py`): 35 hand-labeled cases against the real end-to-end pipeline, each with a hand-written `expected_action` from a closed set (`ask_clarification`, `answered`, `blocked_ingredient_match` — much smaller than before the retrieval/verification removal, since there's no "no evidence" state or claim-verdict-driven block anymore). `eval/pipeline_adapter.py` holds the one place that maps `PipelineResult` onto that closed set. `regr_001`–`regr_003` are regression cases from bugs found through manual live testing during this project's build. `control_answered_*` are fully-specified, conflict-free queries that must actually get answered. `control_general_scope_*` are the two cases that most directly test the new `answer_scope` mechanism — abstract phrasing that should skip the personal-context gate entirely. Latest run (GPT-4o-mini): 32/35 (91.4%) — see "On trusting these numbers" below before treating that as precise. Run: `python -m eval.score` (needs Ollama or an OpenAI key + network; writes `eval/results/functional_run.jsonl`).

**Baseline comparison** (`eval/baseline_compare.py`): the actual research question this redesign is about — does the Context Guardrail beat simpler alternatives? Runs the same case set through three configurations: (1) Main LLM directly, (2) Main LLM with a system prompt telling it to ask for missing information itself (detected via an explicit marker it's asked to use, not a fragile heuristic), (3) the real pipeline. Automates whether each configuration asked or answered versus each case's hand-labeled expectation; does **not** automate answer quality/usefulness, which is printed side by side per case for manual comparison rather than pretended to be scored. Run: `python -m eval.baseline_compare` (needs Ollama + network; slower than `score.py` since it makes ~3x the LLM calls).

**Answer quality rubric** (`eval/quality_judge.py`): `baseline_compare.py` deliberately doesn't score answer quality, just ask-vs-answer correctness. This scores the same 105 replies (35 cases x 3 baselines) on a 5-criterion 0-2 rubric (relevance, completeness, appropriate uncertainty, context use, overall usefulness), graded blind (the judge never sees which baseline produced a reply) by a different model (GPT-4o) than the one generating the replies (GPT-4o-mini), to limit same-model self-preference. Result: Guardrail 1.58 avg vs. 1.87-1.90 for the other two -- but that's almost entirely the mechanical cost of grading a clarifying question as an "incomplete answer": split by outcome, the Guardrail's actual answers (19/35 cases) score 1.95, on par with the other baselines, and the 16 cases it correctly defers on score low on completeness (0.25) by definition. Run: `python -m eval.quality_judge` (same requirements as `baseline_compare.py`, plus a separate judge call per reply).

**On trusting these numbers.** Three things worth stating plainly rather than leaving implicit:
- **The case labels have no independent labeler.** All 35 `expected_action` values in `functional_cases.jsonl` were written by the same person who built the pipeline, already knowing its architecture -- there's no blind ground truth here. At least one case (`home_remedy_unstated_allergy_001`) was flagged as genuinely ambiguous in its own notes at label-writing time, and one (`home_remedy_ingredient_002`, see below) turned out to be an outright wrong label. Treat headline pass rates as having a real +/-1-2 case margin from labeling judgment calls alone, on top of whatever the model itself gets wrong -- a 32/35 could reasonably have been labeled as 30/35 or 34/35 by someone else.
- **One ingredient-check test case was empirically wrong, and the others in that pattern haven't been re-audited.** `home_remedy_ingredient_002` originally expected `blocked_ingredient_match` for a lactose-intolerance-plus-acetaminophen case; checking live against the openFDA API showed the actual returned acetaminophen label contains no lactose, so blocking was never correct here (the case's own notes had already flagged this as unconfirmed) -- relabeled to `answered`. `home_remedy_ingredient_001` (ibuprofen, same allergy) was separately live-verified to be correctly labeled. The other cases built on similar drug/ingredient assumptions have not been individually re-checked against a live label the same way, so treat that pattern's correctness as an open question, not a re-audited certainty.
- **The quality-rubric's "blind" grading is blind to the label, not to identity.** Guardrail clarifying questions are built by joining fixed strings straight from each field's `FieldSpec.clarifying_question` (`context_guardrail/gate.py`) -- never LLM-generated -- so a question like "Do you have any known drug allergies? (If none, just say so.)" identifies the Guardrail on sight regardless of shuffling, for the ~16/35 cases where it asks. A same-vendor bias check (GPT-4o judging GPT-4o-mini's output) was cross-checked with a second, independently-run judge (a fresh Claude agent, given only the rubric and the anonymized replies, with no knowledge of this project or which system produced what) -- the two judges agreed closely (Guardrail avg 1.50 vs. 1.58, all baselines ranked the same way, all criteria within 0.26 except `appropriate_uncertainty` on the Guardrail's ask-cases at 0.34, where GPT-4o was more generous). The Claude judge also surfaced something GPT-4o didn't flag: on 2 of the Guardrail's ask-cases where the user had already stated a directly relevant allergy in the same message, the fixed-template question asked only about the next missing field and never acknowledged the allergy already given -- scored 0 across every criterion. That's a real, specific weakness in the templated-question mechanism (it doesn't reference what's already known), not just the generic "asking costs completeness points" effect.

**Adversarial resistance (spikee)**: built in an earlier version of this project, targeting the old Stage 1 gate / Stage 3 block split. **Deferred, not reworked** — see `eval/targets/guardrail_target.py`'s own docstring; it's been patched just enough to not reference removed types, but it was never run even before the redesign (spikee isn't installed in this environment) and its bypass semantics haven't been reconsidered for the new `answer_scope` mechanism. Treat it as stale until someone deliberately revisits it.

## Known limitations

- **Context Guardrail extraction accuracy is the biggest open gap.** Structured-output `format` (JSON-schema-constrained decoding) eliminated malformed/unparseable output entirely, but constrained decoding guarantees the *shape* of a reply, not that the model places each value under the *semantically correct* key. Mistral has been observed misattributing an explicit "no allergies" statement to the wrong field roughly 30–50% of the time depending on phrasing, and misclassifying `answer_scope`/`query_type` under some phrasings. This is the reason `eval/baseline_compare.py`'s finding matters more than raw accuracy: the guardrail's value isn't that the underlying judgment is more reliable (it uses the same model), it's that the judgment is *enforced* rather than optional.
- **A same-size, newer model (Qwen3-8B) doesn't cleanly solve this either.** Tested head-to-head against known failing queries in an earlier version of this project: Qwen3-8B recognized "no allergies" slightly more often than mistral, but hit the same misclassification pattern on at least one case and introduced a new failure mode (fabricating a drug being asked about as a "current medication"). Net finding at the time: model swaps at this size class traded one accuracy problem for another rather than raising the ceiling — a genuinely larger model looks more promising than lateral swaps.
- **No grounding, by design, not oversight.** Retrieval (RxNorm/openFDA-for-answers/DDInter/MedlinePlus) was removed entirely in this redesign — see "Why this shape" above. Main LLM answers from its own parametric knowledge. The one exception is the medical demo's narrow ingredient/allergy lookup, which exists only to feed that one deterministic check, not to ground the answer text itself.
- **The `answer_scope` general/personal split is coarse on purpose, and imperfect.** It's a single binary classification rather than per-field materiality judgment, chosen specifically because this project has already measured small models being unreliable at fine-grained field-level judgments — a coarser, more answerable question was judged more likely to be *followed correctly* than a more precise one that's harder to get right. `control_general_scope_001`/`002` and `home_remedy_unstated_allergy_001` in the eval set exercise exactly this boundary; treat their actual outcomes as informative, not a guaranteed pass.
- **Medical ingredient check coverage**: only runs when both a drug name and an allergy were mentioned — a home-remedy or symptom query with a stated allergy but no drug name (e.g. a recipe with a named allergen) isn't checked at all, since the lookup is drug-label-specific. Ingredient names are also split from openFDA's free-text label fields with a simple comma/parenthetical/dosage heuristic, not a real parser — footnote markers and pharmacopeia suffixes (e.g. "USP") sometimes survive in the extracted name, though this doesn't affect substring-based allergy matching.
- **Latency**: a CPU-only local Ollama instance evaluates prompts at roughly 40ms/token, and the Context Guardrail re-runs full classification on the whole accumulated conversation each round rather than incrementally — a multi-question interactive run can take a couple of minutes on hardware like this.
- **Adversarial resistance is unverified**: see "Adversarial testing (spikee)" above — deferred, not run.
- **No conversation state in the one-shot path**: `process_query()` takes one raw string and reclassifies from scratch every call — multi-turn state only exists in the interactive tools built on top of it (`context_guardrail/interactive.py`).

## Skills & techniques demonstrated

**LLM systems architecture**
- Designed, shipped, and then substantially simplified a multi-stage LLM pipeline in response to a concretely observed failure mode (the back-pain over-blocking case under "Why this shape"), rather than accreting complexity indefinitely.
- Pluggable domain abstraction (`DomainSchema`/`FieldSpec`) so a single gating mechanism generalizes across domains instead of hardcoding medical-specific logic into the core.
- Dependency-injected LLM backend (`LLMClient` protocol) supporting independent models for the gate vs. the answerer, enabling a "cheap gate, strong answerer" configuration as a first-class thing to test, not a hack.

**Prompt engineering & structured generation**
- JSON-schema-constrained decoding (`format=`) for reliable field extraction, with an explicit fail-closed policy (unparseable → the most conservative outcome) rather than assuming well-formed output.
- System-prompt design for a binary coarseness trade-off (`answer_scope: general | personal`) deliberately chosen over a finer-grained classification after reasoning about a small model's known reliability limits.

**Evaluation methodology & experimental design**
- Designed a 3-way controlled comparison (raw LLM / prompted LLM / gated pipeline) isolating one variable — enforcement vs. instruction — while holding the underlying model constant.
- Built an LLM-as-judge rubric (5 criteria, 0–2 scale) with explicit scoring guidance for edge cases (grading a clarifying question on its own terms rather than penalizing it for not being an answer).
- Identified and corrected for same-vendor judge bias by cross-checking a GPT-4o judge against an independently-run, context-free Claude agent given only the rubric and anonymized outputs — and reported where the two disagreed rather than only where they agreed.
- Audited the project's own eval-case labels rather than treating them as ground truth: traced every functional-test failure to a root cause (real bug vs. mislabeled test case vs. self-flagged ambiguous case vs. classifier noise on rerun), and corrected a test label after empirically verifying it against a live external API.
- Named, rather than hid, every methodological weakness that survives the above (self-labeled cases with no independent labeler, response-shape identity leakage in "blind" grading, small sample size) directly in this document.

**Software engineering**
- Python package structured around single-responsibility modules (`context_guardrail/`, `main_llm/`, `medical/`), with a deterministic, non-LLM safety check (`ingredient_safety.py`) kept clearly separate from LLM-judgment-based logic.
- Test suite with unit tests fully mocked against the `LLMClient` protocol (no network/model dependency) plus a separate `integration` marker for real-backend tests.
- Git history managed as a sequence of self-consistent, individually-green commits across a large (~55-file) architectural rewrite, rather than one opaque commit.

**Integration & tooling**
- Raw `httpx` clients for OpenAI and Ollama (no SDK dependency) with a shared interface, and a live public API integration (openFDA drug labels) for the one deterministic safety check.
- Streamlit GUI with explicit session-state-driven stages and a human-approval checkpoint before the generation call fires.

**How to use this project**: see "Quickstart" above for setup and every entry point (CLI, interactive builder, GUI, tests, eval scripts) in one place.
