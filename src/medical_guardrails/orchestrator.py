"""Wires the two core stages together: the Context Guardrail decides
whether there's enough context to answer, then Main LLM generates the
answer directly from the original request plus whatever context was
gathered -- no retrieval, no separate verification stage. The medical
domain adds one narrow, deterministic safety check on top (ingredient/
allergy conflict via a single openFDA lookup) -- not a pipeline stage
every domain goes through, just this one domain's own post-check.

Scope note: this project used to ground answers in retrieved evidence
(RxNorm/openFDA/DDInter/MedlinePlus) and separately verify claims against
that evidence. Both were removed -- see README for why -- in favor of a
narrower, more honest story: ask enough questions before answering, then
let a capable model answer directly and flag its own uncertainty, rather
than claim the answer is "grounded" or "verified" against a source this
project cannot itself vouch for the correctness of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from medical_guardrails.common.schemas import DomainQuery
from medical_guardrails.config import Settings
from medical_guardrails.context_guardrail.domain import DomainSchema
from medical_guardrails.context_guardrail.domains.medical import MEDICAL_DOMAIN
from medical_guardrails.context_guardrail.gate import slot_fill_gate
from medical_guardrails.llm.base import LLMClient
from medical_guardrails.llm.factory import build_llm_client
from medical_guardrails.main_llm.generation import generate_answer
from medical_guardrails.medical.ingredient_safety import IngredientCheckResult, check_drug_allergy_conflicts
from medical_guardrails.medical.openfda_client import OpenFDAClient

PipelineStatus = Literal["needs_clarification", "answered"]

BLOCKED_INGREDIENT_MESSAGE = (
    "This response has been blocked: it involves an ingredient that matches one of your stated "
    "allergies. Please consult a pharmacist or doctor before taking this medication."
)


@dataclass
class PipelineResult:
    status: PipelineStatus
    structured_query: DomainQuery
    missing_fields: list[str] = field(default_factory=list)
    clarifying_question: str | None = None
    draft_response: str | None = None
    ingredient_check: IngredientCheckResult | None = None
    final_response: str | None = None


class MedicalGuardrailPipeline:
    """One instance is meant to be reused across queries -- the client
    objects are stateless/cheap, so nothing here needs per-query teardown.

    `guardrail_llm_client` and `main_llm_client` can be set independently
    -- e.g. a cheap local model for the Context Guardrail and a stronger
    hosted model for the actual answer -- so that configuration can be
    compared against using one model throughout. Both default to the same
    client when only `llm_client` is given."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
        guardrail_llm_client: LLMClient | None = None,
        main_llm_client: LLMClient | None = None,
        domain: DomainSchema = MEDICAL_DOMAIN,
    ) -> None:
        self.settings = settings or Settings()
        if guardrail_llm_client is None or main_llm_client is None:
            default_client = llm_client or build_llm_client(self.settings)
            guardrail_llm_client = guardrail_llm_client or default_client
            main_llm_client = main_llm_client or default_client
        self.guardrail_llm_client = guardrail_llm_client
        self.main_llm_client = main_llm_client
        self.llm_client = self.main_llm_client  # backward-compatible alias
        self.domain = domain
        self.openfda_client = OpenFDAClient(self.settings.openfda_base_url, self.settings.http_timeout_seconds)

    def process_query(self, raw_text: str) -> PipelineResult:
        gate_result = slot_fill_gate(raw_text, self.guardrail_llm_client, self.domain)

        if gate_result.status == "needs_clarification":
            return PipelineResult(
                status="needs_clarification",
                structured_query=gate_result.structured_query,
                missing_fields=gate_result.missing,
                clarifying_question=gate_result.clarifying_question,
            )

        query = gate_result.structured_query
        draft = generate_answer(query.raw_text, query.fields, [], self.main_llm_client)

        ingredient_result = None
        final_response = draft
        drug_names = query.fields.get("drug_names") or []
        allergies = query.fields.get("allergies") or []
        if self.domain.name == "medical" and drug_names and allergies:
            ingredient_result = check_drug_allergy_conflicts(drug_names, allergies, self.openfda_client)
            if ingredient_result.conflicts:
                conflict_list = "; ".join(ingredient_result.conflicts)
                final_response = f"{BLOCKED_INGREDIENT_MESSAGE} ({conflict_list})"

        return PipelineResult(
            status="answered",
            structured_query=query,
            draft_response=draft,
            ingredient_check=ingredient_result,
            final_response=final_response,
        )
