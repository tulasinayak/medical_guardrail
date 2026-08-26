"""Wires all three guardrail stages into one pipeline, mirroring
pii_guardrails.orchestrator.GuardrailedChat: Stage 1 (slot-filling) gates
whether Stage 2 (retrieval + grounded generation) runs at all, and Stage 3
(claim + ingredient verification) gates what Stage 2 produces before it
reaches the caller.

Scope note: evidence retrieval is drug-name-centric (RxNorm/openFDA/
DDInter). `structured_query.drug_names` is passed to retrieval regardless
of query_type, since retrieval itself is type-agnostic -- for query types
where the user didn't name a drug (most symptom/home_remedy/general_info
queries), evidence will simply be empty and Stage 2's system prompt
correctly falls back to "not in my sources" rather than answering from
parametric memory. This project does not yet implement a symptom- or
recipe-specific evidence source -- see README "Known limitations".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from medical_guardrails.common.schemas import EvidenceChunk, StructuredQuery
from medical_guardrails.config import Settings
from medical_guardrails.llm.base import LLMClient
from medical_guardrails.llm.factory import build_llm_client
from medical_guardrails.stage1_slotfill.gate import slot_fill_gate
from medical_guardrails.stage2_generate.ddinter_lookup import DDInterLookup
from medical_guardrails.stage2_generate.generation import generate_grounded_response
from medical_guardrails.stage2_generate.openfda_client import OpenFDAClient
from medical_guardrails.stage2_generate.retrieval import retrieve_evidence
from medical_guardrails.stage2_generate.rxnorm_client import RxNormClient
from medical_guardrails.stage3_verify.verification import VerificationResult, verify_response

PipelineStatus = Literal["needs_clarification", "answered"]


@dataclass
class PipelineResult:
    status: PipelineStatus
    structured_query: StructuredQuery
    missing_fields: list[str] = field(default_factory=list)
    clarifying_question: str | None = None
    evidence: list[EvidenceChunk] = field(default_factory=list)
    draft_response: str | None = None
    verification: VerificationResult | None = None
    final_response: str | None = None


class MedicalGuardrailPipeline:
    """One instance is meant to be reused across queries -- the client
    objects are stateless/cheap, so nothing here needs per-query teardown."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.llm_client = llm_client or build_llm_client(self.settings)
        self.rxnorm_client = RxNormClient(self.settings.rxnorm_base_url, self.settings.http_timeout_seconds)
        self.openfda_client = OpenFDAClient(self.settings.openfda_base_url, self.settings.http_timeout_seconds)
        self.ddinter_lookup = DDInterLookup(self.settings.ddinter_db_path)

    def process_query(self, raw_text: str) -> PipelineResult:
        gate_result = slot_fill_gate(raw_text, self.llm_client)

        if gate_result.status == "needs_clarification":
            return PipelineResult(
                status="needs_clarification",
                structured_query=gate_result.structured_query,
                missing_fields=gate_result.missing,
                clarifying_question=gate_result.clarifying_question,
            )

        query = gate_result.structured_query
        evidence = retrieve_evidence(
            drug_names=query.drug_names,
            rxnorm_client=self.rxnorm_client,
            openfda_client=self.openfda_client,
            ddinter_lookup=self.ddinter_lookup,
        )

        draft = generate_grounded_response(query.raw_text, evidence, self.llm_client)
        verification = verify_response(draft, evidence, query.allergies or [], self.llm_client)

        return PipelineResult(
            status="answered",
            structured_query=query,
            evidence=evidence,
            draft_response=draft,
            verification=verification,
            final_response=verification.final_response,
        )
