"""Shared Pydantic models passed between guardrail stages.

`EvidenceChunk` is fully used starting with Stage 2. `StructuredQuery` and
`Claim` define the shape the later Stage 1 (slot-filling) and Stage 3
(claim verification) sessions will produce/consume, so the object model is
settled once rather than re-negotiated per stage; neither is wired into any
pipeline yet.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class QueryType(str, Enum):
    DRUG_INTERACTION = "drug_interaction"
    DOSAGE = "dosage"
    SYMPTOM = "symptom"
    HOME_REMEDY = "home_remedy"
    GENERAL_INFO = "general_info"


class StructuredQuery(BaseModel):
    """Produced by Stage 1. Fields are mostly optional here because which
    ones are *required* depends on `query_type`, and that required-fields
    schema is Stage 1's job to enforce, not this model's.

    For the list-valued fields (allergies, current_medications,
    existing_conditions): `None` means the user was never asked / never
    said anything on the topic, while `[]` means they were asked (or
    volunteered) and explicitly said "none". This distinction is the whole
    point of the slot-filling gate -- collapsing both to `[]` is exactly
    the allergy-omission failure mode Stage 1 exists to catch, since an
    unset field and a confirmed "no allergies" would otherwise be
    indistinguishable."""

    raw_text: str
    query_type: QueryType
    drug_names: list[str] = []
    allergies: list[str] | None = None
    current_medications: list[str] | None = None
    age_bracket: str | None = None
    pregnancy_status: str | None = None
    symptom_duration: str | None = None
    symptom_severity: str | None = None
    existing_conditions: list[str] | None = None


EvidenceSource = str  # "rxnorm" | "openfda" | "ddinter"


class EvidenceChunk(BaseModel):
    """One retrieved fact, attributable to exactly one source, that
    Stage 2's generation is grounded in and Stage 3 will later verify
    claims against."""

    source: EvidenceSource
    drug_names: list[str]
    field_name: str
    text: str
    metadata: dict = {}


class ClaimVerdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


class Claim(BaseModel):
    """Produced by Stage 3 once it exists: one atomic factual assertion
    decomposed out of a draft response, plus its entailment verdict against
    the evidence it was checked against."""

    claim_text: str
    verdict: ClaimVerdict | None = None
    supporting_evidence: list[EvidenceChunk] = []
