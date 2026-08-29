"""Shared Pydantic models passed between the three guardrail stages and the
orchestrator that ties them together (see orchestrator.py)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class DomainQuery(BaseModel):
    """Stage 1's output. `query_type` and `fields` are both domain-defined
    (see `stage1_slotfill/domain.py` and `stage1_slotfill/domains/medical.py`)
    -- this model itself doesn't know what a valid query type or field name
    is for any given domain; that's `required_fields.missing_fields`'s job,
    parameterized by a `DomainSchema`.

    For list-valued fields the domain marks `list_with_status` (e.g. this
    project's medical domain: allergies, current_medications,
    existing_conditions): `None` means the user was never asked / never
    said anything on the topic, while `[]` means they were asked (or
    volunteered) and explicitly said "none". This distinction is the whole
    point of the slot-filling gate -- collapsing both to `[]` is exactly
    the allergy-omission failure mode Stage 1 exists to catch, since an
    unset field and a confirmed "no allergies" would otherwise be
    indistinguishable."""

    raw_text: str
    query_type: str
    fields: dict[str, list[str] | str | None] = {}


EvidenceSource = str  # "openfda" | "ddinter" | "medlineplus" in practice today -- RxNorm is used
# only for identity resolution (name -> RxCUI -> canonical name) and never itself surfaced as an
# evidence chunk, so "rxnorm" doesn't currently occur as a value here.

EvidenceAuthority = Literal["regulatory", "curated_secondary"]


class EvidenceChunk(BaseModel):
    """One retrieved fact, attributable to exactly one source, that
    Stage 2's generation is grounded in and Stage 3 will later verify
    claims against.

    `authority` distinguishes an FDA label statement or an NLM/MedlinePlus
    health topic summary (both regulatory/government-authoritative) from a
    DDInter interaction classification (a curated but secondary database)
    -- the two aren't interchangeable evidence, even
    though both are currently treated as equally checkable by Stage 3. This
    tag doesn't yet drive any weighting/scoring logic; it exists so a future
    version of Stage 3 can reason about which kind of source backed a given
    claim instead of treating all evidence as one undifferentiated pool."""

    source: EvidenceSource
    authority: EvidenceAuthority
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
