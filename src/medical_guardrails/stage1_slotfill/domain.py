"""Generic Stage 1 domain schema.

The slot-filling *mechanism* -- classify the raw text into a query type,
look up which fields that type requires, ask about whatever's still
missing -- doesn't depend on medicine at all. Only the concrete query
types/fields/questions do. A `DomainSchema` captures exactly that
domain-specific content; `classifier.py`, `required_fields.py`, and
`gate.py` are written against this generic shape and default to
`domains/medical.py`'s `MEDICAL_DOMAIN`, but a different domain could be
passed in instead without touching any of that mechanism code.

Stages 2/3 (retrieval, ingredient checking) are not part of this
generalization -- they're inherently drug/evidence-specific already (see
orchestrator.py's scope note), so only Stage 1's schema/gate is
domain-pluggable today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FieldKind = Literal["list_with_status", "list_plain", "scalar"]


@dataclass(frozen=True)
class FieldSpec:
    """One field the domain's extraction schema knows about.

    `list_with_status` fields get a companion NOT_MENTIONED/STATED_NONE/
    STATED_PRESENT tri-state (see classifier.py) so "explicitly said none"
    and "never came up" stay distinguishable -- that distinction is the
    whole reason Stage 1 exists as a gate rather than a plain extractor.
    `list_plain` is a bare array with no tri-state (this domain's
    `drug_names` is the only current example -- "not asked yet" and
    "asked, named none" aren't meaningfully different for it). `scalar`
    is a nullable string.
    """

    name: str
    kind: FieldKind
    clarifying_question: str


@dataclass(frozen=True)
class DomainSchema:
    name: str
    query_types: list[str]
    fields: dict[str, FieldSpec]
    required_fields: dict[str, list[str]]  # query_type -> field names
    fail_closed_query_type: str
    extraction_system_prompt: str
