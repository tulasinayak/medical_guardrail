"""Shared Pydantic models passed between the Context Guardrail and Main LLM
(see orchestrator.py)."""

from __future__ import annotations

from pydantic import BaseModel


class DomainQuery(BaseModel):
    """The Context Guardrail's output. `query_type` and `fields` are both
    domain-defined (see `context_guardrail/domain.py` and
    `context_guardrail/domains/medical.py`) -- this model itself doesn't
    know what a valid query type or field name is for any given domain;
    that's `required_fields.missing_fields`'s job, parameterized by a
    `DomainSchema`.

    `answer_scope` is domain-agnostic and lives at the top level rather
    than inside `fields`: it's a routing decision (does this question need
    personal context at all?), not an extracted fact about the user.
    "general" means required fields are skipped entirely regardless of
    query_type -- asking for someone's allergies to answer "what is
    ibuprofen?" would be exactly the unnecessary-questioning this field
    exists to avoid.

    For list-valued fields the domain marks `list_with_status` (e.g. this
    project's medical domain: allergies, current_medications,
    existing_conditions): `None` means the user was never asked / never
    said anything on the topic, while `[]` means they were asked (or
    volunteered) and explicitly said "none". This distinction is the whole
    point of the slot-filling gate -- collapsing both to `[]` is exactly
    the allergy-omission failure mode it exists to catch, since an unset
    field and a confirmed "no allergies" would otherwise be
    indistinguishable."""

    raw_text: str
    query_type: str
    answer_scope: str  # "general" | "personal"
    fields: dict[str, list[str] | str | None] = {}
