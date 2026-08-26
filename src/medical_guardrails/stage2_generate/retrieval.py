"""Combines RxNorm identity resolution, openFDA label evidence, and local
DDInter pairwise interaction lookups into one flat, source-attributed
evidence list for a set of drug names.
"""

from __future__ import annotations

from itertools import combinations

from medical_guardrails.common.schemas import EvidenceChunk
from medical_guardrails.stage2_generate.ddinter_lookup import DDInterLookup
from medical_guardrails.stage2_generate.openfda_client import OpenFDAClient
from medical_guardrails.stage2_generate.rxnorm_client import RxNormClient, RxNormNotFoundError


def retrieve_evidence(
    drug_names: list[str],
    rxnorm_client: RxNormClient,
    openfda_client: OpenFDAClient,
    ddinter_lookup: DDInterLookup,
) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    rxcuis: dict[str, str] = {}

    for drug_name in drug_names:
        try:
            rxcuis[drug_name] = rxnorm_client.find_rxcui(drug_name)
        except RxNormNotFoundError:
            continue

    for drug_name, rxcui in rxcuis.items():
        canonical_name = rxnorm_client.get_canonical_name(rxcui) or drug_name
        label_fields = openfda_client.get_label_fields(canonical_name)
        for field_name, values in label_fields.items():
            for value in values:
                chunks.append(
                    EvidenceChunk(
                        source="openfda",
                        authority="regulatory",
                        drug_names=[drug_name],
                        field_name=field_name,
                        text=value,
                        metadata={"rxcui": rxcui},
                    )
                )

    for drug_a, drug_b in combinations(drug_names, 2):
        level = ddinter_lookup.find_interaction(drug_a, drug_b)
        if level is not None:
            chunks.append(
                EvidenceChunk(
                    source="ddinter",
                    authority="curated_secondary",
                    drug_names=[drug_a, drug_b],
                    field_name="interaction_severity",
                    text=f"{drug_a} and {drug_b} have a documented {level.lower()} interaction.",
                    metadata={"level": level},
                )
            )

    return chunks
