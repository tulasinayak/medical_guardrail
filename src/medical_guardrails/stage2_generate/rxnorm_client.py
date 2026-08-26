"""RxNorm (NIH/NLM) client: resolves a free-text drug name (brand or
generic) to an RxCUI identifier. Free, no auth. The exact-match endpoint is
tried first; approximate match is a fallback for typos or less common brand
names, since exact match returns nothing on a near-miss rather than a
best-effort guess.
"""

from __future__ import annotations

import httpx


class RxNormNotFoundError(Exception):
    """Raised when no RxCUI could be resolved for a given drug name."""


class RxNormClient:
    def __init__(self, base_url: str = "https://rxnav.nlm.nih.gov/REST", timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def find_rxcui(self, drug_name: str) -> str:
        """Returns the first matching RxCUI for `drug_name`.
        Raises RxNormNotFoundError if nothing matches, exactly or approximately."""
        rxcui = self._exact_match(drug_name)
        if rxcui is not None:
            return rxcui

        rxcui = self._approximate_match(drug_name)
        if rxcui is not None:
            return rxcui

        raise RxNormNotFoundError(f"No RxCUI found for drug name: {drug_name!r}")

    def _exact_match(self, drug_name: str) -> str | None:
        response = httpx.get(
            f"{self.base_url}/rxcui.json",
            params={"name": drug_name},
            timeout=self.timeout,
        )
        response.raise_for_status()
        ids = response.json().get("idGroup", {}).get("rxnormId", [])
        return ids[0] if ids else None

    def _approximate_match(self, drug_name: str) -> str | None:
        response = httpx.get(
            f"{self.base_url}/approximateTerm.json",
            params={"term": drug_name, "maxEntries": 1},
            timeout=self.timeout,
        )
        response.raise_for_status()
        candidates = response.json().get("approximateGroup", {}).get("candidate", [])
        return candidates[0]["rxcui"] if candidates else None

    def get_canonical_name(self, rxcui: str) -> str | None:
        """Resolves an RxCUI back to its canonical RxNorm name -- used to
        normalize a brand name or typo (e.g. "tylenol", "warfarn") to the
        generic ingredient name that openFDA's label search expects."""
        response = httpx.get(
            f"{self.base_url}/rxcui/{rxcui}/property.json",
            params={"propName": "RxNorm Name"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        concepts = response.json().get("propConceptGroup", {}).get("propConcept", [])
        return concepts[0]["propValue"] if concepts else None
