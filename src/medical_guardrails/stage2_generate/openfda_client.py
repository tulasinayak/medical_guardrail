"""openFDA Drug Label API client: pulls structured label text for a drug by
name. Free, no auth.

Label search is by name (openfda.substance_name, falling back to
openfda.generic_name), not by RxCUI: openFDA's `openfda.rxcui` field holds
product-level RxCUIs (specific formulations/NDCs), while RxNorm's plain
name lookup returns ingredient-level RxCUIs -- the two don't match, so
chaining "name -> RxNorm RxCUI -> openFDA by rxcui" silently returns nothing
for most ingredient names. Many drugs (esp. non-US or less common generics)
have no matching label at all -- that's an expected, non-error outcome, so
lookups return {} rather than raising.
"""

from __future__ import annotations

import httpx

LABEL_FIELDS = [
    "contraindications",
    "warnings",
    "boxed_warning",
    "drug_interactions",
    "active_ingredient",
    "inactive_ingredient",
]


class OpenFDAClient:
    def __init__(
        self, base_url: str = "https://api.fda.gov/drug/label.json", timeout: float = 15.0
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def get_label_fields(self, drug_name: str) -> dict[str, list[str]]:
        """Returns a dict of field_name -> list of text values present on the
        first matching label for this drug name. Missing fields are simply
        absent from the dict. Returns {} if no label matches at all."""
        for search_field in ("substance_name", "generic_name"):
            label = self._search(search_field, drug_name)
            if label is not None:
                return {field: label[field] for field in LABEL_FIELDS if field in label}
        return {}

    def _search(self, search_field: str, drug_name: str) -> dict | None:
        response = httpx.get(
            self.base_url,
            params={"search": f'openfda.{search_field}:"{drug_name}"', "limit": 1},
            timeout=self.timeout,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()

        results = response.json().get("results", [])
        return results[0] if results else None
