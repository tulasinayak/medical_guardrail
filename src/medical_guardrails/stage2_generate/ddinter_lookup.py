"""Offline pairwise drug-drug interaction lookup against the local SQLite
dump built by data/ddinter/build_ddinter_db.py. Interactions are undirected
in the source data, so both (a, b) and (b, a) orderings are checked.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class DDInterLookup:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def find_interaction(self, drug_a: str, drug_b: str) -> str | None:
        """Returns the severity Level ("Major"/"Moderate"/"Minor") for this
        drug pair, or None if the pair isn't in the dataset or the local
        database hasn't been built yet."""
        if not self.db_path.exists():
            return None

        a, b = drug_a.strip().lower(), drug_b.strip().lower()
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                """
                SELECT level FROM interactions
                WHERE (drug_a = ? AND drug_b = ?) OR (drug_a = ? AND drug_b = ?)
                LIMIT 1
                """,
                (a, b, b, a),
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()
