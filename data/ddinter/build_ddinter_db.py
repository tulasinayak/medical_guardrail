"""Builds a local SQLite dump of DDInter's public drug-drug interaction
data, so Stage 2 can do offline pairwise interaction lookups instead of
depending on a live query API (DDInter doesn't expose one).

Source: DDInter's public per-letter CSV bulk export, the same one used by
the open-source community DDInter parser (github.com/mnarayan1/DDInter) --
http://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_{A..Z}.csv
Columns: DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level.

Note: this bulk export only carries severity (`Level`: Major/Moderate/Minor),
not the mechanism/management free text DDInter shows on its per-pair detail
pages -- that would require scraping individual pair pages, which is out of
scope here. `EvidenceChunk`s built from this data will carry severity only.

Run: python data/ddinter/build_ddinter_db.py
"""

from __future__ import annotations

import csv
import io
import sqlite3
import string
from pathlib import Path

import httpx

BASE_URL = "http://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_{letter}.csv"
DB_PATH = Path(__file__).resolve().parent / "ddinter.sqlite"


def _normalize(name: str) -> str:
    return name.strip().lower()


def fetch_letter_csv(letter: str, timeout: float = 30.0) -> list[dict[str, str]]:
    response = httpx.get(BASE_URL.format(letter=letter), timeout=timeout, follow_redirects=True)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text))
    return list(reader)


def build_database(db_path: Path = DB_PATH) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS interactions")
        conn.execute(
            """
            CREATE TABLE interactions (
                drug_a TEXT NOT NULL,
                drug_b TEXT NOT NULL,
                level TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX idx_drug_a ON interactions(drug_a)")
        conn.execute("CREATE INDEX idx_drug_b ON interactions(drug_b)")

        total = 0
        for letter in string.ascii_uppercase:
            rows = fetch_letter_csv(letter)
            conn.executemany(
                "INSERT INTO interactions (drug_a, drug_b, level) VALUES (?, ?, ?)",
                [
                    (_normalize(row["Drug_A"]), _normalize(row["Drug_B"]), row["Level"])
                    for row in rows
                ],
            )
            total += len(rows)
        conn.commit()
        return total
    finally:
        conn.close()


if __name__ == "__main__":
    count = build_database()
    print(f"Loaded {count} interaction records into {DB_PATH}")
