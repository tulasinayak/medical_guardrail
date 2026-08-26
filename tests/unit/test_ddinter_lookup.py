import sqlite3

import pytest

from medical_guardrails.stage2_generate.ddinter_lookup import DDInterLookup


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "ddinter.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE interactions (drug_a TEXT, drug_b TEXT, level TEXT)")
    conn.execute(
        "INSERT INTO interactions VALUES (?, ?, ?)", ("warfarin", "ibuprofen", "Major")
    )
    conn.commit()
    conn.close()
    return path


def test_finds_interaction_in_stored_order(db_path):
    lookup = DDInterLookup(db_path)
    assert lookup.find_interaction("warfarin", "ibuprofen") == "Major"


def test_finds_interaction_in_reversed_order(db_path):
    lookup = DDInterLookup(db_path)
    assert lookup.find_interaction("ibuprofen", "warfarin") == "Major"


def test_is_case_insensitive(db_path):
    lookup = DDInterLookup(db_path)
    assert lookup.find_interaction("Warfarin", "IBUPROFEN") == "Major"


def test_returns_none_for_unknown_pair(db_path):
    lookup = DDInterLookup(db_path)
    assert lookup.find_interaction("aspirin", "metformin") is None


def test_returns_none_when_db_missing(tmp_path):
    lookup = DDInterLookup(tmp_path / "does_not_exist.sqlite")
    assert lookup.find_interaction("warfarin", "ibuprofen") is None
