"""Tests for csv_loader.py - mainly the defense-in-depth property of
open_dataset(): even if sql_guardrails ever had a gap that let a table
function through (exactly what happened once - see test_sql_guardrails.py
and docs/DESIGN.md), the DuckDB connection used to execute question SQL is
itself configured to refuse external file/network access, so the two
layers don't share a single point of failure.
"""
import duckdb
import pytest

from app.core.config import settings
from app.ingestion.csv_loader import load_csv, open_dataset


@pytest.fixture
def dataset_id(tmp_path, monkeypatch, tmp_path_factory):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    csv_path = tmp_path_factory.mktemp("upload") / "data.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n")
    result = load_csv("testds", str(csv_path))
    return result.dataset_id


def test_open_dataset_can_still_run_normal_queries(dataset_id):
    con = open_dataset(dataset_id)
    try:
        assert con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 2
    finally:
        con.close()


def test_open_dataset_refuses_external_file_access_at_the_engine_level(dataset_id, tmp_path):
    """The critical defense-in-depth test: simulate a guardrail gap by
    handing the connection a table-function query directly - bypassing
    sql_guardrails entirely, as if it had already let this through - and
    confirm DuckDB itself still refuses to execute it."""
    leak_target = tmp_path / "secret.txt"
    leak_target.write_text("this must never be readable via the app\n")

    con = open_dataset(dataset_id)
    try:
        with pytest.raises(duckdb.Error, match="disabled"):
            con.execute(f"SELECT * FROM read_csv('{leak_target}')").fetchall()
    finally:
        con.close()


def test_load_csv_itself_still_needs_and_retains_external_access(tmp_path, monkeypatch):
    """load_csv legitimately calls read_csv_auto on the uploaded file -
    that connection must NOT be locked down, or ingestion itself breaks.
    Only open_dataset (used for already-ingested data) is hardened."""
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    csv_path = tmp_path / "upload.csv"
    csv_path.write_text("x,y\n5,6\n")
    result = load_csv("testds2", str(csv_path))
    assert result.row_count == 1
