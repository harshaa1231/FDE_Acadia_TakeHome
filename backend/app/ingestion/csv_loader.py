"""Loads an arbitrary CSV into a dedicated, queryable DuckDB store.

This is the "repeatable step, not a manual one" the brief asks for: every
ingest goes through this same function regardless of the file's shape.
Nothing here assumes particular column names - DuckDB's own CSV sniffer
(delimiter, quoting, header, per-column types) does the schema detection.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import duckdb

from app.core.config import settings

logger = logging.getLogger(__name__)

TABLE_NAME = "transactions"


@dataclass
class IngestResult:
    dataset_id: str
    db_path: str
    table_name: str
    row_count: int
    column_names: list[str]
    rows_skipped: int


def new_dataset_id() -> str:
    return uuid.uuid4().hex[:12]


def dataset_db_path(dataset_id: str) -> Path:
    root = Path(settings.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{dataset_id}.duckdb"


def _count_csv_data_lines(csv_path: str) -> int | None:
    """Best-effort line count for partial-failure transparency. Not used
    for anything load-bearing - if it fails (odd encoding, embedded
    newlines in quoted fields) we simply skip the comparison."""
    try:
        with open(csv_path, "rb") as f:
            lines = sum(1 for _ in f)
        return max(lines - 1, 0)  # minus header
    except OSError:
        return None


def load_csv(dataset_id: str, csv_path: str) -> IngestResult:
    """Load `csv_path` into a fresh DuckDB file dedicated to `dataset_id`.

    Raises on total failure. The caller is responsible for marking the
    ingest job failed and calling `drop_dataset` so no partially-loaded
    dataset is left queryable (rollback semantics for the async job).
    """
    db_path = dataset_db_path(dataset_id)
    if db_path.exists():
        db_path.unlink()

    expected_rows = _count_csv_data_lines(csv_path)

    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            f"""
            CREATE TABLE {TABLE_NAME} AS
            SELECT * FROM read_csv_auto(
                ?,
                normalize_names = true,
                sample_size = -1,
                ignore_errors = true
            )
            """,
            [csv_path],
        )
        row_count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        if row_count == 0:
            raise ValueError(
                "The CSV produced zero usable rows. Check that it has a header "
                "row and at least one data row with a consistent delimiter."
            )
        column_names = [r[0] for r in con.execute(f"DESCRIBE {TABLE_NAME}").fetchall()]
    except Exception:
        con.close()
        db_path.unlink(missing_ok=True)
        raise
    con.close()

    rows_skipped = 0
    if expected_rows is not None and expected_rows > row_count:
        rows_skipped = expected_rows - row_count
        logger.warning(
            "dataset %s: %d of %d rows were skipped during load (malformed rows)",
            dataset_id,
            rows_skipped,
            expected_rows,
        )

    return IngestResult(
        dataset_id=dataset_id,
        db_path=str(db_path),
        table_name=TABLE_NAME,
        row_count=row_count,
        column_names=column_names,
        rows_skipped=rows_skipped,
    )


def open_dataset(dataset_id: str) -> duckdb.DuckDBPyConnection:
    db_path = dataset_db_path(dataset_id)
    if not db_path.exists():
        raise FileNotFoundError(f"No dataset store for '{dataset_id}'")
    return duckdb.connect(str(db_path), read_only=False)


def drop_dataset(dataset_id: str) -> None:
    dataset_db_path(dataset_id).unlink(missing_ok=True)
