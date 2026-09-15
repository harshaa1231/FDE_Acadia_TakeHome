"""Persists the result of ingesting a dataset - its schema profile and
resolved concept map - so question jobs and the API can reuse them without
re-profiling the file on every question."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.ingestion.models import SchemaProfile
from app.semantic.models import ConceptMap


@dataclass
class DatasetRecord:
    id: str
    table_name: str
    db_path: str
    original_filename: str
    row_count: int
    rows_skipped: int
    schema_profile: SchemaProfile
    concept_map: ConceptMap
    created_at: float


class DatasetStore:
    def __init__(self, db_path: str | None = None):
        path = Path(db_path or settings.data_dir) / "jobs.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                db_path TEXT NOT NULL,
                original_filename TEXT,
                row_count INTEGER NOT NULL,
                rows_skipped INTEGER NOT NULL,
                schema_profile TEXT NOT NULL,
                concept_map TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def save(self, record: DatasetRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO datasets "
            "(id, table_name, db_path, original_filename, row_count, rows_skipped, schema_profile, concept_map, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                record.id, record.table_name, record.db_path, record.original_filename,
                record.row_count, record.rows_skipped,
                record.schema_profile.model_dump_json(), record.concept_map.model_dump_json(),
                record.created_at,
            ),
        )
        self._conn.commit()

    def get(self, dataset_id: str) -> DatasetRecord | None:
        row = self._conn.execute(
            "SELECT id, table_name, db_path, original_filename, row_count, rows_skipped, "
            "schema_profile, concept_map, created_at FROM datasets WHERE id = ?",
            (dataset_id,),
        ).fetchone()
        if not row:
            return None
        (id_, table_name, db_path, original_filename, row_count, rows_skipped,
         schema_profile_json, concept_map_json, created_at) = row
        return DatasetRecord(
            id=id_, table_name=table_name, db_path=db_path, original_filename=original_filename,
            row_count=row_count, rows_skipped=rows_skipped,
            schema_profile=SchemaProfile.model_validate_json(schema_profile_json),
            concept_map=ConceptMap.model_validate_json(concept_map_json),
            created_at=created_at,
        )

    def delete(self, dataset_id: str) -> None:
        self._conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        self._conn.commit()


_store: DatasetStore | None = None


def get_dataset_store() -> DatasetStore:
    global _store
    if _store is None:
        _store = DatasetStore()
    return _store
