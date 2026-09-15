"""Job state persistence. SQLite (not the per-dataset DuckDB files) - job
records are small, frequent, single-row read/writes, which is SQLite's
sweet spot, and keeps job bookkeeping decoupled from dataset storage so a
dataset can be dropped without losing its job history.

Calls here are synchronous (sqlite3 is not async) and are used directly
from async code - each call is a local-disk, single-row operation on the
order of microseconds, so the brief event-loop block is an explicit,
reasonable tradeoff at this scale, not an oversight. The concurrency
section of the design doc calls this out as the first thing to change
(e.g. an async driver, or moving to Postgres) if throughput ever demanded it.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from app.core.config import settings
from app.jobs.models import Job, JobStatus, JobType


class JobStore:
    def __init__(self, db_path: str | None = None):
        path = Path(db_path or settings.data_dir) / "jobs.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                result TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def create(self, job: Job) -> None:
        self._conn.execute(
            "INSERT INTO jobs (id, type, status, payload, result, error, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                job.id, job.type.value, job.status.value, json.dumps(job.payload),
                None, None, job.created_at, job.updated_at,
            ),
        )
        self._conn.commit()

    def get(self, job_id: str) -> Job | None:
        row = self._conn.execute(
            "SELECT id, type, status, payload, result, error, created_at, updated_at "
            "FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        return self._row_to_job(row) if row else None

    def update_status(
        self, job_id: str, status: JobStatus, *, result: dict | None = None, error: dict | None = None
    ) -> None:
        now = time.time()
        self._conn.execute(
            "UPDATE jobs SET status=?, result=?, error=?, updated_at=? WHERE id=?",
            (
                status.value,
                json.dumps(result, default=str) if result is not None else None,
                json.dumps(error, default=str) if error is not None else None,
                now, job_id,
            ),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_job(row) -> Job:
        id_, type_, status, payload, result, error, created_at, updated_at = row
        return Job(
            id=id_, type=JobType(type_), status=JobStatus(status),
            payload=json.loads(payload),
            result=json.loads(result) if result else None,
            error=json.loads(error) if error else None,
            created_at=created_at, updated_at=updated_at,
        )


_store: JobStore | None = None


def get_job_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore()
    return _store
