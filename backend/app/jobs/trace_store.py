"""Per-job pipeline trace: the LLM-observability surface. Every stage a
question job goes through (concept map used, prompt sent, raw LLM output,
guardrail verdicts, generated SQL, execution outcome, latency) is appended
here under the job's id, so `GET /jobs/{id}/trace` can show exactly why an
answer - or a refusal - happened, instead of needing to re-derive it live.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from app.core.config import settings


class TraceStore:
    def __init__(self, db_path: str | None = None):
        path = Path(db_path or settings.data_dir) / "jobs.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS traces (job_id TEXT PRIMARY KEY, stages TEXT NOT NULL)"
        )
        self._conn.commit()

    def init_trace(self, job_id: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO traces (job_id, stages) VALUES (?, ?)", (job_id, json.dumps([]))
        )
        self._conn.commit()

    def append_stage(self, job_id: str, name: str, data: dict) -> None:
        row = self._conn.execute("SELECT stages FROM traces WHERE job_id=?", (job_id,)).fetchone()
        stages = json.loads(row[0]) if row else []
        stages.append({"stage": name, "ts": time.time(), **data})
        self._conn.execute(
            "INSERT OR REPLACE INTO traces (job_id, stages) VALUES (?, ?)",
            (job_id, json.dumps(stages, default=str)),
        )
        self._conn.commit()

    def get_trace(self, job_id: str) -> list[dict]:
        row = self._conn.execute("SELECT stages FROM traces WHERE job_id=?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else []


_store: TraceStore | None = None


def get_trace_store() -> TraceStore:
    global _store
    if _store is None:
        _store = TraceStore()
    return _store
