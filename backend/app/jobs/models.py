from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobType(str, Enum):
    INGEST = "ingest"
    QUESTION = "question"


@dataclass
class Job:
    id: str
    type: JobType
    status: JobStatus
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @staticmethod
    def new(job_type: JobType, payload: dict[str, Any]) -> "Job":
        now = time.time()
        return Job(
            id=uuid.uuid4().hex[:16],
            type=job_type,
            status=JobStatus.QUEUED,
            payload=payload,
            created_at=now,
            updated_at=now,
        )

    def to_public_dict(self) -> dict[str, Any]:
        d = {
            "job_id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.status == JobStatus.FAILED and self.error:
            d["error"] = self.error
        return d
