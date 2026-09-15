from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.errors import NotFoundError
from app.jobs.models import JobStatus
from app.jobs.store import get_job_store
from app.jobs.trace_store import get_trace_store

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobStatusResponse(BaseModel):
    job_id: str
    type: str
    status: str
    created_at: float
    updated_at: float
    error: dict | None = None


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    job = get_job_store().get(job_id)
    if job is None:
        raise NotFoundError(f"No job '{job_id}'")
    return JobStatusResponse(**job.to_public_dict())


@router.get("/{job_id}/result")
async def get_job_result(job_id: str):
    job = get_job_store().get(job_id)
    if job is None:
        raise NotFoundError(f"No job '{job_id}'")
    if job.status != JobStatus.SUCCEEDED:
        raise NotFoundError(f"Job '{job_id}' has not succeeded (status: {job.status.value}); no result available yet.")
    return job.result


@router.get("/{job_id}/trace")
async def get_job_trace(job_id: str):
    job = get_job_store().get(job_id)
    if job is None:
        raise NotFoundError(f"No job '{job_id}'")
    return {"job_id": job_id, "stages": get_trace_store().get_trace(job_id)}
