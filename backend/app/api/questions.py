from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.errors import NotFoundError, ValidationError
from app.ingestion.dataset_store import get_dataset_store
from app.jobs.models import Job, JobType
from app.jobs.queue import get_job_queue

router = APIRouter(prefix="/datasets", tags=["questions"])


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class JobSubmitted(BaseModel):
    job_id: str
    status: str


@router.post("/{dataset_id}/questions", status_code=202, response_model=JobSubmitted)
async def ask_question(dataset_id: str, body: QuestionRequest):
    if not body.question.strip():
        raise ValidationError("question must not be empty.")
    if get_dataset_store().get(dataset_id) is None:
        raise NotFoundError(f"No dataset '{dataset_id}' (it may still be ingesting, or ingestion failed).")

    job = Job.new(JobType.QUESTION, {"dataset_id": dataset_id, "question": body.question.strip()})
    get_job_queue().submit(job)
    return JobSubmitted(job_id=job.id, status=job.status.value)
