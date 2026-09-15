from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.core.errors import NotFoundError, ValidationError
from app.ingestion.csv_loader import new_dataset_id
from app.ingestion.dataset_store import get_dataset_store
from app.jobs.models import Job, JobType
from app.jobs.queue import get_job_queue

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _upload_dir() -> Path:
    return Path(settings.data_dir) / "uploads"


class JobSubmitted(BaseModel):
    job_id: str
    status: str


class DatasetSummary(BaseModel):
    dataset_id: str
    original_filename: str
    row_count: int
    rows_skipped: int
    column_count: int
    roles: dict
    dimensions: list[dict]
    unmapped_columns: list[dict]
    created_at: float


@router.post("", status_code=202, response_model=JobSubmitted)
async def upload_dataset(file: UploadFile):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise ValidationError("Only .csv files are accepted.")

    upload_dir = _upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = new_dataset_id()
    dest = upload_dir / f"{dataset_id}.csv"

    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                out.close()
                os.remove(dest)
                raise ValidationError(f"File exceeds the {settings.max_upload_mb}MB upload limit.")
            out.write(chunk)

    if written == 0:
        os.remove(dest)
        raise ValidationError("Uploaded file is empty.")

    job = Job.new(
        JobType.INGEST,
        {"dataset_id": dataset_id, "csv_path": str(dest), "original_filename": file.filename},
    )
    get_job_queue().submit(job)
    return JobSubmitted(job_id=job.id, status=job.status.value)


@router.get("/{dataset_id}", response_model=DatasetSummary)
async def get_dataset(dataset_id: str):
    record = get_dataset_store().get(dataset_id)
    if record is None:
        raise NotFoundError(f"No dataset '{dataset_id}' (it may still be ingesting, or ingestion failed).")

    cmap = record.concept_map
    return DatasetSummary(
        dataset_id=record.id,
        original_filename=record.original_filename,
        row_count=record.row_count,
        rows_skipped=record.rows_skipped,
        column_count=len(record.schema_profile.columns),
        roles={
            name: {"available": r.source != "not_found", "expression": r.expression, "confidence": r.confidence}
            for name, r in cmap.roles.items()
        },
        dimensions=[{"column": d.column, "distinct_count": d.distinct_count, "sample_values": d.sample_values} for d in cmap.dimensions],
        unmapped_columns=[{"column": u.name, "type": u.duckdb_type} for u in cmap.unmapped],
        created_at=record.created_at,
    )
