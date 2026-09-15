from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import datasets, jobs, questions
from app.core.config import settings
from app.core.errors import AppError, InternalError
from app.core.logging import configure_logging
from app.jobs.handlers import ingest_handler, question_handler
from app.jobs.models import JobType
from app.jobs.queue import get_job_queue

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    queue = get_job_queue()
    queue.register_handler(JobType.INGEST, ingest_handler)
    queue.register_handler(JobType.QUESTION, question_handler)
    await queue.start()
    yield
    await queue.stop()


app = FastAPI(title="Natural Language Insights Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(status_code=exc.status_code, content=exc.to_body())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # FastAPI's own request-body validation (e.g. Pydantic model checks)
    # bypasses AppError entirely by default - without this handler it
    # returns its own {"detail": [...]} shape, breaking the "every error is
    # a structured {error: {code, message}} envelope" guarantee.
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    message = f"{field}: {first.get('msg')}" if field else (first.get("msg") or "Invalid request.")
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "validation_error", "message": message, "details": {"errors": exc.errors()}}},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    # Never leak a stack trace or internal exception message to the caller.
    import logging

    logging.getLogger(__name__).exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content=InternalError().to_body())


app.include_router(datasets.router)
app.include_router(jobs.router)
app.include_router(questions.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
