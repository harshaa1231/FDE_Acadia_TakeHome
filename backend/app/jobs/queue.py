"""Lightweight in-process async job queue: a bounded asyncio worker pool.

This is the deliberate scope tradeoff documented in the design doc - no
Redis/Celery, so ingestion and question-answering stay non-blocking with
zero extra infrastructure, at the cost of not surviving a process restart
mid-job and not scaling past one process. Concurrency and partial-failure
behavior:

- Bounded worker pool (JOB_WORKER_CONCURRENCY) + bounded queue
  (JOB_QUEUE_MAX_SIZE): backpressure is explicit - a full queue raises
  QueueFullError (-> HTTP 429) rather than growing unbounded.
- Every job's status transition (queued -> running -> succeeded/failed) is
  persisted before/after the handler runs, so `GET /jobs/{id}` always
  reflects real state, never "stuck".
- A handler exception is caught per-job and turned into a structured
  `failed` status with a mapped error code - one job's failure never takes
  down a worker or blocks the rest of the queue.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from app.core.config import settings
from app.core.errors import QueueFullError
from app.jobs.models import Job, JobStatus, JobType
from app.jobs.store import JobStore, get_job_store

logger = logging.getLogger(__name__)

JobHandler = Callable[[Job], Awaitable[dict]]


class JobQueue:
    def __init__(self, store: JobStore | None = None, concurrency: int | None = None, max_size: int | None = None):
        self._store = store or get_job_store()
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_size or settings.job_queue_max_size)
        self._concurrency = concurrency or settings.job_worker_concurrency
        self._handlers: dict[JobType, JobHandler] = {}
        self._workers: list[asyncio.Task] = []
        self._started = False

    def register_handler(self, job_type: JobType, handler: JobHandler) -> None:
        self._handlers[job_type] = handler

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._workers = [asyncio.create_task(self._worker_loop(i)) for i in range(self._concurrency)]
        logger.info("job queue started with %d workers", self._concurrency)

    async def stop(self) -> None:
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        self._started = False

    def submit(self, job: Job) -> None:
        self._store.create(job)
        try:
            self._queue.put_nowait(job.id)
        except asyncio.QueueFull:
            self._store.update_status(
                job.id, JobStatus.FAILED,
                error={"code": "queue_full", "message": "Too many jobs in flight; try again shortly."},
            )
            raise QueueFullError("Job queue is at capacity; try again shortly.")

    async def _worker_loop(self, worker_id: int) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._run_job(job_id, worker_id)
            finally:
                self._queue.task_done()

    async def _run_job(self, job_id: str, worker_id: int) -> None:
        job = self._store.get(job_id)
        if job is None:
            logger.warning("worker %d: job %s vanished before it could run", worker_id, job_id)
            return

        handler = self._handlers.get(job.type)
        if handler is None:
            self._store.update_status(
                job.id, JobStatus.FAILED,
                error={"code": "internal_error", "message": f"No handler registered for job type {job.type}"},
            )
            return

        self._store.update_status(job.id, JobStatus.RUNNING)
        try:
            result = await handler(job)
            self._store.update_status(job.id, JobStatus.SUCCEEDED, result=result)
        except Exception as e:
            logger.exception("worker %d: job %s failed", worker_id, job.id)
            self._store.update_status(
                job.id, JobStatus.FAILED,
                error={"code": "job_failed", "message": str(e) or e.__class__.__name__},
            )


_queue: JobQueue | None = None


def get_job_queue() -> JobQueue:
    global _queue
    if _queue is None:
        _queue = JobQueue()
    return _queue
