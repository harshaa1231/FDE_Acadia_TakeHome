"""Tests for the in-process async job queue - the concurrency and
partial-failure claims documented in app/jobs/queue.py and docs/DESIGN.md
are architectural bets the brief explicitly asks us to be ready to defend
live ("concurrent load and partial failure"). These tests prove them
rather than just narrate them: a full queue backs off with a structured
error instead of growing unbounded, one job's failure never blocks or
crashes another, and the worker pool genuinely runs jobs concurrently
rather than serializing them.
"""
import asyncio
import time

import pytest

from app.core.errors import QueueFullError
from app.jobs.models import Job, JobStatus, JobType
from app.jobs.queue import JobQueue
from app.jobs.store import JobStore


def make_queue(tmp_path, concurrency=2, max_size=10):
    store = JobStore(db_path=str(tmp_path))
    return JobQueue(store=store, concurrency=concurrency, max_size=max_size), store


def test_full_queue_raises_and_marks_the_rejected_job_failed(tmp_path):
    """Backpressure, not unbounded growth: with no workers draining it, a
    queue at capacity must reject the next submission immediately rather
    than blocking or silently queuing forever."""
    queue, store = make_queue(tmp_path, concurrency=1, max_size=1)
    filler = Job.new(JobType.INGEST, {})
    queue.submit(filler)  # fills the one slot; no worker running to drain it

    overflow = Job.new(JobType.INGEST, {})
    with pytest.raises(QueueFullError):
        queue.submit(overflow)

    stored = store.get(overflow.id)
    assert stored.status == JobStatus.FAILED
    assert stored.error["code"] == "queue_full"


@pytest.mark.asyncio
async def test_one_job_failure_does_not_affect_another_job(tmp_path):
    """A handler exception must be caught per-job - it must not crash the
    worker or leave a sibling job stuck. One job raises, one succeeds;
    both must reach a correct terminal status independently."""
    queue, store = make_queue(tmp_path, concurrency=2)

    async def handler(job: Job) -> dict:
        if job.payload.get("should_fail"):
            raise ValueError("simulated handler failure")
        return {"ok": True}

    queue.register_handler(JobType.INGEST, handler)
    await queue.start()
    try:
        failing = Job.new(JobType.INGEST, {"should_fail": True})
        succeeding = Job.new(JobType.INGEST, {"should_fail": False})
        queue.submit(failing)
        queue.submit(succeeding)

        for _ in range(50):
            if store.get(failing.id).status != JobStatus.QUEUED and store.get(succeeding.id).status not in (
                JobStatus.QUEUED, JobStatus.RUNNING,
            ):
                break
            await asyncio.sleep(0.02)

        failed_job = store.get(failing.id)
        succeeded_job = store.get(succeeding.id)
        assert failed_job.status == JobStatus.FAILED
        assert failed_job.error["code"] == "job_failed"
        assert "simulated handler failure" in failed_job.error["message"]
        assert succeeded_job.status == JobStatus.SUCCEEDED
        assert succeeded_job.result == {"ok": True}
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_worker_pool_runs_jobs_concurrently_not_serially(tmp_path):
    """With concurrency=3 and three jobs that each sleep 0.2s, total wall
    time should look like one job's duration, not three stacked up - proof
    the pool actually parallelizes rather than processing one at a time."""
    queue, store = make_queue(tmp_path, concurrency=3)

    async def slow_handler(job: Job) -> dict:
        await asyncio.sleep(0.2)
        return {"job_id": job.id}

    queue.register_handler(JobType.INGEST, slow_handler)
    await queue.start()
    try:
        jobs = [Job.new(JobType.INGEST, {}) for _ in range(3)]
        start = time.monotonic()
        for j in jobs:
            queue.submit(j)

        for _ in range(100):
            statuses = [store.get(j.id).status for j in jobs]
            if all(s == JobStatus.SUCCEEDED for s in statuses):
                break
            await asyncio.sleep(0.02)
        elapsed = time.monotonic() - start

        assert all(store.get(j.id).status == JobStatus.SUCCEEDED for j in jobs)
        # Serial execution would take >=0.6s; concurrent execution should
        # finish well under that even with scheduling overhead.
        assert elapsed < 0.5
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_missing_handler_fails_job_without_crashing_worker(tmp_path):
    """A job type with no registered handler must fail cleanly (and leave
    the worker able to pick up the next job), not raise into the worker
    loop and take it down."""
    queue, store = make_queue(tmp_path, concurrency=1)
    await queue.start()
    try:
        orphan = Job.new(JobType.QUESTION, {})  # no handler registered for QUESTION
        queue.submit(orphan)

        for _ in range(50):
            if store.get(orphan.id).status != JobStatus.QUEUED:
                break
            await asyncio.sleep(0.02)

        assert store.get(orphan.id).status == JobStatus.FAILED

        # Worker must still be alive and able to process a subsequent job.
        queue.register_handler(JobType.INGEST, lambda job: _ok())
        follow_up = Job.new(JobType.INGEST, {})
        queue.submit(follow_up)
        for _ in range(50):
            if store.get(follow_up.id).status not in (JobStatus.QUEUED, JobStatus.RUNNING):
                break
            await asyncio.sleep(0.02)
        assert store.get(follow_up.id).status == JobStatus.SUCCEEDED
    finally:
        await queue.stop()


async def _ok() -> dict:
    return {"ok": True}
