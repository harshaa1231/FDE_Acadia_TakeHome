"""Job handlers: the actual work behind each job type, run by the queue's
worker pool. Kept separate from the API layer (which only submits jobs and
reports status) and from the pipeline modules themselves (ingestion,
semantic layer, nlq) so each stays independently testable.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from app.core.config import settings
from app.ingestion.csv_loader import drop_dataset, load_csv, open_dataset
from app.ingestion.dataset_store import DatasetRecord, get_dataset_store
from app.ingestion.schema_profiler import build_schema_profile
from app.jobs.models import Job
from app.jobs.trace_store import get_trace_store
from app.nlq.answer_synth import synthesize_answer
from app.nlq.answerability_gate import dataset_has_minimal_structure
from app.nlq.column_formats import infer_column_formats
from app.nlq.llm_client import get_llm_client
from app.nlq.prompt_builder import SYSTEM_PROMPT, build_retry_prompt, build_sql_prompt
from app.nlq.sql_guardrails import SqlGuardrailError, validate_and_finalize_sql
from app.semantic.concept_mapper import build_concept_map
from app.semantic.metric_registry import resolve_available_metrics

logger = logging.getLogger(__name__)


async def ingest_handler(job: Job) -> dict:
    dataset_id = job.payload["dataset_id"]
    csv_path = job.payload["csv_path"]
    original_filename = job.payload.get("original_filename", "")

    try:
        result = load_csv(dataset_id, csv_path)  # rolls back its own partial DuckDB file on failure

        con = open_dataset(dataset_id)
        try:
            profile = build_schema_profile(con, dataset_id, result.table_name)
        finally:
            con.close()

        llm = get_llm_client()
        concept_map = await build_concept_map(profile, llm, model=settings.fast_model)

        get_dataset_store().save(
            DatasetRecord(
                id=dataset_id, table_name=result.table_name, db_path=result.db_path,
                original_filename=original_filename, row_count=result.row_count,
                rows_skipped=result.rows_skipped, schema_profile=profile, concept_map=concept_map,
                created_at=time.time(),
            )
        )
    except Exception:
        # No partially-ingested, half-registered dataset should ever be
        # left queryable - drop the DuckDB file so a failed ingest leaves
        # nothing behind for the API to accidentally serve.
        drop_dataset(dataset_id)
        raise
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)  # the data now lives in the DuckDB file; the raw upload is redundant

    resolved_roles = [name for name in concept_map.roles if concept_map.has(name)]
    usage = getattr(llm, "usage", None)
    return {
        "dataset_id": dataset_id,
        "row_count": result.row_count,
        "rows_skipped": result.rows_skipped,
        "column_count": len(profile.columns),
        "resolved_roles": resolved_roles,
        "dimensions": [d.column for d in concept_map.dimensions],
        "usage": usage.to_dict() if usage else None,
    }


def _fetch_sample_rows(con, table_name: str, limit: int = 5) -> list[dict]:
    cur = con.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, r)) for r in cur.fetchall()]


def _tidy_value(v):
    """Rounds float results to 2dp before they reach the LLM or the API
    response. DuckDB SUM/AVG on floats routinely produces noise like
    267760.9999999998 - technically correct, unreadable to a human, and
    the answer-synthesis LLM is instructed to only repeat numbers verbatim
    from the result rows, so an unrounded value here leaks straight into
    the prose. Display-layer rounding only; the underlying data is
    untouched."""
    if isinstance(v, float):
        return round(v, 2)
    return v


async def _execute_with_timeout(con, sql: str) -> tuple[list[str], list[list]]:
    def _run():
        cur = con.execute(sql)
        columns = [d[0] for d in cur.description]
        rows = [[_tidy_value(v) for v in row] for row in cur.fetchall()]
        return columns, rows

    try:
        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=settings.sql_execution_timeout_s)
    except asyncio.TimeoutError:
        con.interrupt()
        raise RuntimeError(f"query exceeded {settings.sql_execution_timeout_s}s timeout")


async def _generate_and_validate(
    llm, model: str, user_prompt: str, table_name: str, known_columns: set[str], con,
    job_id: str, trace, attempt: int,
) -> dict:
    raw = await llm.complete_text(SYSTEM_PROMPT, user_prompt, model=model)
    trace.append_stage(job_id, f"sql_attempt_{attempt}", {"raw_response": raw})

    if raw.strip().upper().startswith("REFUSE"):
        reason = raw.split(":", 1)[1].strip() if ":" in raw else raw.strip()
        return {"status": "refused", "reason": reason}

    if raw.strip().upper() == "OVERVIEW":
        return {"status": "overview"}

    try:
        final_sql = validate_and_finalize_sql(raw, table_name, known_columns)
    except SqlGuardrailError as e:
        trace.append_stage(job_id, f"guardrail_rejected_{attempt}", {"reason": e.reason, "raw_sql": raw})
        return {"status": "error", "error": f"guardrail: {e.reason}", "raw_sql": raw}

    try:
        columns, rows = await _execute_with_timeout(con, final_sql)
    except Exception as e:
        trace.append_stage(job_id, f"execution_failed_{attempt}", {"error": str(e), "sql": final_sql})
        return {"status": "error", "error": f"execution: {e}", "raw_sql": raw}

    trace.append_stage(job_id, f"execution_succeeded_{attempt}", {"sql": final_sql, "row_count": len(rows)})
    return {"status": "ok", "sql": final_sql, "columns": columns, "rows": rows}


def _usage_dict(llm) -> dict | None:
    usage = getattr(llm, "usage", None)
    return usage.to_dict() if usage else None


def _dataset_overview_facts(con, dataset, cmap, metrics: list) -> list[tuple[str, object]]:
    """Builds a grounded fact table for an 'overview' question - no
    LLM-generated SQL involved, so no guardrail risk: row/column counts are
    already known, the date range comes from the validated event_time
    expression, and metric values come from the metric_registry's own
    pre-built, deterministic expressions (the same ones a specific-question
    answer would use)."""
    facts: list[tuple[str, object]] = [
        ("row_count", dataset.row_count),
        ("column_count", len(dataset.schema_profile.columns)),
    ]
    if dataset.rows_skipped:
        facts.append(("rows_skipped_as_malformed", dataset.rows_skipped))

    time_role = cmap.get("event_time")
    if time_role:
        try:
            lo, hi = con.execute(
                f"SELECT MIN({time_role.expression}), MAX({time_role.expression}) FROM {dataset.table_name}"
            ).fetchone()
            facts.append(("date_range", f"{lo} to {hi}"))
        except Exception:
            pass

    for name, r in cmap.roles.items():
        if r.source != "not_found":
            facts.append((f"concept: {name}", r.expression))

    for d in cmap.dimensions:
        facts.append((f"dimension: {d.column}", f"{d.distinct_count} distinct values, e.g. {d.sample_values[:3]}"))

    for m in metrics:
        if not m.available:
            continue
        try:
            value = con.execute(f"SELECT {m.sql_expression} FROM {dataset.table_name}").fetchone()[0]
            facts.append((f"metric: {m.name}", _tidy_value(value)))
        except Exception:
            continue

    return facts


async def question_handler(job: Job) -> dict:
    dataset_id = job.payload["dataset_id"]
    question = job.payload["question"]

    dataset = get_dataset_store().get(dataset_id)
    if dataset is None:
        raise RuntimeError(f"dataset '{dataset_id}' not found")

    trace = get_trace_store()
    trace.init_trace(job.id)
    trace.append_stage(job.id, "dataset_loaded", {"dataset_id": dataset_id, "row_count": dataset.row_count})

    cmap = dataset.concept_map
    ok, reason = dataset_has_minimal_structure(cmap)
    trace.append_stage(job.id, "answerability_gate", {"passed": ok, "reason": reason})
    if not ok:
        return {"status": "refused", "reason": reason}

    metrics = resolve_available_metrics(cmap)
    known_columns = {c.name for c in dataset.schema_profile.columns}
    llm = get_llm_client()

    con = open_dataset(dataset_id)
    try:
        sample_rows = _fetch_sample_rows(con, dataset.table_name)
        user_prompt = build_sql_prompt(dataset.table_name, cmap, metrics, sample_rows, question)

        attempt = await _generate_and_validate(
            llm, settings.sql_model, user_prompt, dataset.table_name, known_columns, con, job.id, trace, 1
        )
        if attempt["status"] == "error":
            retry_prompt = user_prompt + "\n\n" + build_retry_prompt(attempt.get("raw_sql", ""), attempt["error"])
            attempt = await _generate_and_validate(
                llm, settings.sql_model, retry_prompt, dataset.table_name, known_columns, con, job.id, trace, 2
            )
            if attempt["status"] == "error":
                reason = f"Could not produce valid SQL for this question: {attempt['error']}"
                trace.append_stage(job.id, "refused_after_retry", {"reason": reason})
                trace.append_stage(job.id, "usage", _usage_dict(llm) or {})
                return {"status": "refused", "reason": reason, "usage": _usage_dict(llm)}

        if attempt["status"] == "refused":
            trace.append_stage(job.id, "usage", _usage_dict(llm) or {})
            return {"status": "refused", "reason": attempt["reason"], "usage": _usage_dict(llm)}

        if attempt["status"] == "overview":
            facts = _dataset_overview_facts(con, dataset, cmap, metrics)
            trace.append_stage(job.id, "overview_facts", {"facts": facts})
            sql_note = "(dataset overview - not a single query; computed from schema profile and named metrics)"
            columns = ["fact", "value"]
            rows = [[k, v] for k, v in facts]
            answer = await synthesize_answer(llm, settings.fast_model, question, sql_note, columns, rows)
            trace.append_stage(job.id, "answer_synthesized", {"answer": answer})
            trace.append_stage(job.id, "usage", _usage_dict(llm) or {})
            return {
                "status": "answered", "answer": answer, "sql": sql_note, "columns": columns, "rows": rows,
                "usage": _usage_dict(llm),
            }

        answer = await synthesize_answer(
            llm, settings.fast_model, question, attempt["sql"], attempt["columns"], attempt["rows"]
        )
        trace.append_stage(job.id, "answer_synthesized", {"answer": answer})
        column_formats = infer_column_formats(attempt["sql"], attempt["columns"], cmap)
        trace.append_stage(job.id, "usage", _usage_dict(llm) or {})

        return {
            "status": "answered",
            "answer": answer,
            "sql": attempt["sql"],
            "columns": attempt["columns"],
            "rows": attempt["rows"][: settings.max_result_rows],
            "column_formats": column_formats,
            "usage": _usage_dict(llm),
        }
    finally:
        con.close()
