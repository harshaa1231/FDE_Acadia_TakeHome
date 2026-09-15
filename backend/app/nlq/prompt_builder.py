"""Builds the SQL-generation prompt from the resolved concept map and
metric registry - never from raw column names guessed in advance. This is
the point where "the file's own structure" (profiled and role-mapped
upstream) becomes the only context the model is allowed to reason from.
"""
from __future__ import annotations

import json

from app.semantic.metric_registry import AvailableMetric
from app.semantic.models import ConceptMap

SYSTEM_PROMPT = """\
You are a careful data analyst writing SQL for a DuckDB table, answering \
one business question about transactional data.

You are given the ACTUAL structure of this specific file, already \
resolved from its raw schema for you:
- structural roles, each with the exact SQL expression you must use for \
that concept (a plain column reference, or a derived expression)
- named aggregate metrics you may call directly
- categorical dimensions available for grouping/filtering, with real \
sample values from the file
- a few real sample rows from the table
- any columns that did not map to a known role, exposed raw in case \
they are relevant to the question

Rules:
1. Use ONLY the column names and expressions given below, copied exactly. \
Never invent, guess, or rename a column.
2. Write exactly one read-only SQL SELECT statement, DuckDB dialect. No \
DDL/DML, no comments, no markdown fences.
3. If the question needs a concept, metric, or dimension that is NOT \
available below (marked not_found, or simply absent), do not guess or \
approximate it - respond with exactly one line: `REFUSE: <one sentence \
reason>` and nothing else.
4. If the question is a general request to describe, summarize, or \
explore the dataset as a whole rather than compute a specific metric \
(e.g. "tell me about this dataset", "what's in this data", "give me an \
overview"), it IS answerable - respond with exactly the single word \
`OVERVIEW` and nothing else. This is not a refusal case: an overview is \
built separately from the same structure given to you.
5. If you can answer with a specific query, respond with ONLY the SQL \
statement - no explanation before or after it.
6. Prefer a named metric expression over writing your own aggregation \
when one already covers what is asked.
7. When ranking or identifying a specific entity, customer, or item \
(e.g. "top customer", "best-selling product", "which driver..."), exclude \
rows where that identifier is NULL, unless the question is explicitly \
about missing or unidentified records. Real transactional data commonly \
has NULL foreign keys (guest checkouts, unlinked accounts) - a NULL group \
is missing data, never a real, nameable answer to "which one".
"""


def _roles_section(cmap: ConceptMap) -> list[dict]:
    out = []
    for name, r in cmap.roles.items():
        if r.source == "not_found":
            out.append({"role": name, "available": False})
        else:
            out.append({"role": name, "available": True, "sql_expression": r.expression})
    return out


def _metrics_section(metrics: list[AvailableMetric]) -> list[dict]:
    return [
        {"name": m.name, "description": m.description, "sql_expression": m.sql_expression}
        for m in metrics
        if m.available
    ]


def _dimensions_section(cmap: ConceptMap) -> list[dict]:
    return [
        {"column": d.column, "distinct_count": d.distinct_count, "sample_values": d.sample_values}
        for d in cmap.dimensions
    ]


def _unmapped_section(cmap: ConceptMap) -> list[dict]:
    return [
        {"column": u.name, "type": u.duckdb_type, "sample_values": u.sample_values}
        for u in cmap.unmapped
    ]


def build_sql_prompt(
    table_name: str, cmap: ConceptMap, metrics: list[AvailableMetric], sample_rows: list[dict], question: str
) -> str:
    context = {
        "table_name": table_name,
        "roles": _roles_section(cmap),
        "available_named_metrics": _metrics_section(metrics),
        "dimensions": _dimensions_section(cmap),
        "other_columns": _unmapped_section(cmap),
        "sample_rows": sample_rows,
    }
    return f"{json.dumps(context, default=str, indent=2)}\n\nQuestion: {question}"


def build_retry_prompt(previous_sql: str, error_message: str) -> str:
    return (
        f"The following SQL you generated failed:\n{previous_sql}\n\n"
        f"Error: {error_message}\n\n"
        "Fix the SQL and respond again following the same rules - ONLY the "
        "corrected SQL statement, or `REFUSE: <reason>` if it cannot be fixed "
        "within the available columns/expressions."
    )
