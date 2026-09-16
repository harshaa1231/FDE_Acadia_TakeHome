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
DDL/DML, no comments, no markdown fences. A `WITH ... SELECT` (common \
table expressions) is still one statement and is encouraged when it makes \
the query correct - see rule 8.
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
when one already covers what is asked - but a named metric is only \
complete if it accounts for every relevant column. Before using \
`net_revenue` or any other monetary metric, check `other_columns`: if one \
of them looks, from its name or sample values, like it still adjusts a \
row's monetary total (a rate, percentage, fee, tax, discount, or refund \
column not already folded into a structural role), do not treat the named \
metric as complete. Either incorporate that column into your own SQL \
expression, or if you are not confident how it combines with the total, \
`REFUSE` rather than present a number that may be silently wrong.
7. When ranking or identifying a specific entity, customer, or item \
(e.g. "top customer", "best-selling product", "which driver..."), exclude \
rows where that identifier is NULL, unless the question is explicitly \
about missing or unidentified records. Real transactional data commonly \
has NULL foreign keys (guest checkouts, unlinked accounts) - a NULL group \
is missing data, never a real, nameable answer to "which one".
8. A question can ask for several things at once - filter to a subset, \
break the result down by one or more dimensions, AND identify the top (or \
bottom) entry within each group, all in the same question. Answer ALL of \
it, not just the first clause: decompose the question into its filter, \
its grouping dimension(s), and its per-group ranking, and write ONE query \
that produces all of them together as columns of a single result set - a \
`WITH` clause staging an aggregation followed by a window function \
(`ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)`, or similar) to pick \
the top row per group is the standard way to do this in one statement. Do \
not silently drop part of a multi-part question because a single flat \
GROUP BY doesn't cover all of it.
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
