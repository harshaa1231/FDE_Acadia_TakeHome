"""Infers a display format ("currency" | "integer" | "number" | "text") for
each column of a query result, so the UI can format a monetary total
differently from a plain count without guessing from the column's name (an
LLM-chosen alias like "total" or "amt" tells you nothing reliable) and
without hardcoding any business vocabulary.

The signal used instead is entirely structural: which of the *dataset's
own already-resolved* structural roles - unit_amount, line_amount (only
when it's a genuine direct total column, not a derived expression that
already double-counts unit_amount), line_amount_adjustment - does this
output column's SQL expression reference, and is it wrapped in COUNT (a
count is never a currency figure no matter what it counts). This is the
same semantic-layer pattern used everywhere else in this app: the raw
column names in *this* file are never known in advance, but what a
concept resolved to for *this* dataset is - so classification is keyed off
that resolved concept map, not off guessed naming conventions.

A misclassification here only affects cosmetic number formatting, never a
value or an answer, so this stays a best-effort heuristic rather than
something requiring guardrail-level rejection.
"""
from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

from app.semantic.models import ConceptMap

_QUOTED_IDENT = re.compile(r'"([^"]+)"')


def _monetary_base_columns(cmap: ConceptMap) -> set[str]:
    names: set[str] = set()
    unit = cmap.get("unit_amount")
    if unit and unit.expression:
        names.update(_QUOTED_IDENT.findall(unit.expression))

    adjustment = cmap.get("line_amount_adjustment")
    if adjustment and adjustment.expression:
        names.update(_QUOTED_IDENT.findall(adjustment.expression))

    line = cmap.roles.get("line_amount")
    if line and line.source == "direct_column" and line.expression:
        names.update(_QUOTED_IDENT.findall(line.expression))

    return names


def infer_column_formats(sql: str, columns: list[str], cmap: ConceptMap) -> dict[str, str]:
    monetary_columns = _monetary_base_columns(cmap)
    if not monetary_columns:
        return {}

    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
        select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
        if select is None:
            return {}
    except Exception:
        return {}

    # A compound question answered via CTEs stages aggregation through
    # several SELECTs - e.g. "SUM(line_revenue) AS month_revenue" in one
    # CTE, then "SUM(month_revenue) AS total_revenue" in the next, with
    # only the final SELECT actually producing the output columns. A CTE
    # can only reference CTEs defined earlier in the same WITH clause, so
    # walking them in that order and growing `monetary_columns` as each
    # one's own output is recognized as monetary correctly propagates
    # "this is money" through any depth of staged aggregation, ending
    # with the final outer SELECT that the output `columns` came from.
    cte_selects: list[exp.Select] = []
    with_clause = select.args.get("with")
    if with_clause:
        for cte in with_clause.find_all(exp.CTE):
            cte_selects.extend(cte.this.find_all(exp.Select))

    integer_names: set[str] = set()
    for s in [*cte_selects, select]:
        for proj in s.expressions:
            if proj.find(exp.Count):
                integer_names.add(proj.alias_or_name)
                continue
            referenced = {c.name for c in proj.find_all(exp.Column)}
            if referenced & monetary_columns:
                monetary_columns.add(proj.alias_or_name)

    formats: dict[str, str] = {}
    for out_name, proj in zip(columns, select.expressions):
        proj_name = proj.alias_or_name
        if proj_name in integer_names:
            formats[out_name] = "integer"
        elif proj_name in monetary_columns:
            formats[out_name] = "currency"

    return formats
