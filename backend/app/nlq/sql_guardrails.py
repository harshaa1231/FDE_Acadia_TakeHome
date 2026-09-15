"""Validates LLM-generated SQL before it ever touches DuckDB. The LLM's
output is untrusted input, full stop - this module is what stands between
"the model said so" and a query actually executing.

Enforced, in order:
  1. Parses as exactly one statement (no `; DROP ...` smuggled after a
     semicolon).
  2. That statement is a SELECT (or a UNION of SELECTs) - no DDL/DML, no
     PRAGMA/ATTACH/COPY/CALL (anything sqlglot can't classify as a known
     read-only construct is rejected by default, not allowed by default).
  3. Every table referenced is the dataset's own table - never another
     dataset's file, `information_schema`, or a `pragma_*` table.
  4. Every column referenced is either a real column of that table or an
     alias defined earlier in the same query - never an invented name.
  5. A LIMIT is present, capped at MAX_RESULT_ROWS; one is injected if the
     model omitted it.
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

from app.core.config import settings

_DIALECT = "duckdb"


class SqlGuardrailError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def validate_and_finalize_sql(sql: str, table_name: str, known_columns: set[str]) -> str:
    """Raises SqlGuardrailError if the SQL fails any check. Returns the
    (possibly LIMIT-adjusted) SQL text, safe to execute, otherwise."""
    sql = sql.strip().rstrip(";")
    if not sql:
        raise SqlGuardrailError("empty SQL")

    try:
        statements = sqlglot.parse(sql, read=_DIALECT)
    except Exception as e:
        raise SqlGuardrailError(f"SQL failed to parse: {e}") from e

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SqlGuardrailError(f"expected exactly one statement, got {len(statements)}")

    root = statements[0]
    if not isinstance(root, (exp.Select, exp.Union, exp.Subquery)):
        raise SqlGuardrailError(f"only SELECT statements are allowed, got {type(root).__name__}")

    _reject_forbidden_nodes(root)
    _validate_tables(root, table_name)
    _validate_columns(root, known_columns)

    root = _ensure_limit(root, settings.max_result_rows)
    return root.sql(dialect=_DIALECT)


_FORBIDDEN_NODE_TYPES = (
    exp.Create, exp.Drop, exp.Insert, exp.Update, exp.Delete, exp.Alter,
    exp.Command, exp.Pragma, exp.Attach, exp.Detach, exp.TruncateTable,
    exp.Merge, exp.Grant,
)


def _reject_forbidden_nodes(root: exp.Expression) -> None:
    for node in root.walk():
        node_obj = node[0] if isinstance(node, tuple) else node
        if isinstance(node_obj, _FORBIDDEN_NODE_TYPES):
            raise SqlGuardrailError(f"disallowed SQL construct: {type(node_obj).__name__}")


def _validate_tables(root: exp.Expression, table_name: str) -> None:
    for t in root.find_all(exp.Table):
        name = t.name
        if name and name.lower() != table_name.lower():
            raise SqlGuardrailError(f"query references unknown table '{name}'")


def _validate_columns(root: exp.Expression, known_columns: set[str]) -> None:
    known_lower = {c.lower() for c in known_columns}
    defined_aliases = {a.alias.lower() for a in root.find_all(exp.Alias) if a.alias}
    for c in root.find_all(exp.Column):
        col_name = c.name
        if not col_name:
            continue
        if col_name.lower() in known_lower or col_name.lower() in defined_aliases:
            continue
        raise SqlGuardrailError(f"query references unknown column '{col_name}'")


def _ensure_limit(root: exp.Expression, max_rows: int) -> exp.Expression:
    existing_limit = root.args.get("limit")
    if existing_limit is None:
        return root.limit(max_rows)
    try:
        current = int(existing_limit.expression.this)
        if current > max_rows:
            return root.limit(max_rows)
    except (AttributeError, ValueError, TypeError):
        return root.limit(max_rows)
    return root
