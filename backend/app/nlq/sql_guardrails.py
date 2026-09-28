"""Validates LLM-generated SQL before it ever touches DuckDB. The LLM's
output is untrusted input, full stop - this module is what stands between
"the model said so" and a query actually executing.

Enforced, in order:
  1. Parses as exactly one statement (no `; DROP ...` smuggled after a
     semicolon).
  2. That statement is a SELECT (or a UNION of SELECTs) - no DDL/DML, no
     PRAGMA/ATTACH/COPY/CALL (anything sqlglot can't classify as a known
     read-only construct is rejected by default, not allowed by default).
  3. Every table position is either the dataset's own table, a CTE (`WITH
     x AS (...)`) defined earlier in the same query, or is rejected
     outright if it isn't a plain name at all. That last part matters:
     `FROM read_csv('/etc/passwd')`, `read_csv_auto(...)`,
     `read_parquet(...)` and any other DuckDB table function are still
     syntactically a SELECT, and instead of resolving to a named table
     they parse as a function call sitting in table position - sqlglot
     gives that node an empty `.name`, so a name-only check silently
     lets it through. The fix isn't a blacklist of function names (the
     next one DuckDB ships would sail right past a list); it's
     structural - a legitimate table position is always a plain
     identifier, so anything else in that position is rejected
     regardless of what it's called. See _validate_tables. CTEs are
     explicitly allowed: a compound question (filter, then break down by
     one dimension, then find the top row per group) often needs a
     window function staged through a CTE to answer in a single
     statement, and that's still one read-only SELECT, not multiple
     statements.
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
    cte_names = _cte_names(root)
    _validate_tables(root, table_name, cte_names)
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


def _cte_names(root: exp.Expression) -> set[str]:
    """Names introduced by a top-level WITH clause - these appear as
    ordinary exp.Table nodes wherever they're referenced in a FROM/JOIN,
    indistinguishable at that point from a real table reference, so they
    must be collected first and treated as known-safe locals rather than
    external tables."""
    with_clause = root.args.get("with")
    if not with_clause:
        return set()
    return {cte.alias.lower() for cte in with_clause.find_all(exp.CTE) if cte.alias}


def _table_function_name(node: exp.Expression) -> str:
    """Best-effort readable name for an error message - never used for the
    security decision itself, only to explain it. Dedicated function
    classes (ReadCSV, GenerateSeries, ...) know their own SQL name;
    sqlglot falls back to exp.Anonymous for any function it doesn't have
    a dedicated class for, which carries the literal name as `.this`."""
    if isinstance(node, exp.Anonymous):
        return str(node.this)
    if isinstance(node, exp.Func) and hasattr(node, "sql_name"):
        return node.sql_name()
    return type(node).__name__


def _validate_tables(root: exp.Expression, table_name: str, cte_names: set[str]) -> None:
    for t in root.find_all(exp.Table):
        # A legitimate table position is always a plain name (the
        # dataset's own table, or a CTE alias) - anything else sitting in
        # table position is a function call DuckDB will execute, which is
        # exactly how read_csv()/read_csv_auto()/read_parquet()/etc. reach
        # arbitrary files: syntactically still "just a SELECT", but not a
        # reference to any table this app registered. Reject the shape,
        # not a list of names, so a DuckDB table function this list has
        # never heard of is rejected the same way.
        if not isinstance(t.this, exp.Identifier):
            raise SqlGuardrailError(
                f"query uses a disallowed table function '{_table_function_name(t.this)}' - "
                "only the dataset's own table and CTEs defined in the query are allowed"
            )
        name = t.name
        if not name:
            continue
        if name.lower() == table_name.lower() or name.lower() in cte_names:
            continue
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
