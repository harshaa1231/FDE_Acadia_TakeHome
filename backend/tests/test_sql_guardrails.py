"""Tests for the AST-level SQL guardrail - the last line of defense between
LLM-generated SQL (untrusted input) and actual execution. Covers each of
the checks the module's own docstring documents, plus CTE/window-function
support (needed for compound, multi-facet questions - "break these down by
X, then find the top one per group" - that don't fit a single flat
aggregate) and the security property that relaxing the table check for
CTEs doesn't open a hole for a real external table to sneak in disguised
as one.
"""
import pytest

from app.nlq.sql_guardrails import SqlGuardrailError, validate_and_finalize_sql

TABLE = "transactions"
COLUMNS = {"country", "line_revenue", "invoice_month", "description", "is_return", "customer_id"}


def test_rejects_multiple_statements():
    with pytest.raises(SqlGuardrailError, match="one statement"):
        validate_and_finalize_sql(f"SELECT * FROM {TABLE}; DROP TABLE {TABLE}", TABLE, COLUMNS)


def test_rejects_ddl():
    with pytest.raises(SqlGuardrailError):
        validate_and_finalize_sql(f"DROP TABLE {TABLE}", TABLE, COLUMNS)


def test_rejects_dml():
    with pytest.raises(SqlGuardrailError):
        validate_and_finalize_sql(f"DELETE FROM {TABLE} WHERE country = 'X'", TABLE, COLUMNS)


def test_rejects_attach():
    with pytest.raises(SqlGuardrailError):
        validate_and_finalize_sql("ATTACH 'other.db' AS other", TABLE, COLUMNS)


def test_rejects_pragma():
    with pytest.raises(SqlGuardrailError):
        validate_and_finalize_sql("PRAGMA database_list", TABLE, COLUMNS)


def test_rejects_unknown_table():
    with pytest.raises(SqlGuardrailError, match="unknown table"):
        validate_and_finalize_sql("SELECT * FROM other_dataset", TABLE, COLUMNS)


def test_rejects_information_schema():
    with pytest.raises(SqlGuardrailError, match="unknown table"):
        validate_and_finalize_sql("SELECT * FROM information_schema.tables", TABLE, COLUMNS)


def test_rejects_unknown_column():
    with pytest.raises(SqlGuardrailError, match="unknown column"):
        validate_and_finalize_sql(f'SELECT "made_up_column" FROM {TABLE}', TABLE, COLUMNS)


def test_accepts_known_column_and_injects_limit():
    sql = validate_and_finalize_sql(f'SELECT SUM("line_revenue") AS revenue FROM {TABLE}', TABLE, COLUMNS)
    assert "LIMIT" in sql.upper()


def test_caps_an_oversized_limit():
    sql = validate_and_finalize_sql(f'SELECT * FROM {TABLE} LIMIT 999999', TABLE, COLUMNS)
    assert "LIMIT 500" in sql.upper() or "LIMIT 500" in sql


def test_keeps_a_reasonable_existing_limit():
    sql = validate_and_finalize_sql(f'SELECT * FROM {TABLE} LIMIT 10', TABLE, COLUMNS)
    assert "LIMIT 10" in sql


def test_cte_with_window_function_is_accepted():
    """The compound-question case: filter, aggregate by one dimension, then
    pick the top row per group via a window function staged through CTEs -
    all still one read-only SELECT statement."""
    sql = f"""
    WITH filtered AS (
      SELECT "country", "invoice_month", "line_revenue"
      FROM {TABLE}
      WHERE "description" ILIKE '%CHRISTMAS%' AND "is_return" = 0
    ),
    monthly AS (
      SELECT "country", "invoice_month", SUM("line_revenue") AS month_revenue,
             ROW_NUMBER() OVER (PARTITION BY "country" ORDER BY SUM("line_revenue") DESC) AS rn
      FROM filtered
      GROUP BY "country", "invoice_month"
    )
    SELECT "country", "invoice_month" AS top_month, month_revenue
    FROM monthly
    WHERE rn = 1
    ORDER BY month_revenue DESC
    """
    result = validate_and_finalize_sql(sql, TABLE, COLUMNS)
    assert "unknown table" not in result  # sanity: it didn't raise, and returned real SQL
    assert "LIMIT" in result.upper()


def test_cte_name_colliding_with_a_real_table_name_elsewhere_still_safe():
    """A CTE can only be reached by name from within the same query, and
    its body is itself fully validated - a CTE can't be used to smuggle a
    reference to a genuinely different, unknown table."""
    sql = """
    WITH sneaky AS (
      SELECT * FROM other_dataset
    )
    SELECT * FROM sneaky
    """
    with pytest.raises(SqlGuardrailError, match="unknown table"):
        validate_and_finalize_sql(sql, TABLE, COLUMNS)


def test_cte_referencing_undefined_alias_is_rejected():
    """A FROM clause naming something that is neither the real table nor a
    CTE defined in this query must still be rejected."""
    sql = f"""
    WITH filtered AS (SELECT "country" FROM {TABLE})
    SELECT * FROM not_a_real_cte
    """
    with pytest.raises(SqlGuardrailError, match="unknown table"):
        validate_and_finalize_sql(sql, TABLE, COLUMNS)
