"""Tests for the deterministic (no LLM call) column-format classifier used
to tell the frontend which result columns are monetary vs plain counts vs
everything else - keyed off the dataset's own already-resolved concept map,
never off guessed column-name vocabulary.
"""
from app.nlq.column_formats import infer_column_formats
from app.semantic.models import ConceptMap, ResolvedRole


def cmap_with_roles(**roles: ResolvedRole) -> ConceptMap:
    return ConceptMap(dataset_id="d", table_name="transactions", roles=roles)


def resolved(role: str, expression: str, source: str = "direct_column") -> ResolvedRole:
    return ResolvedRole(role=role, source=source, expression=expression, confidence=0.9)


def not_found(role: str) -> ResolvedRole:
    return ResolvedRole(role=role, source="not_found", confidence=0.0)


def test_aggregate_over_unit_amount_is_currency():
    cmap = cmap_with_roles(
        unit_amount=resolved("unit_amount", '"unit_price"'),
        line_amount=not_found("line_amount"),
    )
    sql = 'SELECT SUM("unit_price") AS revenue FROM transactions LIMIT 500'
    assert infer_column_formats(sql, ["revenue"], cmap) == {"revenue": "currency"}


def test_count_star_is_integer_even_if_it_shares_a_row_with_currency():
    cmap = cmap_with_roles(unit_amount=resolved("unit_amount", '"unit_price"'))
    sql = 'SELECT COUNT(*) AS n, SUM("unit_price") AS revenue FROM transactions LIMIT 500'
    assert infer_column_formats(sql, ["n", "revenue"], cmap) == {"n": "integer", "revenue": "currency"}


def test_unrelated_column_is_not_classified():
    cmap = cmap_with_roles(unit_amount=resolved("unit_amount", '"unit_price"'))
    sql = 'SELECT "country" AS country, "quantity" AS qty FROM transactions LIMIT 500'
    assert infer_column_formats(sql, ["country", "qty"], cmap) == {}


def test_no_monetary_roles_resolved_returns_empty_without_parsing():
    cmap = cmap_with_roles(unit_amount=not_found("unit_amount"), line_amount=not_found("line_amount"))
    sql = 'SELECT SUM("unit_price") AS revenue FROM transactions LIMIT 500'
    assert infer_column_formats(sql, ["revenue"], cmap) == {}


def test_direct_line_amount_column_contributes():
    cmap = cmap_with_roles(line_amount=resolved("line_amount", '"line_total"', source="direct_column"))
    sql = 'SELECT SUM("line_total") AS revenue FROM transactions LIMIT 500'
    assert infer_column_formats(sql, ["revenue"], cmap) == {"revenue": "currency"}


def test_derived_line_amount_does_not_itself_leak_its_rate_columns():
    """line_amount here is a *derived* expression built from unit_amount and
    a discount rate - the rate column must not be treated as monetary just
    because it appears inside that derivation string; only unit_amount's
    own column should be."""
    cmap = cmap_with_roles(
        unit_amount=resolved("unit_amount", '"unit_price"'),
        line_amount=resolved(
            "line_amount", '("quantity" * "unit_price" * (1 - "promo_rate"))', source="derived_expression"
        ),
    )
    sql = 'SELECT AVG("promo_rate") AS avg_discount FROM transactions LIMIT 500'
    assert infer_column_formats(sql, ["avg_discount"], cmap) == {}


def test_line_amount_adjustment_contributes():
    cmap = cmap_with_roles(line_amount_adjustment=resolved("line_amount_adjustment", '"freight_value"'))
    sql = 'SELECT SUM("freight_value") AS shipping_total FROM transactions LIMIT 500'
    assert infer_column_formats(sql, ["shipping_total"], cmap) == {"shipping_total": "currency"}


def test_malformed_sql_fails_soft_to_empty_dict():
    cmap = cmap_with_roles(unit_amount=resolved("unit_amount", '"unit_price"'))
    assert infer_column_formats("not valid sql {{{", ["x"], cmap) == {}


def test_monetary_ness_traces_through_a_chain_of_ctes():
    """Reproduces a real gap found live: a compound question answered via
    staged CTEs (SUM(line_revenue) in one CTE, then SUM of that in the
    next) must still be recognized as currency in the final output, even
    though the final SELECT's own projection only references the
    intermediate alias ('month_revenue'), never 'line_revenue' directly."""
    cmap = cmap_with_roles(line_amount=resolved("line_amount", '"line_revenue"', source="direct_column"))
    sql = """
    WITH monthly AS (
      SELECT "country", "invoice_month", SUM("line_revenue") AS month_revenue
      FROM transactions GROUP BY "country", "invoice_month"
    ),
    total AS (
      SELECT "country", SUM(month_revenue) AS total_revenue FROM monthly GROUP BY "country"
    ),
    ranked AS (
      SELECT "country", "invoice_month", month_revenue,
             ROW_NUMBER() OVER (PARTITION BY "country" ORDER BY month_revenue DESC) AS rn
      FROM monthly
    )
    SELECT t.country, t.total_revenue, r.invoice_month AS top_month, r.month_revenue AS top_month_revenue
    FROM total AS t JOIN ranked AS r ON t.country = r.country AND r.rn = 1
    """
    formats = infer_column_formats(sql, ["country", "total_revenue", "top_month", "top_month_revenue"], cmap)
    assert formats == {"total_revenue": "currency", "top_month_revenue": "currency"}


def test_count_inside_a_cte_is_still_integer_in_the_final_output():
    cmap = cmap_with_roles(unit_amount=resolved("unit_amount", '"unit_price"'))
    sql = """
    WITH per_order AS (SELECT "order_id", COUNT(*) AS line_count FROM transactions GROUP BY "order_id")
    SELECT "order_id", line_count FROM per_order
    """
    assert infer_column_formats(sql, ["order_id", "line_count"], cmap) == {"line_count": "integer"}
