import duckdb
import pytest

from app.ingestion.schema_profiler import build_schema_profile


@pytest.fixture
def con():
    c = duckdb.connect(":memory:")
    yield c
    c.close()


def test_profiles_basic_types(con):
    con.execute(
        """
        CREATE TABLE t AS SELECT * FROM (VALUES
            (1, 'a', 9.99, DATE '2024-01-01'),
            (2, 'b', 19.5, DATE '2024-01-02'),
            (3, 'a', 4.25, DATE '2024-01-03')
        ) AS v(id, category, price, dt)
        """
    )
    profile = build_schema_profile(con, "ds1", "t")
    assert profile.row_count == 3
    kinds = {c.name: c.kind for c in profile.columns}
    assert kinds["id"] == "integer"
    assert kinds["category"] == "text"
    assert kinds["price"] == "numeric"
    assert kinds["dt"] == "datetime"


def test_null_fraction_and_distinct(con):
    con.execute(
        """
        CREATE TABLE t AS SELECT * FROM (VALUES
            (1, 'x'), (2, 'x'), (3, NULL), (4, 'y')
        ) AS v(id, tag)
        """
    )
    profile = build_schema_profile(con, "ds2", "t")
    tag = profile.column("tag")
    assert tag.null_count == 1
    assert tag.null_fraction == pytest.approx(0.25)
    assert tag.distinct_count == 2


def test_zero_one_integer_column_reclassified_as_boolean(con):
    """A CSV with a 0/1 flag column loads as a plain integer type in DuckDB
    - there is no way to tell it apart from a real numeric measure without
    this structural check, which is what stops a flag column from being
    mistaken for revenue/quantity in the semantic layer."""
    con.execute(
        """
        CREATE TABLE t AS SELECT * FROM (VALUES
            (1, 0), (2, 1), (3, 0), (4, 1)
        ) AS v(id, is_returned)
        """
    )
    profile = build_schema_profile(con, "ds3", "t")
    assert profile.column("is_returned").kind == "boolean"
    assert profile.column("id").kind == "integer"


def test_single_valued_integer_column_also_reclassified_as_boolean(con):
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1, 0), (2, 0)) AS v(id, flag)")
    profile = build_schema_profile(con, "ds4", "t")
    assert profile.column("flag").kind == "boolean"


def test_integer_column_with_values_outside_zero_one_stays_integer(con):
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1, 2), (2, 5)) AS v(id, qty)")
    profile = build_schema_profile(con, "ds5", "t")
    assert profile.column("qty").kind == "integer"


def test_min_max_captured_for_numeric_and_datetime(con):
    con.execute(
        """
        CREATE TABLE t AS SELECT * FROM (VALUES
            (10.0, DATE '2024-01-01'),
            (30.0, DATE '2024-06-15')
        ) AS v(amount, dt)
        """
    )
    profile = build_schema_profile(con, "ds6", "t")
    amt = profile.column("amount")
    assert amt.min_value == 10.0
    assert amt.max_value == 30.0
    dt = profile.column("dt")
    assert dt.min_value == "2024-01-01"
    assert dt.max_value == "2024-06-15"
