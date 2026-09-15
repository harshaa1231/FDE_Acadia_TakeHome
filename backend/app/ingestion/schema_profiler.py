"""Builds a purely descriptive statistical profile of an already-loaded
DuckDB table. No business interpretation happens here - see
app.semantic.concept_mapper for that. This module only answers "what is
literally in this column": type, nulls, cardinality, samples, range.
"""
from __future__ import annotations

import duckdb

from app.core.config import settings
from app.ingestion.models import ColumnKind, ColumnProfile, SchemaProfile

_LOW_CARDINALITY_THRESHOLD = 50

_TYPE_KIND_MAP: dict[str, ColumnKind] = {
    "BIGINT": "integer",
    "INTEGER": "integer",
    "SMALLINT": "integer",
    "TINYINT": "integer",
    "HUGEINT": "integer",
    "UBIGINT": "integer",
    "UINTEGER": "integer",
    "USMALLINT": "integer",
    "UTINYINT": "integer",
    "DOUBLE": "numeric",
    "FLOAT": "numeric",
    "REAL": "numeric",
    "DATE": "datetime",
    "TIMESTAMP": "datetime",
    "TIMESTAMP WITH TIME ZONE": "datetime",
    "TIME": "datetime",
    "BOOLEAN": "boolean",
    "VARCHAR": "text",
}


def _kind_for(duckdb_type: str) -> ColumnKind:
    base = duckdb_type.upper().split("(")[0].strip()
    if base.startswith("DECIMAL"):
        return "numeric"
    return _TYPE_KIND_MAP.get(base, "unknown")


def build_schema_profile(
    con: duckdb.DuckDBPyConnection, dataset_id: str, table_name: str
) -> SchemaProfile:
    row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    described = con.execute(f"DESCRIBE {table_name}").fetchall()  # (name, type, null, key, default, extra)

    columns: list[ColumnProfile] = []
    for name, duckdb_type, *_ in described:
        col = f'"{name}"'
        kind = _kind_for(duckdb_type)

        null_count, distinct_count = con.execute(
            f"SELECT COUNT(*) - COUNT({col}), approx_count_distinct({col}) FROM {table_name}"
        ).fetchone()
        distinct_count = int(distinct_count)
        is_likely_unique = row_count > 0 and distinct_count >= row_count * 0.98

        sample_values: list = []
        min_value = None
        max_value = None

        if kind in ("numeric", "integer", "datetime"):
            min_value, max_value = con.execute(
                f"SELECT MIN({col}), MAX({col}) FROM {table_name}"
            ).fetchone()
            min_value = _jsonable(min_value)
            max_value = _jsonable(max_value)

        if distinct_count <= _LOW_CARDINALITY_THRESHOLD:
            rows = con.execute(
                f"SELECT DISTINCT {col} FROM {table_name} "
                f"WHERE {col} IS NOT NULL LIMIT {_LOW_CARDINALITY_THRESHOLD}"
            ).fetchall()
            sample_values = [_jsonable(r[0]) for r in rows]
        else:
            rows = con.execute(
                f"SELECT {col} FROM {table_name} "
                f"WHERE {col} IS NOT NULL LIMIT {settings.profile_sample_rows}"
            ).fetchall()
            sample_values = [_jsonable(r[0]) for r in rows]

        if kind == "integer" and distinct_count in (1, 2) and set(sample_values) <= {0, 1}:
            # A 0/1-valued integer column is structurally a boolean flag,
            # regardless of what it's named - DuckDB has no way to know
            # that from a CSV's raw digits, so we infer it here. This keeps
            # flag/indicator columns out of quantity and monetary detection
            # in the semantic layer without any name-based special-casing.
            kind = "boolean"

        columns.append(
            ColumnProfile(
                name=name,
                duckdb_type=duckdb_type,
                kind=kind,
                null_count=int(null_count),
                null_fraction=(null_count / row_count) if row_count else 0.0,
                distinct_count=distinct_count,
                is_likely_unique=is_likely_unique,
                sample_values=sample_values,
                min_value=min_value,
                max_value=max_value,
            )
        )

    return SchemaProfile(
        dataset_id=dataset_id, table_name=table_name, row_count=row_count, columns=columns
    )


def _jsonable(value):
    """DuckDB can hand back date/datetime/Decimal objects; make them JSON
    and prompt safe without losing information."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        import decimal

        if isinstance(value, decimal.Decimal):
            return float(value)
    except ImportError:
        pass
    return value
