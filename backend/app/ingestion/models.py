"""Data models for the raw, purely-descriptive schema profile.

This layer knows nothing about business meaning (revenue, customer, ...) -
that interpretation happens one layer up, in app.semantic. Keeping the split
strict is what lets an unfamiliar CSV be profiled the same way every time.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

ColumnKind = Literal["numeric", "integer", "text", "datetime", "boolean", "unknown"]


class ColumnProfile(BaseModel):
    name: str
    duckdb_type: str
    kind: ColumnKind
    null_count: int
    null_fraction: float
    distinct_count: int
    is_likely_unique: bool
    sample_values: list[Any] = []
    min_value: Any = None
    max_value: Any = None


class SchemaProfile(BaseModel):
    dataset_id: str
    table_name: str
    row_count: int
    columns: list[ColumnProfile]

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)
