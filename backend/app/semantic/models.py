"""Data models for the semantic layer.

The semantic layer resolves *structural* roles from the raw, purely
descriptive schema profile - a grouping id, an entity id, an item
identifier, a quantity, a monetary amount, an event time, and a list of
categorical dimensions - using generic, cross-domain heuristics (column
name patterns, dtype, cardinality, uniqueness ratio). It does not assume
retail vocabulary: nothing here requires a column literally named
"Country" or "CustomerID" to work, and the same detectors are meant to fire
on a banking ledger or a SaaS billing export as well as on the dev dataset.

Business-level naming only happens one layer up, in the LLM prompt, where
the model matches the user's own wording in the question against whatever
roles and dimension labels were actually found in *this* file. A column the
heuristics can't confidently place is never silently dropped - it is
exposed as `unmapped`, raw, so the LLM can still reference it.
"""
from __future__ import annotations

from pydantic import BaseModel

SourceKind = str  # "direct_column" | "derived_expression" | "not_found"

# Structural roles the mapper actively tries to resolve. This list (plus one
# detector function per role in concept_mapper.py) is the extensibility
# surface for new structural concepts - it never requires touching prompts,
# SQL, or the API.
STRUCTURAL_ROLES: list[str] = [
    "event_group_id",  # groups line items into one order/invoice/session/visit
    "entity_id",        # the "who" - customer/account/patient/member...
    "item_id",           # the "what" - product/service/SKU/procedure...
    "item_label",        # free-text label/description paired with item_id
    "quantity",
    "unit_amount",       # a per-unit monetary value
    "line_discount_rate",     # a fractional (0-1) discount/promo rate that reduces a row's total
    "line_surcharge_rate",    # a fractional (0-1) tax/markup rate that increases a row's total
    "line_amount_adjustment",  # a flat monetary delta (fee, flat discount, refund) on a row
    "line_amount",       # the monetary total for the row (direct or derived)
    "event_time",
]


class ResolvedRole(BaseModel):
    role: str
    source: SourceKind
    expression: str | None = None  # quoted column ref, or a DuckDB SQL expression
    confidence: float = 0.0
    note: str = ""
    # Other columns that matched this role's hints just as strongly. When
    # non-empty, the pick was a coin flip (column order) and the prompt
    # builder surfaces these as alternatives so the LLM can pick using the
    # question's own wording instead of the mapper silently guessing.
    alternatives: list[str] = []


class Dimension(BaseModel):
    """A categorical column usable for grouping/breakdown - could be a
    country, a channel, a payment method, a department, anything. Exposed
    as a candidate list rather than pre-bound to a business label, so the
    question's own wording (via the LLM) decides which one applies."""

    column: str
    distinct_count: int
    sample_values: list = []


class UnmappedColumn(BaseModel):
    name: str
    duckdb_type: str
    kind: str
    sample_values: list = []


class ConceptMap(BaseModel):
    dataset_id: str
    table_name: str
    roles: dict[str, ResolvedRole]
    dimensions: list[Dimension] = []
    unmapped: list[UnmappedColumn] = []

    def get(self, role: str) -> ResolvedRole | None:
        r = self.roles.get(role)
        return r if r and r.source != "not_found" else None

    def has(self, role: str) -> bool:
        return self.get(role) is not None
