"""Resolves structural roles (grouping id, entity id, item id, quantity,
monetary amounts, event time) from a raw SchemaProfile.

This is LLM-driven, not keyword-driven: the LLM is given only objective,
file-derived evidence - column names, types, cardinality, and real sample
values pulled from the file itself - and proposes which column (if any)
fills each role, using its own world knowledge to interpret vocabulary no
static synonym list could anticipate (a "rider_handle" is the customer in a
rideshare export just as clearly as a "CustomerID" is in a retail one, but
no fixed dictionary can enumerate every domain's wording in advance).

Every proposal is then validated deterministically against the real schema
before being trusted: the column must exist, must not already be claimed by
another role, and must have a structurally sane type for that role. A
proposal that fails validation is rejected, never silently patched - the
role is left `not_found` rather than guessed. If the LLM call itself fails
(no key, network error, bad JSON), a minimal, conservative structural
fallback resolves only what is structurally unambiguous and leaves
everything else `not_found` - degraded, but never a wrong guess.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.ingestion.models import ColumnProfile, SchemaProfile
from app.nlq.llm_client import LLMClient
from app.semantic.models import (
    STRUCTURAL_ROLES,
    ConceptMap,
    Dimension,
    ResolvedRole,
    UnmappedColumn,
)

logger = logging.getLogger(__name__)

_LOW_CARDINALITY_DIMENSION_MAX = 200
_MAX_DIMENSION_NULL_FRACTION = 0.5

_ID_LIKE_KINDS = ("text", "integer")
_AMOUNT_LIKE_KINDS = ("integer", "numeric")

_SYSTEM_PROMPT = """\
You are inferring the business meaning of columns in an unfamiliar \
transactional CSV. It could be retail orders, ride bookings, subscription \
invoices, healthcare charges, restaurant tabs, or any other transactional \
data - you are not told the domain in advance. You are given only \
objective, file-derived statistics for each column: its name, data type, \
null rate, distinct-value count, whether it looks like a unique key, and a \
handful of real sample values actually present in the file. Use this \
evidence - especially the sample values - to decide what each column \
represents. Do not assume any single domain's vocabulary; reason from the \
evidence given, for this file specifically.

Propose, for each structural role below, which single column (if any) \
fills it. A role having no matching column is a valid, expected answer - \
never force a match you are not confident about.

- event_group_id: identifier grouping multiple rows into one order, \
booking, invoice, or session (several line items sharing one number)
- entity_id: identifier for who the transaction is with (customer, \
account, rider, patient, member, guest...)
- item_id: identifier for what was transacted (product, service, SKU, \
procedure...)
- item_label: a free-text name/description column paired with item_id
- quantity: a numeric count of units in the row
- unit_amount: a per-unit monetary value
- line_discount_rate: a fractional discount/promo/rebate rate (between 0 \
and 1) that REDUCES a row's monetary total when applied, e.g. a promo \
rate or discount percentage stored as a fraction. Only propose this if \
such a column clearly exists - most datasets will not have one.
- line_surcharge_rate: a fractional tax/markup/service-charge rate \
(between 0 and 1) that INCREASES a row's monetary total when applied, \
e.g. a sales tax rate, VAT rate, or markup percentage stored as a \
fraction. Only propose this if such a column clearly exists, and never \
propose the same column as line_discount_rate.
- line_amount_adjustment: a flat monetary amount added to or subtracted \
from a row's total that is NOT already unit_amount/quantity/a rate - \
e.g. a shipping fee, flat service charge, flat discount amount, or \
refund amount expressed in the same currency as the transaction \
(positive values increase the total, negative values decrease it). Only \
propose this if such a column clearly exists.
- line_amount: the monetary total for the row. If there is no direct \
total column, set derive_from_quantity_and_unit_amount to true whenever \
BOTH a quantity and a unit_amount clearly exist and multiplying them \
yields the row's PRE-adjustment subtotal - do this even if you also \
proposed a line_discount_rate, line_surcharge_rate, or \
line_amount_adjustment above; you do not need to compute the final \
total yourself, the system automatically applies whichever of those \
adjustment roles you resolved on top of quantity * unit_amount. Only \
leave derive_from_quantity_and_unit_amount false if quantity or \
unit_amount themselves are missing or don't multiply to a meaningful \
subtotal. If there is a single monetary column and no separate quantity \
concept, propose that column directly as line_amount instead (it \
already represents the row's total) rather than deriving.
- event_time: the timestamp of the transaction

Respond with strict JSON only (no markdown fences, no other text): a single \
object with one key, "roles", holding a JSON array with exactly one entry \
per role listed above, each shaped \
{"role": "<role name>", "column": "<exact column name or null>", \
"confidence": 0.0-1.0, "reasoning": "<one sentence>"} - except the \
line_amount entry, which also includes \
"derive_from_quantity_and_unit_amount": true|false. Example shape:
{
  "roles": [
    {"role": "event_group_id", "column": "<exact column name or null>", "confidence": 0.0-1.0, "reasoning": "<one sentence>"},
    {"role": "entity_id", "column": ..., "confidence": ..., "reasoning": ...},
    {"role": "item_id", "column": ..., "confidence": ..., "reasoning": ...},
    {"role": "item_label", "column": ..., "confidence": ..., "reasoning": ...},
    {"role": "quantity", "column": ..., "confidence": ..., "reasoning": ...},
    {"role": "unit_amount", "column": ..., "confidence": ..., "reasoning": ...},
    {"role": "line_discount_rate", "column": ..., "confidence": ..., "reasoning": ...},
    {"role": "line_surcharge_rate", "column": ..., "confidence": ..., "reasoning": ...},
    {"role": "line_amount_adjustment", "column": ..., "confidence": ..., "reasoning": ...},
    {"role": "line_amount", "column": ..., "derive_from_quantity_and_unit_amount": true|false, "confidence": ..., "reasoning": ...},
    {"role": "event_time", "column": ..., "confidence": ..., "reasoning": ...}
  ]
}

Column names in your response must be copied exactly as given in the \
input - do not alter case, spacing, or punctuation.
"""


class _RoleProposal(BaseModel):
    column: str | None = None
    confidence: float = 0.5
    reasoning: str = ""


class _LineAmountProposal(_RoleProposal):
    derive_from_quantity_and_unit_amount: bool = False


class _LLMRoleResponse(BaseModel):
    event_group_id: _RoleProposal = Field(default_factory=_RoleProposal)
    entity_id: _RoleProposal = Field(default_factory=_RoleProposal)
    item_id: _RoleProposal = Field(default_factory=_RoleProposal)
    item_label: _RoleProposal = Field(default_factory=_RoleProposal)
    quantity: _RoleProposal = Field(default_factory=_RoleProposal)
    unit_amount: _RoleProposal = Field(default_factory=_RoleProposal)
    line_discount_rate: _RoleProposal = Field(default_factory=_RoleProposal)
    line_surcharge_rate: _RoleProposal = Field(default_factory=_RoleProposal)
    line_amount_adjustment: _RoleProposal = Field(default_factory=_RoleProposal)
    line_amount: _LineAmountProposal = Field(default_factory=_LineAmountProposal)
    event_time: _RoleProposal = Field(default_factory=_RoleProposal)


def _roles_array_to_dict(raw: dict) -> dict:
    """Converts the wire format {"roles": [{"role": "x", ...}, ...]} into
    the {"x": {...}, ...} shape _LLMRoleResponse expects. The array shape
    is what we actually ask the model for - see the system prompt's
    docstring note above for why: an object with many identically-shaped
    keys (one per structural role) turned out to reliably produce
    malformed JSON from the fast model on wider schemas (a stray '{'
    before each subsequent key, as if the model wanted to emit a list),
    reproduced consistently against a real 25-column file. Asking for what
    the model already wants to produce - a JSON array of similarly-shaped
    items - removed the failure entirely in repeated live testing.

    Accepts the flat {"event_group_id": {...}, ...} shape unchanged too
    (what every test fixture and fallback path in this module already
    builds) - only a top-level "roles" array is unwrapped."""
    if not isinstance(raw, dict):
        return {}
    if "roles" not in raw:
        return raw
    out: dict = {}
    for item in raw.get("roles") or []:
        if not isinstance(item, dict):
            continue
        role_name = item.get("role")
        if not role_name:
            continue
        out[role_name] = {k: v for k, v in item.items() if k != "role"}
    return out


def _column_evidence(profile: SchemaProfile) -> list[dict]:
    return [
        {
            "name": c.name,
            "type": c.duckdb_type,
            "kind": c.kind,
            "null_fraction": round(c.null_fraction, 3),
            "distinct_count": c.distinct_count,
            "is_likely_unique_key": c.is_likely_unique,
            "sample_values": c.sample_values[:5],
        }
        for c in profile.columns
    ]


async def _propose_roles(profile: SchemaProfile, llm: LLMClient, model: str) -> _LLMRoleResponse:
    import json

    user_prompt = (
        f"Table has {profile.row_count} rows. Columns:\n"
        f"{json.dumps(_column_evidence(profile), default=str)}"
    )
    # One retry before giving up: LLMs occasionally emit malformed JSON
    # under response_format=json_object (a dropped comma, a stray brace) -
    # a transient generation glitch, not a signal the schema is unusual.
    # Mirrors the same one-retry pattern already used for SQL generation.
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            raw = await llm.complete_json(_SYSTEM_PROMPT, user_prompt, model=model)
            return _LLMRoleResponse.model_validate(_roles_array_to_dict(raw))
        except Exception as e:
            last_error = e
            logger.warning(
                "concept role proposal attempt %d/2 failed for dataset %s (%s)",
                attempt + 1, profile.dataset_id, e,
            )
    raise last_error


def _find_column(profile: SchemaProfile, name: str | None) -> ColumnProfile | None:
    if not name:
        return None
    for c in profile.columns:
        if c.name == name:
            return c
    normalized = name.strip().lower()
    for c in profile.columns:
        if c.name.lower() == normalized:
            return c
    return None


def _validate_simple_role(
    role: str, proposal: _RoleProposal, profile: SchemaProfile, claimed: set[str], allowed_kinds: tuple[str, ...]
) -> ResolvedRole:
    col = _find_column(profile, proposal.column)
    if col is None:
        reason = "no column proposed" if not proposal.column else f"proposed column '{proposal.column}' not found in schema"
        return ResolvedRole(role=role, source="not_found", confidence=0.0, note=reason)
    if col.name in claimed:
        return ResolvedRole(
            role=role, source="not_found", confidence=0.0,
            note=f"proposed column '{col.name}' was already claimed by another role",
        )
    if col.kind not in allowed_kinds:
        return ResolvedRole(
            role=role, source="not_found", confidence=0.0,
            note=f"proposed column '{col.name}' has kind '{col.kind}', not valid for {role} ({allowed_kinds})",
        )
    confidence = max(0.0, min(1.0, proposal.confidence))
    claimed.add(col.name)
    return ResolvedRole(
        role=role, source="direct_column", expression=f'"{col.name}"',
        confidence=confidence, note=f"LLM-proposed: {proposal.reasoning}"[:300],
    )


_RATE_TOLERANCE = 1e-9


def _validate_rate_role(
    role: str, proposal: _RoleProposal, profile: SchemaProfile, claimed: set[str]
) -> ResolvedRole:
    """Shared validator for any fractional (0-1) per-line rate role -
    line_discount_rate and line_surcharge_rate both reduce to the same
    structural check: numeric, and every actual value in the file falls
    within [0, 1]. Whether the rate increases or decreases the total is a
    fixed property of which role it was proposed for, decided by the LLM
    from the column's name/samples - never guessed from the numbers alone."""
    col = _find_column(profile, proposal.column)
    if col is None:
        reason = "no column proposed" if not proposal.column else f"proposed column '{proposal.column}' not found in schema"
        return ResolvedRole(role=role, source="not_found", confidence=0.0, note=reason)
    if col.name in claimed:
        return ResolvedRole(
            role=role, source="not_found", confidence=0.0,
            note=f"proposed column '{col.name}' was already claimed by another role",
        )
    if col.kind != "numeric":
        return ResolvedRole(
            role=role, source="not_found", confidence=0.0,
            note=f"proposed column '{col.name}' has kind '{col.kind}', not a fractional rate",
        )
    lo, hi = col.min_value, col.max_value
    if lo is None or hi is None or lo < -_RATE_TOLERANCE or hi > 1 + _RATE_TOLERANCE:
        return ResolvedRole(
            role=role, source="not_found", confidence=0.0,
            note=f"proposed column '{col.name}' has values outside [0, 1] ({lo}..{hi}), not a fractional rate",
        )
    confidence = max(0.0, min(1.0, proposal.confidence))
    claimed.add(col.name)
    return ResolvedRole(
        role=role, source="direct_column", expression=f'"{col.name}"',
        confidence=confidence, note=f"LLM-proposed: {proposal.reasoning}"[:300],
    )


def _validate_flat_adjustment(
    proposal: _RoleProposal, profile: SchemaProfile, claimed: set[str]
) -> ResolvedRole:
    """A flat per-line monetary delta (fee, flat discount, refund) - any
    sign is valid (it can increase or decrease the total), so the only
    structural check is numeric type and that it isn't already claimed by
    another role."""
    col = _find_column(profile, proposal.column)
    if col is None:
        reason = "no column proposed" if not proposal.column else f"proposed column '{proposal.column}' not found in schema"
        return ResolvedRole(role="line_amount_adjustment", source="not_found", confidence=0.0, note=reason)
    if col.name in claimed:
        return ResolvedRole(
            role="line_amount_adjustment", source="not_found", confidence=0.0,
            note=f"proposed column '{col.name}' was already claimed by another role",
        )
    if col.kind not in _AMOUNT_LIKE_KINDS:
        return ResolvedRole(
            role="line_amount_adjustment", source="not_found", confidence=0.0,
            note=f"proposed column '{col.name}' has kind '{col.kind}', not a monetary amount",
        )
    confidence = max(0.0, min(1.0, proposal.confidence))
    claimed.add(col.name)
    return ResolvedRole(
        role="line_amount_adjustment", source="direct_column", expression=f'"{col.name}"',
        confidence=confidence, note=f"LLM-proposed: {proposal.reasoning}"[:300],
    )


def _compose_adjustments(base_expression: str, roles: dict[str, ResolvedRole]) -> tuple[str, list[str]]:
    """Folds any resolved discount/surcharge/flat-adjustment roles on top of
    a base line-total expression, in a fixed order (discount, then
    surcharge, then flat). Shared by every way line_amount can be
    established without a direct total column of its own - a fully derived
    quantity * unit_amount subtotal, or unit_amount standing in for the
    total when quantity is implicitly 1 - since in both cases we're
    constructing the total ourselves and any adjustment we've already
    validated applies just as legitimately either way."""
    expression = base_expression
    applied: list[str] = []

    discount = roles.get("line_discount_rate")
    if discount and discount.source != "not_found":
        expression = f"{expression} * (1 - {discount.expression})"
        applied.append(f"discounted by {discount.expression}")

    surcharge = roles.get("line_surcharge_rate")
    if surcharge and surcharge.source != "not_found":
        expression = f"{expression} * (1 + {surcharge.expression})"
        applied.append(f"surcharged by {surcharge.expression}")

    adjustment = roles.get("line_amount_adjustment")
    if adjustment and adjustment.source != "not_found":
        expression = f"{expression} + {adjustment.expression}"
        applied.append(f"adjusted by {adjustment.expression}")

    return expression, applied


def _validate_line_amount(
    proposal: _LineAmountProposal, profile: SchemaProfile, claimed: set[str], roles: dict[str, ResolvedRole]
) -> ResolvedRole:
    # Sanctioned exception to "no column serves two roles": line_amount may
    # legitimately be the exact same column as unit_amount when there's no
    # separate quantity concept (a single "fare_amount" both prices and
    # totals the row - quantity=1 is implicit). Every other duplicate claim
    # is still rejected below.
    unit_role = roles.get("unit_amount")
    if (
        proposal.column and unit_role and unit_role.source == "direct_column"
        and unit_role.expression == f'"{proposal.column}"'
    ):
        expression, applied = _compose_adjustments(unit_role.expression, roles)
        note = "same column as unit_amount, treated as the line total (no separate quantity concept)"
        if applied:
            note += ", " + ", ".join(applied)
        note = f"{note}: {proposal.reasoning}"[:300]
        return ResolvedRole(
            role="line_amount",
            source="direct_column" if not applied else "derived_expression",
            expression=expression if not applied else f"({expression})",
            confidence=max(0.0, min(1.0, proposal.confidence)),
            note=note,
        )

    direct = _validate_simple_role("line_amount", proposal, profile, claimed, _AMOUNT_LIKE_KINDS)
    if direct.source == "direct_column":
        return direct

    if proposal.derive_from_quantity_and_unit_amount and roles.get("quantity") and roles.get("unit_amount"):
        qty, unit = roles["quantity"], roles["unit_amount"]
        if qty.source != "not_found" and unit.source != "not_found":
            expression, applied = _compose_adjustments(f"{qty.expression} * {unit.expression}", roles)
            note = "LLM-proposed derivation"
            if applied:
                note += ", " + ", ".join(applied)
            note = f"{note}: {proposal.reasoning}"[:300]

            return ResolvedRole(
                role="line_amount", source="derived_expression",
                expression=f"({expression})",
                confidence=max(0.0, min(1.0, proposal.confidence)),
                note=note,
            )

    return ResolvedRole(
        role="line_amount", source="not_found", confidence=0.0,
        note="no direct or derivable monetary total proposed by the LLM",
    )


def _structural_fallback_roles(profile: SchemaProfile) -> dict[str, ResolvedRole]:
    """Used only when the LLM call itself fails. Resolves nothing beyond
    what is structurally unambiguous (a single datetime column) - anything
    requiring interpretation is left not_found rather than guessed."""
    roles: dict[str, ResolvedRole] = {
        role: ResolvedRole(role=role, source="not_found", confidence=0.0, note="LLM unavailable; structural fallback")
        for role in STRUCTURAL_ROLES
    }
    datetime_cols = [c for c in profile.columns if c.kind == "datetime"]
    if len(datetime_cols) == 1:
        c = datetime_cols[0]
        roles["event_time"] = ResolvedRole(
            role="event_time", source="direct_column", expression=f'"{c.name}"', confidence=0.5,
            note=f"structural fallback: sole datetime-typed column '{c.name}' (LLM unavailable)",
        )
    return roles


def _compute_dimensions(profile: SchemaProfile, claimed: set[str]) -> list[Dimension]:
    dimensions = []
    for c in profile.columns:
        if c.name in claimed or c.kind != "text":
            continue
        if c.distinct_count <= _LOW_CARDINALITY_DIMENSION_MAX and c.null_fraction <= _MAX_DIMENSION_NULL_FRACTION:
            dimensions.append(Dimension(column=c.name, distinct_count=c.distinct_count, sample_values=c.sample_values))
    return dimensions


async def build_concept_map(profile: SchemaProfile, llm: LLMClient, model: str) -> ConceptMap:
    claimed: set[str] = set()
    try:
        proposal = await _propose_roles(profile, llm, model)
        roles: dict[str, ResolvedRole] = {}
        # Order matters: earlier roles claim their column first, so a
        # duplicate proposal for a later role is correctly rejected.
        for role, field_name in [
            ("event_group_id", "event_group_id"),
            ("entity_id", "entity_id"),
            ("item_id", "item_id"),
            ("item_label", "item_label"),
            ("quantity", "quantity"),
            ("unit_amount", "unit_amount"),
            ("event_time", "event_time"),
        ]:
            allowed = (
                _ID_LIKE_KINDS if role in ("event_group_id", "entity_id", "item_id")
                else ("text",) if role == "item_label"
                else _AMOUNT_LIKE_KINDS if role in ("quantity", "unit_amount")
                else ("datetime",)
            )
            roles[role] = _validate_simple_role(role, getattr(proposal, field_name), profile, claimed, allowed)
        roles["line_discount_rate"] = _validate_rate_role(
            "line_discount_rate", proposal.line_discount_rate, profile, claimed
        )
        roles["line_surcharge_rate"] = _validate_rate_role(
            "line_surcharge_rate", proposal.line_surcharge_rate, profile, claimed
        )
        roles["line_amount_adjustment"] = _validate_flat_adjustment(
            proposal.line_amount_adjustment, profile, claimed
        )
        roles["line_amount"] = _validate_line_amount(proposal.line_amount, profile, claimed, roles)
    except Exception as e:
        logger.warning("LLM role proposal failed for dataset %s (%s); using structural fallback", profile.dataset_id, e)
        roles = _structural_fallback_roles(profile)
        claimed = {r.expression.strip('"') for r in roles.values() if r.expression and r.source == "direct_column"}

    dimensions = _compute_dimensions(profile, claimed)
    claimed = claimed | {d.column for d in dimensions}

    unmapped = [
        UnmappedColumn(name=c.name, duckdb_type=c.duckdb_type, kind=c.kind, sample_values=c.sample_values)
        for c in profile.columns
        if c.name not in claimed
    ]

    return ConceptMap(
        dataset_id=profile.dataset_id, table_name=profile.table_name,
        roles=roles, dimensions=dimensions, unmapped=unmapped,
    )
