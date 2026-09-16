"""Tests for the LLM-driven concept mapper. Since role *proposal* comes
from an LLM, these tests use a FakeLLMClient (no network calls, no API
key needed) and focus on what's actually deterministic and safety-critical:
the validation layer that decides whether to trust a proposal. A bad or
malicious proposal (nonexistent column, wrong type, duplicate claim) must
never reach the ConceptMap - that's the guardrail these tests prove.
"""
import pytest

from app.ingestion.models import ColumnProfile, SchemaProfile
from app.semantic.concept_mapper import _roles_array_to_dict, build_concept_map


def col(name, kind, distinct_count, row_count, sample_values=None, null_count=0, min_value=None, max_value=None):
    return ColumnProfile(
        name=name, duckdb_type=kind.upper(), kind=kind,
        null_count=null_count, null_fraction=null_count / row_count if row_count else 0.0,
        distinct_count=distinct_count, is_likely_unique=distinct_count >= row_count * 0.98,
        sample_values=sample_values or [], min_value=min_value, max_value=max_value,
    )


def profile(columns, row_count, dataset_id="ds", table_name="t"):
    return SchemaProfile(dataset_id=dataset_id, table_name=table_name, row_count=row_count, columns=columns)


class FakeLLMClient:
    def __init__(self, response: dict):
        self.response = response
        self.calls = 0

    async def complete_json(self, system: str, user: str, model: str) -> dict:
        self.calls += 1
        return self.response

    async def complete_text(self, system: str, user: str, model: str) -> str:
        raise NotImplementedError


def role(column=None, confidence=0.9, reasoning="test", **extra):
    d = {"column": column, "confidence": confidence, "reasoning": reasoning}
    d.update(extra)
    return d


def test_roles_array_wire_format_is_unwrapped_to_a_flat_dict():
    """The real system prompt now asks the LLM for {"roles": [{"role": "x",
    ...}, ...]} instead of a flat {"x": {...}} object - reproduced live
    against a 25-column real dataset, the flat-object shape reliably made
    the fast model emit malformed JSON (a stray '{' before each subsequent
    key) because an object with many identically-shaped keys reads to the
    model like it wants to be a list. This is the contract translation
    layer between that wire format and the rest of the module."""
    raw = {
        "roles": [
            {"role": "event_group_id", "column": "invoice_no", "confidence": 0.9, "reasoning": "groups rows"},
            {
                "role": "line_amount", "column": None, "derive_from_quantity_and_unit_amount": True,
                "confidence": 0.8, "reasoning": "qty*price",
            },
        ]
    }
    out = _roles_array_to_dict(raw)
    assert out == {
        "event_group_id": {"column": "invoice_no", "confidence": 0.9, "reasoning": "groups rows"},
        "line_amount": {
            "column": None, "derive_from_quantity_and_unit_amount": True,
            "confidence": 0.8, "reasoning": "qty*price",
        },
    }


def test_roles_array_wire_format_ignores_entries_with_no_role_name():
    raw = {"roles": [{"column": "foo", "confidence": 0.5, "reasoning": "no role key"}]}
    assert _roles_array_to_dict(raw) == {}


def test_flat_dict_wire_format_still_passes_through_unchanged():
    """Every existing test fixture, and the structural fallback path, builds
    the old flat {"event_group_id": {...}, ...} shape directly - it must
    keep working exactly as before."""
    raw = {"event_group_id": {"column": "invoice_no", "confidence": 0.9, "reasoning": "x"}}
    assert _roles_array_to_dict(raw) == raw


RIDESHARE_COLUMNS = [
    col("trip_ref", "text", 2900, 3000),
    col("rider_handle", "text", 60, 3000, sample_values=["rider_21@mail.com"]),
    col("driver_id", "text", 30, 3000, sample_values=["D1005"]),
    col("city", "text", 5, 3000),
    col("fare_amount", "numeric", 2000, 3000),
    col("tip_amount", "numeric", 6, 3000),
    col("requested_at", "datetime", 2990, 3000),
]


@pytest.mark.asyncio
async def test_build_concept_map_end_to_end_with_real_array_wire_format():
    """Unlike every other test in this file (which stub the LLM with the
    internal flat-dict shape for convenience), this one uses the actual
    {"roles": [...]} wire format the real system prompt asks for, proving
    the full pipeline - not just the transform function in isolation -
    correctly consumes it."""
    llm = FakeLLMClient(
        {
            "roles": [
                {"role": "event_group_id", "column": "trip_ref", "confidence": 0.9, "reasoning": "groups rows"},
                {"role": "entity_id", "column": "rider_handle", "confidence": 0.9, "reasoning": "the rider"},
                {"role": "unit_amount", "column": "fare_amount", "confidence": 0.9, "reasoning": "per-fare price"},
                {
                    "role": "line_amount", "column": "fare_amount", "confidence": 0.9,
                    "derive_from_quantity_and_unit_amount": False, "reasoning": "same column",
                },
            ]
        }
    )
    p = profile(RIDESHARE_COLUMNS, 3000)
    cmap = await build_concept_map(p, llm, model="fake")
    assert cmap.get("event_group_id").expression == '"trip_ref"'
    assert cmap.get("entity_id").expression == '"rider_handle"'
    assert cmap.get("line_amount").expression == '"fare_amount"'


@pytest.mark.asyncio
async def test_llm_proposal_accepted_when_structurally_valid():
    llm = FakeLLMClient(
        {
            "event_group_id": role("trip_ref"),
            "entity_id": role("rider_handle", reasoning="email-like handle identifying the rider"),
            "item_id": role(None),
            "item_label": role(None),
            "quantity": role(None),
            "unit_amount": role("fare_amount"),
            "line_amount": role("fare_amount"),
            "event_time": role("requested_at"),
        }
    )
    p = profile(RIDESHARE_COLUMNS, 3000)
    cmap = await build_concept_map(p, llm, model="fake")
    assert cmap.get("event_group_id").expression == '"trip_ref"'
    assert cmap.get("entity_id").expression == '"rider_handle"'
    assert cmap.get("line_amount").expression == '"fare_amount"'
    assert cmap.get("item_id") is None


@pytest.mark.asyncio
async def test_proposal_for_nonexistent_column_is_rejected():
    llm = FakeLLMClient({"entity_id": role("customer_id_that_does_not_exist")})
    p = profile(RIDESHARE_COLUMNS, 3000)
    cmap = await build_concept_map(p, llm, model="fake")
    assert cmap.get("entity_id") is None
    assert cmap.roles["entity_id"].source == "not_found"


@pytest.mark.asyncio
async def test_proposal_with_wrong_type_is_rejected():
    """The LLM proposes a text column for a numeric-only role - the
    validator must catch this even though the LLM was confident."""
    llm = FakeLLMClient({"quantity": role("city", confidence=0.95)})
    p = profile(RIDESHARE_COLUMNS, 3000)
    cmap = await build_concept_map(p, llm, model="fake")
    assert cmap.get("quantity") is None
    assert "kind" in cmap.roles["quantity"].note


@pytest.mark.asyncio
async def test_duplicate_claim_second_role_rejected():
    """The LLM proposes the same column for two different roles - only the
    first-processed role may claim it."""
    llm = FakeLLMClient(
        {
            "entity_id": role("rider_handle"),
            "item_id": role("rider_handle"),  # same column, should be rejected here
        }
    )
    p = profile(RIDESHARE_COLUMNS, 3000)
    cmap = await build_concept_map(p, llm, model="fake")
    assert cmap.get("entity_id").expression == '"rider_handle"'
    assert cmap.get("item_id") is None
    assert "already claimed" in cmap.roles["item_id"].note


@pytest.mark.asyncio
async def test_line_amount_derived_from_quantity_and_unit_amount():
    llm = FakeLLMClient(
        {
            "quantity": role("tip_amount"),  # reuse a numeric col to stand in for a real qty
            "unit_amount": role("fare_amount"),
            "line_amount": role(None, derive_from_quantity_and_unit_amount=True),
        }
    )
    p = profile(RIDESHARE_COLUMNS, 3000)
    cmap = await build_concept_map(p, llm, model="fake")
    line = cmap.get("line_amount")
    assert line.source == "derived_expression"
    assert "tip_amount" in line.expression and "fare_amount" in line.expression


@pytest.mark.asyncio
async def test_line_amount_derivation_refused_without_both_roles():
    """derive_from_quantity_and_unit_amount=true is meaningless if one of
    the two roles it depends on was never resolved - must not fabricate."""
    llm = FakeLLMClient(
        {
            "unit_amount": role("fare_amount"),
            "line_amount": role(None, derive_from_quantity_and_unit_amount=True),
        }
    )
    p = profile(RIDESHARE_COLUMNS, 3000)
    cmap = await build_concept_map(p, llm, model="fake")
    assert cmap.get("line_amount") is None


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_conservative_structural_inference():
    class BrokenLLMClient:
        async def complete_json(self, system, user, model):
            raise RuntimeError("simulated network failure")

        async def complete_text(self, system, user, model):
            raise RuntimeError("simulated network failure")

    p = profile(RIDESHARE_COLUMNS, 3000)
    cmap = await build_concept_map(p, BrokenLLMClient(), model="fake")
    # Only the unambiguous single datetime column should resolve.
    assert cmap.get("event_time").expression == '"requested_at"'
    assert cmap.get("entity_id") is None
    assert cmap.get("item_id") is None
    assert cmap.roles["entity_id"].note != ""


@pytest.mark.asyncio
async def test_transient_malformed_json_is_retried_once_before_falling_back():
    """Reproduces a real failure seen against an external dataset: the LLM
    returned malformed JSON on the first attempt (a dropped comma) under
    response_format=json_object. A single transient glitch should not
    degrade a perfectly resolvable dataset to the conservative fallback -
    one retry should recover it."""

    class FlakyOnceLLMClient:
        def __init__(self, good_response: dict):
            self.good_response = good_response
            self.calls = 0

        async def complete_json(self, system, user, model):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("LLM returned malformed JSON: dropped comma")
            return self.good_response

        async def complete_text(self, system, user, model):
            raise NotImplementedError

    llm = FlakyOnceLLMClient({"event_group_id": role("trip_ref"), "unit_amount": role("fare_amount")})
    p = profile(RIDESHARE_COLUMNS, 3000)
    cmap = await build_concept_map(p, llm, model="fake")

    assert llm.calls == 2
    assert cmap.get("event_group_id").expression == '"trip_ref"'
    assert cmap.get("unit_amount").expression == '"fare_amount"'


@pytest.mark.asyncio
async def test_persistent_failure_still_falls_back_after_retry():
    class AlwaysBrokenLLMClient:
        def __init__(self):
            self.calls = 0

        async def complete_json(self, system, user, model):
            self.calls += 1
            raise ValueError("still broken")

        async def complete_text(self, system, user, model):
            raise NotImplementedError

    llm = AlwaysBrokenLLMClient()
    p = profile(RIDESHARE_COLUMNS, 3000)
    cmap = await build_concept_map(p, llm, model="fake")

    assert llm.calls == 2  # retried once, then gave up and fell back
    assert cmap.get("event_time").expression == '"requested_at"'
    assert cmap.get("entity_id") is None


@pytest.mark.asyncio
async def test_no_role_matches_leaves_everything_not_found_but_no_crash():
    llm = FakeLLMClient({})
    p = profile([col("notes", "text", 900, 1000)], 1000)
    cmap = await build_concept_map(p, llm, model="fake")
    assert all(cmap.get(r) is None for r in cmap.roles)


@pytest.mark.asyncio
async def test_unclaimed_low_cardinality_text_columns_become_dimensions():
    llm = FakeLLMClient({"event_group_id": role("trip_ref")})
    p = profile(RIDESHARE_COLUMNS, 3000)
    cmap = await build_concept_map(p, llm, model="fake")
    dim_cols = {d.column for d in cmap.dimensions}
    assert "city" in dim_cols
    # rider_handle/driver_id weren't claimed by any role in this response,
    # and they're low-cardinality text, so they still surface as dimensions
    # rather than being silently dropped.
    assert "rider_handle" in dim_cols
    assert "driver_id" in dim_cols


DISCOUNTED_RETAIL_COLUMNS = [
    col("order_ref", "text", 1400, 2625),
    col("buyer_ref", "text", 300, 2625),
    col("units_sold", "integer", 10, 2625, min_value=1, max_value=10),
    col("unit_price", "numeric", 200, 2625, min_value=1.0, max_value=500.0),
    col("promo_rate", "numeric", 4, 2625, sample_values=[0.0, 0.05, 0.1], min_value=0.0, max_value=0.15),
    col("purchased_on", "datetime", 300, 2625),
]


@pytest.mark.asyncio
async def test_line_amount_derivation_applies_discount_rate_when_resolved():
    """Reproduces a real bug found against an external dataset: a
    row-level discount/promo rate column was silently ignored because no
    structural role existed for it, so 'net revenue' was computed as gross
    (qty * unit_price) instead of qty * unit_price * (1 - rate) - overstating
    revenue by the full discount amount on every row."""
    llm = FakeLLMClient(
        {
            "event_group_id": role("order_ref"),
            "entity_id": role("buyer_ref"),
            "quantity": role("units_sold"),
            "unit_amount": role("unit_price"),
            "line_discount_rate": role("promo_rate"),
            "line_amount": role(None, derive_from_quantity_and_unit_amount=True),
            "event_time": role("purchased_on"),
        }
    )
    p = profile(DISCOUNTED_RETAIL_COLUMNS, 2625)
    cmap = await build_concept_map(p, llm, model="fake")

    assert cmap.get("line_discount_rate").expression == '"promo_rate"'
    line = cmap.get("line_amount")
    assert line.source == "derived_expression"
    assert line.expression == '("units_sold" * "unit_price" * (1 - "promo_rate"))'


@pytest.mark.asyncio
async def test_line_amount_derivation_without_discount_rate_stays_gross():
    """When no discount rate is proposed (or none exists), line_amount must
    stay a plain qty * unit_amount - the discount factor is additive, never
    assumed."""
    llm = FakeLLMClient(
        {
            "quantity": role("units_sold"),
            "unit_amount": role("unit_price"),
            "line_amount": role(None, derive_from_quantity_and_unit_amount=True),
        }
    )
    p = profile(DISCOUNTED_RETAIL_COLUMNS, 2625)
    cmap = await build_concept_map(p, llm, model="fake")
    line = cmap.get("line_amount")
    assert line.expression == '("units_sold" * "unit_price")'
    assert cmap.get("line_discount_rate") is None


@pytest.mark.asyncio
async def test_discount_rate_proposal_outside_zero_to_one_range_rejected():
    """A numeric column outside [0, 1] cannot be a fractional discount rate
    (e.g. a column of raw discount amounts in dollars, or a percentage
    stored as 0-100) - the structural check must catch this regardless of
    what the LLM proposed or how confident it was."""
    columns = DISCOUNTED_RETAIL_COLUMNS + [
        col("discount_amount_dollars", "numeric", 50, 2625, min_value=0.0, max_value=120.0)
    ]
    llm = FakeLLMClient(
        {
            "quantity": role("units_sold"),
            "unit_amount": role("unit_price"),
            "line_discount_rate": role("discount_amount_dollars", confidence=0.9),
            "line_amount": role(None, derive_from_quantity_and_unit_amount=True),
        }
    )
    p = profile(columns, 2625)
    cmap = await build_concept_map(p, llm, model="fake")
    assert cmap.get("line_discount_rate") is None
    assert "outside [0, 1]" in cmap.roles["line_discount_rate"].note
    line = cmap.get("line_amount")
    assert line.expression == '("units_sold" * "unit_price")'


@pytest.mark.asyncio
async def test_discount_rate_not_applied_when_line_amount_is_a_direct_column():
    """If line_amount already comes from a direct total column, we don't
    know whether that total already reflects the discount - applying the
    rate again would risk double-discounting, so it must be left alone."""
    columns = DISCOUNTED_RETAIL_COLUMNS + [
        col("line_total", "numeric", 2000, 2625, min_value=1.0, max_value=4000.0)
    ]
    llm = FakeLLMClient(
        {
            "line_discount_rate": role("promo_rate"),
            "line_amount": role("line_total"),
        }
    )
    p = profile(columns, 2625)
    cmap = await build_concept_map(p, llm, model="fake")
    assert cmap.get("line_discount_rate").expression == '"promo_rate"'
    assert cmap.get("line_amount").expression == '"line_total"'


@pytest.mark.asyncio
async def test_line_amount_derivation_applies_surcharge_rate_when_resolved():
    """A tax/VAT/markup rate column is the mirror-image case of a discount:
    it should increase the derived total, not decrease it."""
    columns = DISCOUNTED_RETAIL_COLUMNS + [
        col("tax_rate", "numeric", 3, 2625, sample_values=[0.0, 0.07, 0.08], min_value=0.0, max_value=0.08)
    ]
    llm = FakeLLMClient(
        {
            "quantity": role("units_sold"),
            "unit_amount": role("unit_price"),
            "line_surcharge_rate": role("tax_rate"),
            "line_amount": role(None, derive_from_quantity_and_unit_amount=True),
        }
    )
    p = profile(columns, 2625)
    cmap = await build_concept_map(p, llm, model="fake")
    assert cmap.get("line_surcharge_rate").expression == '"tax_rate"'
    line = cmap.get("line_amount")
    assert line.source == "derived_expression"
    assert line.expression == '("units_sold" * "unit_price" * (1 + "tax_rate"))'


@pytest.mark.asyncio
async def test_line_amount_derivation_applies_flat_adjustment_when_resolved():
    """A flat shipping fee (or flat discount/refund) column adds/subtracts
    a raw monetary delta rather than scaling the total by a fraction."""
    columns = DISCOUNTED_RETAIL_COLUMNS + [
        col("shipping_fee", "numeric", 5, 2625, min_value=0.0, max_value=15.0)
    ]
    llm = FakeLLMClient(
        {
            "quantity": role("units_sold"),
            "unit_amount": role("unit_price"),
            "line_amount_adjustment": role("shipping_fee"),
            "line_amount": role(None, derive_from_quantity_and_unit_amount=True),
        }
    )
    p = profile(columns, 2625)
    cmap = await build_concept_map(p, llm, model="fake")
    assert cmap.get("line_amount_adjustment").expression == '"shipping_fee"'
    line = cmap.get("line_amount")
    assert line.source == "derived_expression"
    assert line.expression == '("units_sold" * "unit_price" + "shipping_fee")'


@pytest.mark.asyncio
async def test_line_amount_derivation_composes_all_three_adjustments_together():
    """The realistic worst case: a dataset with a discount, a tax, and a
    flat fee all at once. All three must compose in one expression, applied
    in a fixed, predictable order (discount, then surcharge, then flat)."""
    columns = DISCOUNTED_RETAIL_COLUMNS + [
        col("tax_rate", "numeric", 3, 2625, min_value=0.0, max_value=0.08),
        col("shipping_fee", "numeric", 5, 2625, min_value=0.0, max_value=15.0),
    ]
    llm = FakeLLMClient(
        {
            "quantity": role("units_sold"),
            "unit_amount": role("unit_price"),
            "line_discount_rate": role("promo_rate"),
            "line_surcharge_rate": role("tax_rate"),
            "line_amount_adjustment": role("shipping_fee"),
            "line_amount": role(None, derive_from_quantity_and_unit_amount=True),
        }
    )
    p = profile(columns, 2625)
    cmap = await build_concept_map(p, llm, model="fake")
    line = cmap.get("line_amount")
    assert line.expression == (
        '("units_sold" * "unit_price" * (1 - "promo_rate") * (1 + "tax_rate") + "shipping_fee")'
    )


@pytest.mark.asyncio
async def test_surcharge_rate_proposal_outside_zero_to_one_range_rejected():
    columns = DISCOUNTED_RETAIL_COLUMNS + [
        col("markup_multiplier", "numeric", 20, 2625, min_value=1.0, max_value=3.5)
    ]
    llm = FakeLLMClient(
        {
            "quantity": role("units_sold"),
            "unit_amount": role("unit_price"),
            "line_surcharge_rate": role("markup_multiplier", confidence=0.9),
            "line_amount": role(None, derive_from_quantity_and_unit_amount=True),
        }
    )
    p = profile(columns, 2625)
    cmap = await build_concept_map(p, llm, model="fake")
    assert cmap.get("line_surcharge_rate") is None
    line = cmap.get("line_amount")
    assert line.expression == '("units_sold" * "unit_price")'


@pytest.mark.asyncio
async def test_flat_adjustment_proposal_with_non_numeric_column_rejected():
    llm = FakeLLMClient(
        {
            "quantity": role("units_sold"),
            "unit_amount": role("unit_price"),
            "line_amount_adjustment": role("order_ref", confidence=0.9),  # a text id, not an amount
            "line_amount": role(None, derive_from_quantity_and_unit_amount=True),
        }
    )
    p = profile(DISCOUNTED_RETAIL_COLUMNS, 2625)
    cmap = await build_concept_map(p, llm, model="fake")
    assert cmap.get("line_amount_adjustment") is None
    line = cmap.get("line_amount")
    assert line.expression == '("units_sold" * "unit_price")'


@pytest.mark.asyncio
async def test_flat_adjustment_applied_when_line_amount_aliases_unit_amount():
    """Reproduces a real bug found against an external dataset (Olist order
    items - no explicit quantity column, so unit_amount ("price") aliases
    directly as line_amount with an implicit qty=1). A resolved
    line_amount_adjustment (a per-item "freight_value" shipping fee) must
    still be folded in here exactly as it would be in the fully-derived
    quantity * unit_amount case - it was previously silently dropped
    because only that other branch composed adjustments."""
    llm = FakeLLMClient(
        {
            "unit_amount": role("fare_amount"),
            "line_amount_adjustment": role("booking_fee"),
            "line_amount": role("fare_amount"),  # same column as unit_amount
        }
    )
    columns = RIDESHARE_COLUMNS + [col("booking_fee", "numeric", 4, 3000, min_value=0.0, max_value=5.0)]
    p = profile(columns, 3000)
    cmap = await build_concept_map(p, llm, model="fake")
    line = cmap.get("line_amount")
    assert line.source == "derived_expression"
    assert line.expression == '("fare_amount" + "booking_fee")'


@pytest.mark.asyncio
async def test_line_amount_aliases_unit_amount_unchanged_without_adjustments():
    """When no adjustment role resolves, the aliasing case must stay a
    plain direct-column reference (not wrapped in redundant parens/math) -
    no behavior change for every dataset that looked like this before
    adjustment roles existed."""
    llm = FakeLLMClient({"unit_amount": role("fare_amount"), "line_amount": role("fare_amount")})
    p = profile(RIDESHARE_COLUMNS, 3000)
    cmap = await build_concept_map(p, llm, model="fake")
    line = cmap.get("line_amount")
    assert line.source == "direct_column"
    assert line.expression == '"fare_amount"'


@pytest.mark.asyncio
async def test_unmapped_columns_never_dropped():
    llm = FakeLLMClient({"event_group_id": role("trip_ref"), "unit_amount": role("fare_amount")})
    p = profile(RIDESHARE_COLUMNS, 3000)
    cmap = await build_concept_map(p, llm, model="fake")
    all_surfaced = {d.column for d in cmap.dimensions} | {u.name for u in cmap.unmapped} | {
        c.expression.strip('"') for c in cmap.roles.values() if c.expression
    }
    all_names = {c.name for c in RIDESHARE_COLUMNS}
    assert all_names <= all_surfaced
