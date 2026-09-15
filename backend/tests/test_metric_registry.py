import pytest

from app.ingestion.models import ColumnProfile, SchemaProfile
from app.semantic.concept_mapper import build_concept_map
from app.semantic.metric_registry import resolve_available_metrics
from tests.test_concept_mapper import FakeLLMClient, role


def col(name, kind, distinct_count, row_count):
    return ColumnProfile(
        name=name, duckdb_type=kind.upper(), kind=kind, null_count=0, null_fraction=0.0,
        distinct_count=distinct_count, is_likely_unique=distinct_count >= row_count * 0.98,
        sample_values=[],
    )


@pytest.mark.asyncio
async def test_all_metrics_available_with_full_schema():
    p = SchemaProfile(
        dataset_id="d", table_name="t", row_count=1000,
        columns=[
            col("InvoiceNo", "text", 900, 1000),
            col("CustomerID", "integer", 300, 1000),
            col("StockCode", "text", 400, 1000),
            col("Quantity", "integer", 20, 1000),
            col("UnitPrice", "numeric", 150, 1000),
            col("InvoiceDate", "datetime", 900, 1000),
        ],
    )
    llm = FakeLLMClient(
        {
            "event_group_id": role("InvoiceNo"),
            "entity_id": role("CustomerID"),
            "item_id": role("StockCode"),
            "quantity": role("Quantity"),
            "unit_amount": role("UnitPrice"),
            "line_amount": role(None, derive_from_quantity_and_unit_amount=True),
            "event_time": role("InvoiceDate"),
        }
    )
    cmap = await build_concept_map(p, llm, model="fake")
    metrics = {m.name: m for m in resolve_available_metrics(cmap)}
    assert metrics["net_revenue"].available
    assert metrics["unique_customers"].available
    assert metrics["order_count"].available
    assert metrics["average_order_value"].available


@pytest.mark.asyncio
async def test_customer_dependent_metrics_unavailable_without_entity_id():
    p = SchemaProfile(
        dataset_id="d", table_name="t", row_count=1000,
        columns=[
            col("order_ref", "text", 900, 1000),
            col("sku", "text", 400, 1000),
            col("qty", "integer", 20, 1000),
            col("price", "numeric", 150, 1000),
        ],
    )
    llm = FakeLLMClient(
        {
            "event_group_id": role("order_ref"),
            "item_id": role("sku"),
            "quantity": role("qty"),
            "unit_amount": role("price"),
            "line_amount": role(None, derive_from_quantity_and_unit_amount=True),
        }
    )
    cmap = await build_concept_map(p, llm, model="fake")
    metrics = {m.name: m for m in resolve_available_metrics(cmap)}
    assert not metrics["unique_customers"].available
    assert metrics["unique_customers"].missing_roles == ["entity_id"]
    assert metrics["net_revenue"].available


@pytest.mark.asyncio
async def test_no_metrics_available_on_a_bare_schema():
    p = SchemaProfile(
        dataset_id="d", table_name="t", row_count=1000,
        columns=[col("notes", "text", 900, 1000)],
    )
    llm = FakeLLMClient({})
    cmap = await build_concept_map(p, llm, model="fake")
    metrics = resolve_available_metrics(cmap)
    assert all(not m.available for m in metrics)
