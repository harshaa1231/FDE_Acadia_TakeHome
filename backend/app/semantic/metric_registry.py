"""Named metrics defined purely in terms of resolved structural roles, never
raw column names. This is the extensibility hook the design doc points to:
adding a new metric means adding one entry here, not touching prompts, SQL
generation, or the API - and a metric request during the live feature-ask
lands in exactly this file.

Each metric declares the roles it needs. If a role isn't resolved for the
dataset, the metric is simply unavailable for it - the answerability gate
uses that to refuse cleanly instead of asking the LLM to invent a column.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.semantic.models import ConceptMap


@dataclass
class MetricDefinition:
    name: str
    description: str
    required_roles: list[str]
    build_expression: Callable[[ConceptMap], str]


@dataclass
class AvailableMetric:
    name: str
    description: str
    sql_expression: str | None
    available: bool
    missing_roles: list[str]


def _net_revenue(cmap: ConceptMap) -> str:
    return f"SUM({cmap.get('line_amount').expression})"


def _units_sold(cmap: ConceptMap) -> str:
    return f"SUM({cmap.get('quantity').expression})"


def _order_count(cmap: ConceptMap) -> str:
    return f"COUNT(DISTINCT {cmap.get('event_group_id').expression})"


def _unique_customers(cmap: ConceptMap) -> str:
    return f"COUNT(DISTINCT {cmap.get('entity_id').expression})"


def _average_order_value(cmap: ConceptMap) -> str:
    return (
        f"SUM({cmap.get('line_amount').expression}) / "
        f"NULLIF(COUNT(DISTINCT {cmap.get('event_group_id').expression}), 0)"
    )


def _average_unit_amount(cmap: ConceptMap) -> str:
    return f"AVG({cmap.get('unit_amount').expression})"


REGISTRY: list[MetricDefinition] = [
    MetricDefinition("net_revenue", "Sum of line-level monetary amount", ["line_amount"], _net_revenue),
    MetricDefinition("units_sold", "Sum of quantity", ["quantity"], _units_sold),
    MetricDefinition("order_count", "Count of distinct grouping/order ids", ["event_group_id"], _order_count),
    MetricDefinition("unique_customers", "Count of distinct entity/customer ids", ["entity_id"], _unique_customers),
    MetricDefinition(
        "average_order_value", "Net revenue divided by order count",
        ["line_amount", "event_group_id"], _average_order_value,
    ),
    MetricDefinition(
        "average_unit_amount", "Average per-unit monetary amount",
        ["unit_amount"], _average_unit_amount,
    ),
]


def resolve_available_metrics(cmap: ConceptMap) -> list[AvailableMetric]:
    out: list[AvailableMetric] = []
    for metric in REGISTRY:
        missing = [r for r in metric.required_roles if not cmap.has(r)]
        expr = None
        if not missing:
            try:
                expr = metric.build_expression(cmap)
            except Exception:  # a role resolved but the expression build still failed
                missing = metric.required_roles
        out.append(
            AvailableMetric(
                name=metric.name,
                description=metric.description,
                sql_expression=expr,
                available=not missing,
                missing_roles=missing,
            )
        )
    return out
