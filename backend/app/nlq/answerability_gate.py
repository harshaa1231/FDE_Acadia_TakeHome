"""A cheap, purely structural pre-check - not a keyword match against the
question's text (that would just reintroduce the same hardcoded-vocabulary
problem the semantic layer had to fix, one level up). It only asks: does
this dataset have *any* usable transactional structure at all? If not,
refuse instantly and skip the LLM round trip entirely.

Per-question answerability (does *this specific* question map to what's
available) is a language-understanding problem, not a structural one - that
decision is left to the SQL-generation LLM call itself, which is instructed
to emit a `REFUSE: <reason>` sentinel when it can't answer from the given
context. This gate only catches the degenerate case upstream.
"""
from __future__ import annotations

from app.semantic.models import ConceptMap


def dataset_has_minimal_structure(cmap: ConceptMap) -> tuple[bool, str]:
    has_grain = cmap.has("event_group_id") or cmap.has("entity_id") or cmap.has("item_id")
    has_measure = cmap.has("line_amount") or cmap.has("quantity")
    if has_grain and has_measure:
        return True, ""
    missing = []
    if not has_grain:
        missing.append("no identifiable transaction/entity/item grain")
    if not has_measure:
        missing.append("no monetary amount or quantity measure")
    return False, "This dataset has " + " and ".join(missing) + " - not enough structure to answer transactional questions about it."
