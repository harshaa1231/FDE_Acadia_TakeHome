"""Phrases a SQL result as a natural-language answer. Deliberately
constrained: the model is given only the executed query's own result rows
and told to describe them, never to add outside knowledge or numbers that
did not come from the query - the guard against a fluent-sounding but
invented answer.
"""
from __future__ import annotations

import json

from app.nlq.llm_client import LLMClient

SYSTEM_PROMPT = """\
You turn a SQL query's result into a short, direct natural-language \
answer to the question that was asked. You may state ONLY facts and \
numbers that literally appear in the result rows given to you - never add \
outside knowledge, never round or estimate beyond what's shown, never \
mention a figure that is not in the data. If the result is empty, say \
plainly that no matching data was found. Respond with the answer only, no \
SQL, no preamble like "Based on the query".

If the result is a single value or a short summary, answer in one to two \
plain sentences, stating the figure directly.

If the result has more than one row worth looking at individually (a top-N \
list, a breakdown by category, a set of matching records), do NOT restate \
every row's values in your answer - the caller already renders the full \
result as a table right below your answer, so repeating it is redundant. \
Instead, give ONLY a one-sentence lead-in that names what the table shows \
(e.g. "Here are the top 10 products by revenue, highest first." or "3 \
countries grew between Q1 and Q2 2011."), optionally naming the single \
standout row (the top or most notable one) if that directly answers the \
question, but leave the rest of the rows to the table."""


async def synthesize_answer(
    llm: LLMClient, model: str, question: str, sql: str, columns: list[str], rows: list[list]
) -> str:
    row_dicts = [dict(zip(columns, r)) for r in rows[:50]]
    user_prompt = (
        f"Question: {question}\n\n"
        f"SQL executed: {sql}\n\n"
        f"Result ({len(rows)} row(s), showing up to 50):\n{json.dumps(row_dicts, default=str)}"
    )
    return await llm.complete_text(SYSTEM_PROMPT, user_prompt, model=model)
