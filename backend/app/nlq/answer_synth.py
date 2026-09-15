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
plainly that no matching data was found. Respond with the answer only, \
one to three sentences, no SQL, no preamble like "Based on the query"."""


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
