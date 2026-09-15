"""Thin, provider-agnostic LLM client wrapper. Every call site in the app
goes through this interface, not the Groq SDK directly - swapping providers
later (or injecting a fake for tests) never touches the semantic layer,
prompt builder, or guardrail code.
"""
from __future__ import annotations

import json
import logging
from typing import Protocol

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    async def complete_json(self, system: str, user: str, model: str) -> dict:
        """Returns a parsed JSON object. Raises on network/parse failure -
        callers are responsible for falling back gracefully."""
        ...

    async def complete_text(self, system: str, user: str, model: str) -> str:
        ...


class GroqLLMClient:
    def __init__(self, api_key: str | None = None):
        from groq import AsyncGroq  # imported lazily so tests never need the SDK installed

        self._client = AsyncGroq(api_key=api_key or settings.groq_api_key)

    async def complete_json(self, system: str, user: str, model: str) -> dict:
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0,
            timeout=settings.llm_request_timeout_s,
        )
        text = resp.choices[0].message.content
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("LLM returned non-JSON despite json_object mode: %s", text[:500])
            raise ValueError(f"LLM did not return valid JSON: {e}") from e

    async def complete_text(self, system: str, user: str, model: str) -> str:
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            timeout=settings.llm_request_timeout_s,
        )
        return resp.choices[0].message.content


def get_llm_client() -> LLMClient:
    return GroqLLMClient()
