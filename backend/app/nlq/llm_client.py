"""Thin, provider-agnostic LLM client wrapper. Every call site in the app
goes through this interface, not the Groq SDK directly - swapping providers
later (or injecting a fake for tests) never touches the semantic layer,
prompt builder, or guardrail code.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Accumulates real token usage across every call made through one
    client instance - each job handler creates its own client via
    get_llm_client() and reads this off at the end, so it naturally scopes
    to "tokens this ingest/question actually cost", not a global counter."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    per_model: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.calls += 1
        self.per_model[model] = self.per_model.get(model, 0) + prompt_tokens + completion_tokens

    def to_dict(self) -> dict:
        d = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.calls,
            "tokens_by_model": self.per_model,
        }
        cost = self.estimated_cost_usd()
        if cost is not None:
            d["estimated_cost_usd"] = round(cost, 6)
        return d

    def estimated_cost_usd(self) -> float | None:
        """Only computed if the deployer has configured a real price via
        env vars - we don't fabricate a $/token rate we haven't verified,
        so an unconfigured deployment reports tokens only, honestly."""
        rates = {settings.sql_model: settings.sql_model_cost_per_1k_tokens, settings.fast_model: settings.fast_model_cost_per_1k_tokens}
        if not any(rates.values()):
            return None
        return sum(tokens / 1000 * rates.get(model, 0.0) for model, tokens in self.per_model.items())


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
        self.usage = UsageStats()

    def _record_usage(self, model: str, resp) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        self.usage.record(model, usage.prompt_tokens or 0, usage.completion_tokens or 0)

    async def complete_json(self, system: str, user: str, model: str) -> dict:
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0,
            timeout=settings.llm_request_timeout_s,
        )
        self._record_usage(model, resp)
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
        self._record_usage(model, resp)
        return resp.choices[0].message.content


def get_llm_client() -> LLMClient:
    return GroqLLMClient()
