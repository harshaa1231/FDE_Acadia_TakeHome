"""Tests for UsageStats - the real, measured token accounting behind the
"cost tracking" surface (job result + trace). No live API calls: this
tests the accumulation and reporting logic in isolation.
"""
from app.nlq.llm_client import UsageStats


def test_accumulates_across_multiple_calls_and_models():
    usage = UsageStats()
    usage.record("model-a", prompt_tokens=100, completion_tokens=20)
    usage.record("model-b", prompt_tokens=50, completion_tokens=10)
    usage.record("model-a", prompt_tokens=30, completion_tokens=5)

    assert usage.prompt_tokens == 180
    assert usage.completion_tokens == 35
    assert usage.total_tokens == 215
    assert usage.calls == 3
    assert usage.per_model == {"model-a": 155, "model-b": 60}


def test_to_dict_has_no_cost_when_unconfigured(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "sql_model_cost_per_1k_tokens", 0.0)
    monkeypatch.setattr(settings, "fast_model_cost_per_1k_tokens", 0.0)
    monkeypatch.setattr(settings, "sql_model", "sql-model")
    monkeypatch.setattr(settings, "fast_model", "fast-model")

    usage = UsageStats()
    usage.record("sql-model", prompt_tokens=1000, completion_tokens=200)
    d = usage.to_dict()

    assert d["total_tokens"] == 1200
    assert "estimated_cost_usd" not in d  # honest: no fabricated price


def test_to_dict_computes_cost_when_rates_are_configured(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "sql_model", "sql-model")
    monkeypatch.setattr(settings, "fast_model", "fast-model")
    monkeypatch.setattr(settings, "sql_model_cost_per_1k_tokens", 0.50)
    monkeypatch.setattr(settings, "fast_model_cost_per_1k_tokens", 0.10)

    usage = UsageStats()
    usage.record("sql-model", prompt_tokens=1000, completion_tokens=1000)  # 2000 tokens
    usage.record("fast-model", prompt_tokens=500, completion_tokens=500)  # 1000 tokens

    d = usage.to_dict()
    # 2000/1000 * 0.50 + 1000/1000 * 0.10 = 1.0 + 0.10 = 1.10
    assert d["estimated_cost_usd"] == 1.10
