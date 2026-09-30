"""A slow first provider gets company from the next one (LLM_HEDGE): the first answer wins, the other is cancelled."""

import asyncio
import time

import pytest


@pytest.fixture
def providers(monkeypatch):
    from tulpar_ai import llm
    from tulpar_ai.config import get_settings

    for k, v in {"OLLAMA_API_KEY": "x", "GROQ_API_KEY": "x", "ROUTE_MODELS": "ollama:primary,groq:backup",
                 "VISION_MODELS": "ollama:primary,groq:backup", "LLM_HEDGE": "route:0.2",
                 "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    llm.set_fake(None)
    plan = {"primary": (0.0, None), "backup": (0.0, None)}  # model → (delay, exception)
    calls, cancelled = [], []

    def make(name):
        async def call(model, *a, **kw):
            calls.append(model)
            delay, exc = plan[model]
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                cancelled.append(model)
                raise
            if exc:
                raise exc
            return f'{{"from": "{model}"}}', 10, 5
        return call

    monkeypatch.setitem(llm._PROVIDERS, "ollama", make("ollama"))
    monkeypatch.setitem(llm._PROVIDERS, "groq", make("groq"))
    yield plan, calls, cancelled
    get_settings.cache_clear()


async def test_slow_primary_is_overtaken_by_the_backup(providers):
    from tulpar_ai import llm

    plan, calls, cancelled = providers
    plan["primary"] = (3.0, None)
    t0 = time.perf_counter()
    data, res = await llm.json_call("route", "sys", "user")
    assert data == {"from": "backup"} and res.hedged and res.fallback_used and res.model == "backup"
    assert time.perf_counter() - t0 < 1.0  # 0.2 s of patience + the backup, not the primary's 3 s
    await asyncio.sleep(0)
    assert cancelled == ["primary"]


async def test_fast_primary_asks_nobody_else(providers):
    from tulpar_ai import llm

    plan, calls, _ = providers
    data, res = await llm.json_call("route", "sys", "user")
    assert data == {"from": "primary"} and not res.hedged and not res.fallback_used and calls == ["primary"]


async def test_primary_that_answers_first_still_wins(providers):
    from tulpar_ai import llm

    plan, calls, cancelled = providers
    plan["primary"], plan["backup"] = (0.4, None), (2.0, None)
    data, res = await llm.json_call("route", "sys", "user")
    assert data == {"from": "primary"} and res.hedged and not res.fallback_used
    await asyncio.sleep(0)
    assert calls == ["primary", "backup"] and cancelled == ["backup"]


async def test_a_failure_falls_back_at_once(providers):
    from tulpar_ai import llm

    plan, calls, _ = providers
    plan["primary"] = (0.0, RuntimeError("503"))
    data, res = await llm.json_call("route", "sys", "user")
    assert data == {"from": "backup"} and not res.hedged and res.fallback_used
    assert any("503" in e for e in res.errors)


async def test_both_failing_is_an_error(providers):
    from tulpar_ai import llm

    plan, _, _ = providers
    plan["primary"], plan["backup"] = (0.5, RuntimeError("slow and broken")), (0.0, RuntimeError("down"))
    with pytest.raises(llm.LLMError) as e:
        await llm.json_call("route", "sys", "user")
    assert "slow and broken" in str(e.value) and "down" in str(e.value)


async def test_roles_without_hedging_wait_for_their_provider(providers):
    from tulpar_ai import llm

    plan, calls, _ = providers
    plan["primary"] = (0.5, None)
    data, res = await llm.json_call("vision", "sys", "user")
    assert data == {"from": "primary"} and not res.hedged and calls == ["primary"]
