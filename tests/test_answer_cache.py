"""Semantic answer cache: hits skip the answer model, near-misses and non-answers never come from the cache."""

import time

import pytest
import pytest_asyncio

Q = "Как правильно жать штангу лёжа? Под каким углом держать локти?"
NEAR = "Как правильно жать штангу стоя? Под каким углом держать локти?"
REPLY = "Локти держите примерно под углом 45° к корпусу [1].\n\nЭто общая информация, а не медицинская консультация."
CITES = [{"n": 1, "title": "Жим штанги лёжа", "page": None, "source": "exercises"}]


@pytest_asyncio.fixture
async def cache(env):
    """An answer cache over an (unbuilt) index with the local embedder: the cache needs only its embedder."""
    from tulpar_ai.rag.answer_cache import get_answer_cache, set_answer_cache
    from tulpar_ai.rag.index import Index
    from tulpar_ai.rag.retrieve import set_index

    idx = Index()
    set_index(idx)
    yield get_answer_cache()
    set_answer_cache(None)
    set_index(None)
    idx.close()


async def test_repeat_hits_with_the_same_reply(cache, fake_llm):
    from tulpar_ai.graph import chat

    assert await cache.lookup(Q) is None
    assert await cache.put(Q, REPLY, CITES)
    hit = await cache.lookup("  как правильно жать штангу лёжа?  Под каким углом держать локти? ")
    assert hit["reply"] == REPLY and hit["citations"] == CITES and hit["score"] >= cache.min_score
    out = await chat.cache_lookup({"text": Q, "intent": "question"})
    assert out["reply"] == REPLY and out["kind"] == "answer" and chat.after_cache(out) == chat.END
    assert fake_llm.calls == []


async def test_near_miss_does_not_hit(cache):
    await cache.put(Q, REPLY, CITES)
    vec = await cache.index.embed_query(NEAR)
    nearest = cache._nearest(cache.collection, vec, time.time())
    assert nearest is not None and nearest["score"] > 0.9  # lexically very close — and still a different exercise
    assert await cache.lookup(NEAR) is None


async def test_only_grounded_answers_are_stored(cache):
    assert not await cache.put(Q, REPLY, [])
    assert not await cache.put(Q, "", CITES)
    assert cache.count() == 0


async def test_ttl_expiry_and_purge(cache):
    from tulpar_ai.config import get_settings

    old = time.time() - (get_settings().answer_cache_ttl_h + 1) * 3600
    await cache.put(Q, REPLY, CITES, created_at=old)
    assert await cache.lookup(Q) is None
    await cache.put("Сколько секунд держать вакуум живота?", "Около 20 секунд [1].", CITES)
    assert cache.count() == 1  # the expired entry was purged on the next store


async def test_fingerprint_change_invalidates(cache, monkeypatch):
    from tulpar_ai.config import get_settings

    await cache.put(Q, REPLY, CITES)
    before = cache.collection
    assert await cache.lookup(Q) is not None

    s = get_settings()
    chain = s.text_models
    monkeypatch.setattr(s, "text_models", "groq:some-other-model")
    assert cache.collection != before
    assert await cache.lookup(Q) is None

    monkeypatch.setattr(s, "text_models", chain)
    monkeypatch.setenv("PROMPT_ANSWER", "v2")
    assert cache.collection != before
    assert await cache.lookup(Q) is None

    monkeypatch.delenv("PROMPT_ANSWER")
    assert cache.collection == before and await cache.lookup(Q) is not None


async def test_cache_can_be_switched_off(env, monkeypatch):
    from tulpar_ai.config import get_settings
    from tulpar_ai.rag.answer_cache import get_answer_cache

    monkeypatch.setattr(get_settings(), "answer_cache", False)
    assert get_answer_cache() is None


async def test_cache_store_node_skips_everything_but_answers(cache):
    from tulpar_ai.graph import chat

    base = {"text": Q, "reply": REPLY, "citations": CITES}
    await chat.cache_store({**base, "kind": "escalated"})
    await chat.cache_store({**base, "kind": "answer", "red_flag": True})
    await chat.cache_store({**base, "kind": "answer", "citations": []})
    await chat.cache_store({**base, "kind": "meal_card"})
    assert cache.count() == 0
    await chat.cache_store({**base, "kind": "answer"})
    assert cache.count() == 1


async def test_graph_second_turn_skips_answer_model(app_state, fake_llm):
    """Through the real graph: the repeat makes only the router call, and escalations are never cached."""
    from tulpar_ai.graph import runner
    from tulpar_ai.rag.answer_cache import get_answer_cache, set_answer_cache
    from tulpar_ai.rag.index import Index
    from tulpar_ai.rag.retrieve import set_index

    _, gw = app_state
    idx = Index()
    await idx.build()
    set_index(idx)
    embeds: list[list[str]] = []
    real_embed = idx.embedder.embed

    async def counting_embed(texts, task):
        embeds.append(texts)
        return await real_embed(texts, task)

    idx.embedder.embed = counting_embed
    try:
        client = await gw.demo_user("client")
        q = "Сколько минут активности в неделю рекомендует ВОЗ?"
        first = await runner.run_chat_turn(client.id, text=q)
        assert first["kind"] == "answer" and first["citations"] and first.get("cache_score") is None
        assert get_answer_cache().count() == 1
        assert embeds.count([q]) == 1  # lookup, search and store share one embedding (a rewrite needs its own)

        n = len(fake_llm.calls)
        second = await runner.run_chat_turn(client.id, text=q)
        new_calls = fake_llm.calls[n:]
        assert second["reply"] == first["reply"] and second["citations"] == first["citations"]
        assert second["cache_score"] >= get_answer_cache().min_score
        assert [c["role"] for c in new_calls] == ["route"]  # no answer, no rewrite

        for _ in range(2):  # a red flag goes to the trainer both times, never through the cache
            esc = await runner.run_chat_turn(client.id, text="Колено болит при приседе, что делать?")
            assert esc["kind"] == "escalated" and esc.get("cache_score") is None
        assert get_answer_cache().count() == 1
    finally:
        set_answer_cache(None)
        set_index(None)
        idx.close()
