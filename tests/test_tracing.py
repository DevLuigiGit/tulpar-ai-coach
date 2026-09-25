"""Tracing audit: every external call of a chat turn (LLM roles, STT, embeddings, rerank, the draft it starts)
lands in ONE LangSmith trace rooted at `coach_turn`. A recording client stands in for LangSmith: no network."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from langsmith import Client
from langsmith.run_helpers import tracing_context

from conftest import FakeLLM, boot, shutdown


class StubIndex:
    """Two canned hits, but the query embedding and the rerank go through the real Jina classes (HTTP faked)."""

    def __init__(self):
        from tulpar_ai.rag.embed import JinaEmbedder

        self.embedder = JinaEmbedder()
        self.collection = "stub"

    async def search(self, query: str, limit: int) -> list[dict]:
        await self.embedder.embed([query], task="retrieval.query")
        return [{"title": "WHO guidelines", "text": "150–300 минут умеренной активности", "source": "who.pdf", "page": 1,
                 "score": 0.5}, {"title": "Питание", "text": "белок", "source": "nutrition", "page": None, "score": 0.4}]

    def count(self) -> int:
        return 2

    def close(self) -> None:
        return None


async def fake_jina(c, path: str, body: dict, attempts: int = 5) -> dict:
    if path == "/embeddings":
        return {"data": [{"index": i, "embedding": [0.1] * 1024} for i in range(len(body["input"]))]}
    # low score until the rewrite adds the English terms → exercises the rewrite loop too
    score = 0.9 if "physical" in body["query"] else 0.05
    return {"results": [{"index": i, "relevance_score": score} for i in range(min(body["top_n"], len(body["documents"])))]}


@pytest.fixture
def traced(env, monkeypatch):
    """Real llm.chat → _traced_call path with a fake provider, and a recording LangSmith client."""
    from tulpar_ai import llm, stt
    from tulpar_ai.config import get_settings
    from tulpar_ai.rag import embed

    for k, v in {"OLLAMA_API_KEY": "test", "JINA_API_KEY": "test", "RAG_RERANK": "on",
                 "ROUTE_MODELS": "ollama:fake", "TEXT_MODELS": "ollama:fake", "VISION_MODELS": "ollama:fake",
                 "ANSWER_CACHE": "false"}.items():  # the stub index has no Qdrant storage for the cache
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    fake = FakeLLM()

    async def provider(model, system, user, *, images, json_mode, temperature, top_p, max_tokens):
        return fake(role="?", system=system, user=user, images=images, json_mode=json_mode), 10, 5

    monkeypatch.setitem(llm._PROVIDERS, "ollama", provider)
    monkeypatch.setattr(embed, "_post", fake_jina)
    stt.set_fake(lambda audio, name: "съел 200 г плова")
    client = MagicMock(spec=Client)
    with tracing_context(enabled=True, client=client, project_name="test"):
        yield client
    stt.set_fake(None)
    get_settings.cache_clear()


def runs_by_trace(client: MagicMock) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for call in client.create_run.call_args_list:
        run = call.kwargs
        out.setdefault(str(run.get("trace_id") or run["id"]), []).append(run)
    return out


async def test_each_chat_turn_is_one_trace_with_all_its_external_calls(env, traced):
    from tulpar_ai.graph import runner
    from tulpar_ai.rag.retrieve import set_index

    store, gw = await boot(env)
    set_index(StubIndex())
    try:
        client = await gw.demo_user("client")
        turns = [
            ({"text": "Сколько минут активности в неделю рекомендует ВОЗ?"},
             {"llm", "retrieve", "jina_embed", "jina_rerank", "rewrite", "answer"}),
            ({"text": "съел 200 г плова и чай"}, {"llm", "meal_text"}),
            ({"image": b"\xff\xd8fakejpeg"}, {"llm", "meal_photo"}),
            ({"audio": b"OggS-fake", "audio_name": "v.ogg"}, {"speech_to_text", "llm", "meal_text"}),
            ({"text": "Замени, пожалуйста, упражнение в программе"}, {"llm", "program_request", "program_change", "draft"}),
        ]
        for kw, expected in turns:
            before = set(runs_by_trace(traced))
            res = await runner.run_chat_turn(client.id, **kw)
            if res.get("proposal_id"):  # the draft runs in a background task: wait for it
                for _ in range(200):
                    if (await store.get_proposal(res["proposal_id"]))["status"] != "drafting":
                        break
                    await asyncio.sleep(0.05)
            new = {t: runs for t, runs in runs_by_trace(traced).items() if t not in before}
            assert len(new) == 1, f"{kw}: {len(new)} traces, expected one"
            [runs] = new.values()
            roots = [r for r in runs if not r.get("parent_run_id")]
            assert [r["name"] for r in roots] == ["coach_turn"], kw
            names = {r["name"] for r in runs}
            assert expected <= names, f"{kw}: missing {expected - names}"
    finally:
        set_index(None)
        await shutdown(store)


def test_embedding_trace_payload_is_summarized():
    from tulpar_ai.rag.embed import trace_embed_inputs, trace_embed_outputs

    texts = [f"chunk {i} " + "x" * 1000 for i in range(1579)]
    assert trace_embed_inputs({"texts": texts, "task": "retrieval.passage"}) == {
        "task": "retrieval.passage", "count": 1579, "texts": [t[:300] for t in texts[:3]]}
    assert trace_embed_outputs([[0.0] * 1024] * 1579) == {"vectors": 1579, "dim": 1024}
    assert trace_embed_outputs([]) == {"vectors": 0, "dim": 0}


async def test_photo_and_voice_bytes_stay_out_of_traces(env, traced, monkeypatch):
    """A photo is sent once per vision model tried and a voice note once: the trace keeps only their sizes."""
    import json

    from tulpar_ai import llm
    from tulpar_ai.config import get_settings
    from tulpar_ai.graph import runner

    monkeypatch.setenv("VISION_MODELS", "ollama:down,ollama:fake")
    get_settings.cache_clear()
    ok = llm._PROVIDERS["ollama"]

    async def first_fails(model, *a, **kw):
        if model == "down":
            raise llm.LLMError("ollama 503: down")
        return await ok(model, *a, **kw)

    monkeypatch.setitem(llm._PROVIDERS, "ollama", first_fails)
    store, gw = await boot(env)
    try:
        client = await gw.demo_user("client")
        photo, voice = b"\xff\xd8" + b"p" * 600_000, b"OggS" + b"v" * 400_000
        await runner.run_chat_turn(client.id, image=photo)
        await runner.run_chat_turn(client.id, audio=voice, audio_name="v.ogg")
    finally:
        await shutdown(store)
    runs = [c.kwargs for c in traced.create_run.call_args_list]
    vision = [r for r in runs if r["name"] == "llm" and r["inputs"].get("images")]
    assert len(vision) == 2  # the failed model and the fallback both saw the photo
    for r in vision:
        assert r["inputs"]["images"] == [{"type": "image/jpeg", "bytes": len(photo)}]
    [stt_run] = [r for r in runs if r["name"] == "speech_to_text"]
    assert stt_run["inputs"] == {"audio": {"bytes": len(voice), "format": "ogg"}, "language": "ru"}
    biggest = max(len(json.dumps(r.get("inputs"), default=str)) for r in runs)
    assert biggest < 50_000, biggest


def test_trace_input_summaries_without_media():
    from tulpar_ai.llm import trace_llm_inputs
    from tulpar_ai.stt import trace_stt_inputs

    plain = {"provider": "ollama", "model": "m", "user": "текст", "images": None}
    assert trace_llm_inputs(plain) is plain
    assert trace_stt_inputs({"audio": b"", "filename": "voice", "language": "ru"}) == {
        "audio": {"bytes": 0, "format": "ogg"}, "language": "ru"}
