"""Streamed replies below HTTP: the partial-JSON reader, the guarded sentence release, provider fallback and
tracing. The endpoint itself is tested in test_stream_api.py. No network: fake LLMs only."""

from __future__ import annotations

import json
import random
from unittest.mock import MagicMock

import pytest

from conftest import FakeLLM, boot, shutdown
from test_tracing import StubIndex, runs_by_trace, traced  # noqa: F401  (traced is a fixture)
from tulpar_ai import guardrails, llm, streaming


# ── JsonStringField ──────────────────────────────────────────────────────────
def _read(raw: str, pieces: list[str] | None = None) -> str:
    f = streaming.JsonStringField("answer")
    return "".join(f.feed(p) for p in (pieces if pieces is not None else list(raw)))


TRICKY = [
    'Жим лёжа: локти под углом 45° [1].',
    'Кавычки "внутри", обратный слеш \\ и слеш /.',
    "Строки\nс переводом\tи табом.",
    "Эмодзи 💪 и 🏋️‍♀️, казахские буквы әіңғүұқөһ.",
    '{"answer": "не настоящий ключ"}',
    "",
]


@pytest.mark.parametrize("text", TRICKY)
@pytest.mark.parametrize("ascii_only", [True, False])
def test_json_field_decodes_like_json_loads(text, ascii_only):
    raw = json.dumps({"answer": text, "citations": [1], "sufficient": True}, ensure_ascii=ascii_only)
    assert _read(raw) == json.loads(raw)["answer"]  # one character at a time: every escape is split
    rnd = random.Random(len(text) * 7 + ascii_only)
    for _ in range(20):
        cuts = sorted(rnd.sample(range(1, len(raw)), k=min(len(raw) - 1, rnd.randint(1, 12))))
        pieces = [raw[a:b] for a, b in zip([0, *cuts], [*cuts, len(raw)])]
        assert _read(raw, pieces) == text


def test_json_field_follows_structure_not_text():
    raw = ('Вот ответ:\n```json\n{"note": "ключ \\"answer\\": тут не считается", "meta": {"answer": "вложенный"},'
           ' "list": ["answer"], "answer": "настоящий ответ", "sufficient": true}\n```')
    assert _read(raw) == "настоящий ответ"
    assert _read('{"answer": null, "sufficient": false}') == ""
    assert _read('{"sufficient": false}') == ""
    f = streaming.JsonStringField("answer")
    assert f.feed('{"answer": "первое') == "первое"
    assert f.feed(' продолжение", "answer": "второе"}') == " продолжение"  # a repeated key is ignored


def test_json_field_lone_surrogate_does_not_break_utf8():
    out = _read('{"answer": "a\\ud83d b"}')
    assert out == "a� b"
    out.encode("utf-8")


# ── GuardedText ──────────────────────────────────────────────────────────────
def _guarded(pieces: list[str], final: str | None = None) -> tuple[list[str], streaming.GuardedText]:
    sent: list[str] = []
    g = streaming.GuardedText(sent.append)
    for p in pieces:
        g.push(p)
    if final is not None:
        g.finish(final)
    return sent, g


def test_guarded_text_releases_whole_sentences_then_the_tail():
    text = "Первое предложение [1]. Второе предложение! Хвост без точки"
    sent, g = _guarded(list(text))
    assert sent == ["Первое предложение [1]. ", "Второе предложение! "]
    final = text + "\n\nЭто общая информация."
    g.finish(final)
    assert "".join(sent) == final


def test_guarded_text_close_sends_the_last_sentence_before_the_json_ends():
    answer, disclaimer = "Одно предложение [1].", "\n\nЭто общая информация."
    f = streaming.JsonStringField("answer")
    sent: list[str] = []
    g = streaming.GuardedText(sent.append)
    for ch in json.dumps({"answer": answer, "citations": [1]}, ensure_ascii=False)[:-12]:  # stop inside "citations"
        g.push(f.feed(ch))
        if f.closed:
            g.close(g.raw.strip() + disclaimer)
    assert f.closed and sent == [answer]  # no trailing space needed once the string has ended
    g.finish(answer + disclaimer)
    assert "".join(sent) == answer + disclaimer


def test_guarded_text_close_judges_the_whole_final_reply():
    sent, g = _guarded(list("Отдых важен. Принимайте ибупрофен по 400 мг"))
    g.close("Отдых важен. Принимайте ибупрофен по 400 мг\n\nДисклеймер.")
    g.finish("Отдых важен. Принимайте ибупрофен по 400 мг\n\nДисклеймер.")
    assert "".join(sent) == "Отдых важен. " and g.blocked == "blocked_dosage"


def test_guarded_text_masks_contacts_per_sentence():
    sent, g = _guarded(list("Пишите на coach@tulpar.kz или звоните 8 777 123 45 67. Дальше. "))
    assert "".join(sent) == "Пишите на [email] или звоните [телефон]. Дальше. "
    assert not any("777" in s or "@" in s for s in sent)


def test_guarded_text_stops_at_a_dosage_sentence():
    text = ("Отдых важен [1]. Принимайте ибупрофен по 400 мг три раза в день. "
            "После этого можно тренироваться. Последнее.")
    sent, g = _guarded(list(text), final=text + "\n\nДисклеймер.")
    assert "".join(sent) == "Отдых важен [1]. "
    assert g.blocked == "blocked_dosage"
    assert guardrails.guard_reply(text)[1] == "blocked_dosage"  # the service guard agrees


def test_guarded_text_stops_on_a_prompt_leak_spread_over_sentences():
    from tulpar_ai.prompts import prompt

    words = prompt("answer").split()
    leak_a, leak_b = " ".join(words[:10]), " ".join(words[20:30])
    text = f"Вот что мне сказали: {leak_a}. И ещё: {leak_b}. Конец."
    sent, g = _guarded(list(text), final=text)
    assert g.blocked == "blocked_prompt_leak"
    assert leak_b not in "".join(sent)  # the second run is where the prefix becomes a leak


def test_guarded_text_stops_when_masking_rewrites_sent_text():
    # a newline is both a sentence end and a separator the phone pattern allows
    sent, g = _guarded(["Номер 8\n ", "777 123 45 67. "])
    g.finish("Номер 8\n 777 123 45 67.")
    assert "".join(sent) == "Номер 8\n "  # stopped before the rest of the number
    assert g.stopped and guardrails.guard_reply("Номер 8\n 777 123 45 67.")[0] == "Номер [телефон]."


# ── chat_stream: fallback and failure semantics ──────────────────────────────
@pytest.fixture
def two_providers(env, monkeypatch):
    from tulpar_ai.config import get_settings

    monkeypatch.setenv("OLLAMA_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    get_settings.cache_clear()
    calls: list[str] = []
    behaviour: dict[str, str] = {}

    def fake(name):
        async def stream(model, system, user, *, images, json_mode, temperature, top_p, max_tokens, on_delta):
            calls.append(name)
            mode = behaviour.get(name, "ok")
            if mode == "fail_before":
                raise llm.LLMError(f"{name} 503")
            text = json.dumps({"answer": f"ответ от {name}. Второе.", "sufficient": True}, ensure_ascii=False)
            half = len(text) // 2
            await on_delta(text[:half])
            if mode == "fail_after":
                raise llm.LLMError(f"{name} connection reset")
            await on_delta(text[half:])
            return text, 10, 5
        return stream

    monkeypatch.setitem(llm._STREAMERS, "ollama", fake("ollama"))
    monkeypatch.setitem(llm._STREAMERS, "groq", fake("groq"))
    yield calls, behaviour
    get_settings.cache_clear()


CHAIN = [("ollama", "a"), ("groq", "b")]


async def test_stream_falls_back_before_the_first_piece(two_providers):
    calls, behaviour = two_providers
    behaviour["ollama"] = "fail_before"
    pieces: list[str] = []
    res = await llm.chat_stream("text", "s", "u", on_delta=pieces.append, json_mode=True, models=CHAIN)
    assert calls == ["ollama", "groq"]
    assert res.provider == "groq" and res.fallback_used and res.data["answer"].startswith("ответ от groq")
    assert "".join(pieces) == res.text and res.first_delta_ms is not None


async def test_stream_does_not_fall_back_after_the_first_piece(two_providers):
    calls, behaviour = two_providers
    behaviour["ollama"] = "fail_after"
    pieces: list[str] = []
    with pytest.raises(llm.StreamBroken):
        await llm.chat_stream("text", "s", "u", on_delta=pieces.append, json_mode=True, models=CHAIN)
    assert calls == ["ollama"] and pieces  # the half already shown is not mixed with another model's text


async def test_stream_invalid_json_after_pieces_is_final(two_providers, monkeypatch):
    calls, _ = two_providers

    async def broken(model, system, user, *, on_delta, **kw):
        calls.append("broken")
        await on_delta('{"answer": "обрыв')
        return '{"answer": "обрыв', 1, 1

    monkeypatch.setitem(llm._STREAMERS, "ollama", broken)
    with pytest.raises(llm.StreamBroken):
        await llm.chat_stream("text", "s", "u", on_delta=lambda p: None, json_mode=True, models=CHAIN)
    assert calls == ["broken"]


async def test_stream_call_is_one_llm_run_in_the_trace(two_providers):
    from langsmith import Client
    from langsmith.run_helpers import tracing_context

    client = MagicMock(spec=Client)
    with tracing_context(enabled=True, client=client, project_name="test"):
        await llm.chat_stream("text", "s", "u", on_delta=lambda p: None, json_mode=True, models=CHAIN)
    runs = [c.kwargs for c in client.create_run.call_args_list]
    assert [r["name"] for r in runs] == ["llm"]
    assert "on_delta" not in runs[0]["inputs"]
    [end] = [c.kwargs for c in client.update_run.call_args_list]
    meta = end["extra"]["metadata"]
    assert meta["streamed"] is True and meta["ls_provider"] == "ollama" and meta["first_delta_ms"] is not None
    assert end["outputs"]["text"].startswith('{"answer"') and end["outputs"]["usage_metadata"]["output_tokens"] == 5
    assert [e["name"] for e in end["events"]] == ["new_token"]  # LangSmith's time to first token


async def test_streamed_turn_is_still_one_trace(env, traced, monkeypatch):
    """The stream endpoint runs chat_turn in its own task: the turn must stay one trace with the same runs."""
    from tulpar_ai import service
    from tulpar_ai.rag.retrieve import set_index

    fake = FakeLLM()

    async def streamer(model, system, user, *, images, json_mode, temperature, top_p, max_tokens, on_delta):
        text = fake(role="?", system=system, user=user, images=images, json_mode=json_mode)
        for i in range(0, len(text), 9):
            await on_delta(text[i:i + 9])
        return text, 10, 5

    monkeypatch.setitem(llm._STREAMERS, "ollama", streamer)
    store, gw = await boot(env)
    set_index(StubIndex())
    try:
        user = await gw.demo_user("client")
        before = set(runs_by_trace(traced))
        events = [e async for e in service.chat_turn_events(user, "Сколько минут активности в неделю рекомендует ВОЗ?")]
        assert events[-1]["type"] == "done" and events[-1]["kind"] == "answer"
        assert any(e["type"] == "delta" for e in events)
        new = {t: runs for t, runs in runs_by_trace(traced).items() if t not in before}
        assert len(new) == 1
        [runs] = new.values()
        assert [r["name"] for r in runs if not r.get("parent_run_id")] == ["chat_turn"]
        assert {"coach_turn", "route", "retrieve", "answer", "llm"} <= {r["name"] for r in runs}
        streamed = [c.kwargs for c in traced.update_run.call_args_list
                    if (c.kwargs.get("extra") or {}).get("metadata", {}).get("streamed")]
        assert len(streamed) == 1  # only the answer model streams; route and rewrite stay plain calls
    finally:
        set_index(None)
        await shutdown(store)
