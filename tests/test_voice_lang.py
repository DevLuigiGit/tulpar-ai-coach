"""Kazakh voice: which language Whisper is asked for, and how the Groq request looks — fakes only, no network."""

from __future__ import annotations

import pytest

from tulpar_ai import lang, stt

KK = "Сәлеметсіз бе! Бүгін жаттығуға келе алмаймын, бағдарламаны жеңілдетіп беріңізші."
MIXED = "Бүгін обедке плов жедім, примерно 300 грамм, и чай с молоком іштім."
RU = "Сколько белка нужно в день, если я набираю массу?"
# A Russian meal card that quotes two Kazakh food names: 4 Kazakh letters, but a Russian text.
RU_CARD = ("Нашёл 1 поз., всего ≈ 480 ккал:\n• плов: 200 г ≈ 480 ккал\n"
           "• «сұлы ботқасы» — нет в справочнике, можно добавить вручную\n"
           "• «құрт» — нет в справочнике, можно добавить вручную\n\nПроверьте граммы и нажмите «Записать».")


# ── is it Kazakh ─────────────────────────────────────────────────────────────
def test_word_share_separates_kazakh_mixed_and_russian(env):
    assert lang.kk_word_share(KK) >= 0.5
    assert 0.15 <= lang.kk_word_share(MIXED) < 0.5
    assert lang.kk_word_share(RU) == 0.0
    assert lang.kk_word_share("") == 0.0 and lang.kk_word_share("300 ккал 👍") == 0.0
    assert lang.is_kazakh(KK) and lang.is_kazakh(MIXED) and not lang.is_kazakh(RU)


def test_quoted_kazakh_words_do_not_make_a_russian_text_kazakh(env):
    assert sum(ch in lang.KK_LETTERS for ch in RU_CARD) >= 3  # the old letter count would call it Kazakh
    assert not lang.is_kazakh(RU_CARD)


# ── which language Whisper gets ─────────────────────────────────────────────
def test_fixed_modes(env):
    assert stt.choose_language("ru", recent_texts=[KK]) == "ru"
    assert stt.choose_language("kk", recent_texts=[RU]) == "kk"
    assert stt.choose_language("auto") is None
    with pytest.raises(ValueError):
        stt.choose_language("de")


def test_hint_mode_follows_what_the_client_wrote(env):
    assert stt.choose_language("hint", recent_texts=[KK]) == "kk"
    assert stt.choose_language("hint", recent_texts=[MIXED, RU]) == "kk"  # 3 of 18 words: still a Kazakh speaker
    assert stt.choose_language("hint", recent_texts=[RU, "Привет!"]) == "ru"
    assert stt.choose_language("hint") == "ru"  # nothing known yet: the old default
    assert stt.choose_language("hint", recent_texts=[RU], tg_language="kk") == "kk"
    assert stt.choose_language("hint", recent_texts=[RU], tg_language="ru") == "ru"


def test_mode_comes_from_settings(env, monkeypatch):
    from tulpar_ai.config import get_settings

    monkeypatch.setenv("STT_LANGUAGE", "auto")
    get_settings.cache_clear()
    assert stt.choose_language() is None
    monkeypatch.setenv("STT_LANGUAGE", "KK")
    get_settings.cache_clear()
    assert stt.choose_language() == "kk"


def test_trace_shows_auto_for_no_language():
    assert stt.trace_stt_inputs({"audio": b"abc", "filename": "v.ogg", "language": None}) == {
        "audio": {"bytes": 3, "format": "ogg"}, "language": "auto"}


# ── the Groq request ─────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, status: int, body: dict):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


@pytest.fixture
def groq(env, monkeypatch):
    from tulpar_ai.config import get_settings

    monkeypatch.setenv("GROQ_API_KEY", "test")
    get_settings.cache_clear()
    sent: list[dict] = []
    replies: list[_Resp] = []

    class Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, data=None, files=None):
            sent.append({"url": url, "data": dict(data), "filename": files["file"][0]})
            return replies.pop(0)

    monkeypatch.setattr(stt.httpx, "AsyncClient", Client)
    return sent, replies


async def test_fixed_language_is_sent_and_echoed(groq):
    sent, replies = groq
    replies.append(_Resp(200, {"text": " Сәлем! "}))
    res = await stt.transcribe_detailed(b"OggS", "voice", language="kk")
    assert res == {"text": "Сәлем!", "language": "kk"}
    assert sent[0]["data"]["language"] == "kk" and sent[0]["data"]["response_format"] == "json"
    assert sent[0]["filename"] == "voice.ogg"


async def test_auto_sends_no_language_and_reads_the_detected_one(groq):
    sent, replies = groq
    replies += [_Resp(200, {"text": "Сәлем", "language": "Kazakh"}), _Resp(200, {"text": "Привет", "language": "russian"}),
                _Resp(200, {"text": "", "language": None})]
    assert (await stt.transcribe_detailed(b"OggS", "v.ogg", language=None))["language"] == "kk"
    assert (await stt.transcribe_detailed(b"OggS", "v.ogg", language=None))["language"] == "ru"
    assert (await stt.transcribe_detailed(b"OggS", "v.ogg", language=None)) == {"text": "", "language": None}
    assert all("language" not in s["data"] and s["data"]["response_format"] == "verbose_json" for s in sent)


async def test_recognize_asks_again_in_russian_when_auto_hears_another_language(groq):
    sent, replies = groq
    replies += [_Resp(200, {"text": "Сәлем", "language": "kazakh"}),
                _Resp(200, {"text": "Selam", "language": "turkish"}), _Resp(200, {"text": "Салем"})]
    assert await stt.recognize(b"OggS", "v.ogg", language=None) == {"text": "Сәлем", "language": "kk"}
    assert await stt.recognize(b"OggS", "v.ogg", language=None) == {"text": "Салем", "language": "ru", "detected": "tr"}
    assert ["language" in s["data"] for s in sent] == [False, False, True]
    replies.append(_Resp(200, {"text": "Сәлем"}))
    assert (await stt.recognize(b"OggS", "v.ogg", language="kk"))["language"] == "kk"  # fixed: one call
    assert len(sent) == 4


async def test_transcribe_keeps_its_old_contract(groq):
    sent, replies = groq
    replies.append(_Resp(200, {"text": "съел плов"}))
    assert await stt.transcribe(b"OggS", "v.ogg") == "съел плов"
    assert sent[0]["data"]["language"] == "ru"
    replies.append(_Resp(429, {"error": "rate limit"}))
    with pytest.raises(RuntimeError, match="stt 429"):
        await stt.transcribe(b"OggS", "v.ogg")


# ── the chat graph asks for the right language ──────────────────────────────
@pytest.fixture
def heard(monkeypatch):
    calls: list[str | None] = []

    async def fake(audio, filename="voice.ogg", language="ru"):
        calls.append(language)
        return {"text": "съел 200 г плова", "language": language or "kk"}

    monkeypatch.setattr(stt, "transcribe_detailed", fake)
    return calls


async def _voice(user, **kw):
    from tulpar_ai import service

    return await service.chat_turn(user, audio=b"OggS-fake", audio_name="v.ogg", **kw)


@pytest.mark.parametrize("mode, expected", [("ru", "ru"), ("kk", "kk"), ("auto", None)])
async def test_fixed_modes_reach_whisper(app_state, heard, monkeypatch, mode, expected):
    from tulpar_ai.config import get_settings

    monkeypatch.setenv("STT_LANGUAGE", mode)
    get_settings.cache_clear()
    _, gw = app_state
    client = await gw.demo_user("client")
    res = await _voice(client)
    assert heard == [expected]
    assert res["transcript"] == "съел 200 г плова" and res["kind"] == "meal_card"
    assert res["stt_language"] == (expected or "kk")


async def test_hint_mode_uses_the_clients_history(app_state, heard, monkeypatch):
    from tulpar_ai.config import get_settings

    monkeypatch.setenv("STT_LANGUAGE", "hint")
    get_settings.cache_clear()
    store, gw = app_state
    client = await gw.demo_user("client")
    await _voice(client)  # nothing known: Russian, as before
    await store.add_message(client.id, "user", KK)
    await _voice(client)  # the client writes Kazakh: Kazakh
    assert heard == ["ru", "kk"]


async def test_hint_mode_uses_earlier_transcripts_and_telegram_language(app_state, heard, monkeypatch):
    from tulpar_ai.config import get_settings

    monkeypatch.setenv("STT_LANGUAGE", "hint")
    get_settings.cache_clear()
    store, gw = app_state
    client = await gw.demo_user("client")
    await _voice(client, lang_hint="kk")  # Telegram in Kazakh
    await store.add_message(client.id, "assistant", "ответ", {"kind": "answer", "transcript": KK})
    await _voice(client)  # an earlier voice note came out Kazakh
    other = await gw.telegram_user("777", "Клиент", as_trainer=False)
    await _voice(other, lang_hint="ru")
    assert heard == ["kk", "kk", "ru"]
