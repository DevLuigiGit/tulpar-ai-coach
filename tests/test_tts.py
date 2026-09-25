"""Voice replies: text cleanup for speech, /api/tts, and the bot answering a voice note with voice — fake TTS, no network."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramBadRequest
from fastapi.testclient import TestClient

from tulpar_ai import tts
from tulpar_ai.gateway.base import User

MP3 = b"\xff\xf3d\xc4fake-mp3"
ANSWER = ("По рекомендациям ВОЗ взрослым нужно **150–300 минут** умеренной активности в неделю [1]. "
          "Добавьте силовые 2 раза [2, 3].\n\nЭто общая информация, а не медицинская консультация.")
CLIENT = User(id="c1", role="client", name="Клиент")
TRAINER = User(id="t1", role="trainer", name="Тренер")


@pytest.fixture
def fake_tts():
    spoken: list[tuple[str, str]] = []

    def fake(text: str, voice: str) -> bytes:
        spoken.append((text, voice))
        return MP3

    tts.set_fake(fake)
    yield spoken
    tts.set_fake(None)


# ── text for speech ──────────────────────────────────────────────────────────
def test_speech_text_drops_citations_markdown_and_disclaimer(env):
    out = tts.speech_text(ANSWER)
    assert out == ("По рекомендациям ВОЗ взрослым нужно 150–300 минут умеренной активности в неделю. "
                   "Добавьте силовые 2 раза.")


def test_speech_text_reads_meal_card_units(env):
    card = "Нашёл 2 поз., всего ≈ 520 ккал:\n• плов: 200 г ≈ 480 ккал\n• чай: 200 г ≈ 40 ккал\n\nПроверьте граммы."
    out = tts.speech_text(card)
    assert "•" not in out and "≈" not in out and "ккал" not in out
    assert "плов: 200 грамм примерно 480 килокалорий." in out
    assert out.startswith("Нашёл 2 позиций, всего примерно 520 килокалорий:")


def test_speech_text_caps_on_sentence_boundary(env):
    long = " ".join(f"Предложение номер {i} про технику приседа." for i in range(40))
    out = tts.speech_text(long, max_chars=200)
    assert out.endswith(tts.TRUNCATED_TAIL)
    body = out[: -len(tts.TRUNCATED_TAIL)].strip()
    assert len(body) <= 200 and body.endswith(".")


def test_speech_text_caps_without_sentence_end(env):
    out = tts.speech_text("слово " * 200, max_chars=100)
    body = out[: -len(tts.TRUNCATED_TAIL)].strip()
    assert len(body) <= 100 and body.endswith("слово.")


def test_kazakh_text_gets_kazakh_voice(env):
    from tulpar_ai.config import get_settings

    s = get_settings()
    assert tts.pick_voice("Сәлеметсіз бе! Бүгін жаттығу күні.") == s.tts_voice_kk
    assert tts.pick_voice("Добрый день! Сегодня тренировка.") == s.tts_voice


# ── synthesize ───────────────────────────────────────────────────────────────
async def test_synthesize_speaks_clean_masked_text(env, fake_tts):
    audio = await tts.synthesize(ANSWER + " Звоните +7 701 123 45 67.")
    assert audio == MP3
    text, voice = fake_tts[0]
    assert "[1]" not in text and "медицинская консультация" not in text and "701" not in text
    assert voice == "ru-RU-SvetlanaNeural"


async def test_synthesize_rejects_empty(env, fake_tts):
    with pytest.raises(ValueError):
        await tts.synthesize("[1]\n\nЭто общая информация, а не медицинская консультация.")
    assert fake_tts == []


# ── HTTP ─────────────────────────────────────────────────────────────────────
@pytest.fixture
def api(env, fake_tts):
    from tulpar_ai.api.app import app
    from tulpar_ai.api.auth import current_user

    def as_user(user: User):
        app.dependency_overrides[current_user] = lambda: user

    # No lifespan: /api/tts needs neither the store nor the RAG index.
    yield TestClient(app), as_user
    app.dependency_overrides.clear()


def test_api_tts_returns_mp3(api, fake_tts):
    c, as_user = api
    as_user(CLIENT)
    r = c.post("/api/tts", json={"text": ANSWER})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg"
    assert r.content == MP3 and len(fake_tts) == 1


def test_api_tts_is_for_clients_only(api, fake_tts):
    c, as_user = api
    as_user(TRAINER)
    assert c.post("/api/tts", json={"text": ANSWER}).status_code == 403
    assert fake_tts == []


def test_api_tts_errors(api, monkeypatch):
    c, as_user = api
    as_user(CLIENT)
    assert c.post("/api/tts", json={"text": ""}).status_code == 422
    assert c.post("/api/tts", json={"text": "[1]"}).status_code == 422

    def boom(text, voice):
        raise RuntimeError("edge down")

    tts.set_fake(boom)
    r = c.post("/api/tts", json={"text": ANSWER})
    assert r.status_code == 502 and "озвучить" in r.json()["detail"]
    monkeypatch.setenv("TTS_ENABLED", "false")
    from tulpar_ai.config import get_settings

    get_settings.cache_clear()
    assert c.post("/api/tts", json={"text": ANSWER}).status_code == 404


# ── Telegram bot ─────────────────────────────────────────────────────────────
def _message() -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(id=42), bot=SimpleNamespace(send_chat_action=AsyncMock()),
                           answer=AsyncMock(), answer_voice=AsyncMock(), answer_audio=AsyncMock())


@pytest.fixture
def bot_turn(env, monkeypatch):
    from tulpar_ai.bot import bot

    reply = {"kind": "answer", "reply": ANSWER, "transcript": "сколько ходить", "citations": []}
    monkeypatch.setattr(bot.service, "chat_turn", AsyncMock(return_value=reply))
    return bot


async def test_bot_answers_voice_with_text_and_voice(bot_turn, fake_tts):
    m = _message()
    await bot_turn._run_turn(m, CLIENT, voice_reply=True, audio=b"ogg", audio_name="voice.ogg")
    assert "Распознано" in m.answer.await_args.args[0]
    sent = m.answer_voice.await_args.args[0]
    assert sent.data == MP3 and sent.filename == "reply.mp3"
    assert "медицинская консультация" not in fake_tts[0][0]
    m.answer_audio.assert_not_awaited()


async def test_bot_text_message_gets_no_voice(bot_turn, fake_tts):
    m = _message()
    await bot_turn._run_turn(m, CLIENT, text="сколько ходить")
    m.answer.assert_awaited_once()
    m.answer_voice.assert_not_awaited()
    assert fake_tts == []


async def test_bot_tts_failure_keeps_text_reply(bot_turn):
    def boom(text, voice):
        raise RuntimeError("edge down")

    tts.set_fake(boom)
    try:
        m = _message()
        await bot_turn._run_turn(m, CLIENT, voice_reply=True, audio=b"ogg", audio_name="voice.ogg")
    finally:
        tts.set_fake(None)
    m.answer.assert_awaited_once()
    m.answer_voice.assert_not_awaited()


async def test_bot_falls_back_to_audio_when_voice_is_forbidden(bot_turn, fake_tts):
    m = _message()
    m.answer_voice.side_effect = TelegramBadRequest(method=None, message="Bad Request: VOICE_MESSAGES_FORBIDDEN")
    await bot_turn._run_turn(m, CLIENT, voice_reply=True, audio=b"ogg", audio_name="voice.ogg")
    m.answer.assert_awaited_once()
    assert m.answer_audio.await_args.args[0].data == MP3
