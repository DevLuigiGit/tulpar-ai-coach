"""A decision made in one place (web, bot, MCP) closes the trainer's Telegram cards everywhere.

The screenshot bug: the draft was accepted in the web cabinet, the Telegram card kept its buttons, and pressing
«Принять» there answered with a raw «proposal is applied, not pending».
"""

from __future__ import annotations

from types import SimpleNamespace

from conftest import boot, shutdown
from test_program_resume import wait_status


class FakeBot:
    def __init__(self):
        self.edits: list[dict] = []

    async def edit_message_text(self, text, chat_id, message_id, reply_markup=None):
        self.edits.append({"text": text, "chat_id": chat_id, "message_id": message_id, "markup": reply_markup})


class FakeMessage:
    def __init__(self, chat_id: int, message_id: int, text: str):
        self.chat = SimpleNamespace(id=chat_id)
        self.message_id, self.text = message_id, text
        self.edited: list[tuple] = []
        self.answers: list[str] = []

    async def edit_text(self, text, reply_markup=None):
        self.edited.append((text, reply_markup))

    async def edit_reply_markup(self, reply_markup=None):
        self.edited.append((None, reply_markup))

    async def answer(self, text, **_):
        self.answers.append(text)


class FakeCallback:
    def __init__(self, data: str, message: FakeMessage):
        self.data, self.message = data, message
        self.toasts: list[str] = []

    async def answer(self, text: str | None = None, **_):
        self.toasts.append(text or "")


async def test_web_decision_closes_the_telegram_card(env, fake_llm, monkeypatch):
    from tulpar_ai import notify
    from tulpar_ai.bot import bot
    from tulpar_ai.graph import runner

    store, gw = await boot(env)
    fake = FakeBot()
    monkeypatch.setattr(bot, "_bot", fake)
    notify.clear_sinks()
    notify.add_sink(bot.BotSink())
    try:
        client = await gw.demo_user("client")
        trainer = await gw.demo_user("trainer")
        p = await runner.new_program_proposal(client.id, trainer.id, "Замени первое упражнение на щадящее", "trainer")
        p = await wait_status(store, p["id"], "pending")
        await store.add_tg_message(p["id"], 555, 42, "Черновик для Айдара")

        await runner.resume_program(p["id"], "accept")  # what the web cabinet does

        assert fake.edits and fake.edits[0]["message_id"] == 42 and fake.edits[0]["markup"] is None
        assert "✅ Применено" in fake.edits[0]["text"]
        assert await store.pop_tg_messages(p["id"]) == []  # a card is closed once
    finally:
        notify.clear_sinks()
        await shutdown(store)


async def test_pressing_a_stale_card_explains_in_russian(env, fake_llm, monkeypatch):
    from tulpar_ai.bot import bot
    from tulpar_ai.graph import runner

    store, gw = await boot(env)
    try:
        client = await gw.demo_user("client")
        trainer = await gw.demo_user("trainer")
        p = await runner.new_program_proposal(client.id, trainer.id, "Замени первое упражнение на щадящее", "trainer")
        p = await wait_status(store, p["id"], "pending")
        await runner.resume_program(p["id"], "accept")

        async def as_trainer(_cb):
            return trainer

        monkeypatch.setattr(bot, "_user", as_trainer)
        msg = FakeMessage(555, 43, "Черновик для Айдара")
        cb = FakeCallback(f"acc:{p['id']}", msg)
        await bot.on_decision(cb)

        text, markup = msg.edited[-1]
        assert markup is None and "Применено" in text and "другом окне" in text
        assert not any("pending" in a or "applied" in a for a in msg.answers), msg.answers  # no raw English error
        assert (await store.get_proposal(p["id"]))["status"] == "applied"  # nothing applied twice
    finally:
        await shutdown(store)


async def test_second_resolve_is_refused_and_the_client_gets_one_reply(env, fake_llm):
    import pytest

    from tulpar_ai import service

    store, gw = await boot(env)
    try:
        client = await gw.demo_user("client")
        trainer = await gw.demo_user("trainer")
        e = await store.create_proposal(kind="escalation", client_id=client.id, trainer_id=trainer.id,
                                        source="client", request="Колено болит при приседе", status="open")
        await service.resolve_escalation(trainer, e["id"], "Уберите присед до встречи")
        with pytest.raises(service.AlreadyDecided):
            await service.resolve_escalation(trainer, e["id"], "Test")
        replies = [m for m in await store.history(client.id, limit=50)
                   if (m.get("payload") or {}).get("kind") == "trainer_reply"]
        assert len(replies) == 1 and "Уберите присед" in replies[0]["text"]
    finally:
        await shutdown(store)
