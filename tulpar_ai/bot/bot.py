"""Telegram bot of THIS project (its own token, not Tulpar's bot). Runs inside the service process.

Clients: text, food photos and voice notes go to the chat graph; meal cards get a «Записать» button.
Trainers (Telegram ids in TRAINER_TELEGRAM_IDS): receive drafts and escalations with inline buttons —
the same human-in-the-loop decision as the web queue, one tap from the phone.

Only ONE process may poll a bot token: run the bot either locally or on Railway, not both.
"""

from __future__ import annotations

import asyncio
import io
import logging
from datetime import datetime

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from .. import notify, service
from ..config import get_settings
from ..gateway import get_gateway
from ..gateway.base import User
from ..store import get_store
from .miniapp import open_app_kb, setup_menu_button

log = logging.getLogger("bot")
dp = Dispatcher()
_bot: Bot | None = None
_awaiting: dict[int, tuple[str, str]] = {}  # trainer chat id → ("edit" | "reply", proposal id)


def _is_trainer(tg_id: int) -> bool:
    return str(tg_id) in get_settings().trainer_tg_ids


async def _user(message_or_cb) -> User:
    u = message_or_cb.from_user
    user = await get_gateway().telegram_user(str(u.id), u.first_name or "Клиент", as_trainer=_is_trainer(u.id))
    chat_id = message_or_cb.message.chat.id if isinstance(message_or_cb, CallbackQuery) else message_or_cb.chat.id
    await get_store().upsert_tg_chat(str(u.id), chat_id, user.id, user.role)
    return user


def _meal_now() -> str:
    h = datetime.now().hour
    return "breakfast" if h < 11 else "lunch" if h < 16 else "dinner" if h < 21 else "snack"


def _kb(*rows: list[tuple[str, str]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=t, callback_data=d) for t, d in row] for row in rows])


def _reply_text(r: dict) -> str:
    text = r["reply"]
    if r.get("transcript"):
        text = f"Распознано: «{r['transcript']}»\n\n{text}"
    if r.get("citations"):
        text += "\n\nИсточники: " + "; ".join(
            f"[{c['n']}] {c['title']}" + (f", стр. {c['page']}" if c.get("page") else "") for c in r["citations"])
    return text


# ── clients and trainers ─────────────────────────────────────────────────────
@dp.message(CommandStart())
async def start(m: Message):
    user = await _user(m)
    if user.role == "trainer":
        await m.answer("Вы вошли как тренер. Сюда будут приходить черновики изменений программ и сообщения клиентов, "
                       "которые требуют вашего решения. Очередь: /queue", reply_markup=open_app_kb(m.chat.type))
    else:
        await m.answer("Привет! Я AI-коуч вашего клуба.\n• Пришлите фото еды или напишите «гречка 200 г» — посчитаю "
                       "калории и запишу в дневник.\n• Спросите про технику, питание или нормы активности — отвечу со "
                       "ссылками на источники.\n• Попросите изменить программу — подготовлю черновик для тренера.\n"
                       "Голосовые тоже понимаю.", reply_markup=open_app_kb(m.chat.type))


@dp.message(Command("help"))
async def help_cmd(m: Message):
    await start(m)


@dp.message(Command("queue"))
async def queue_cmd(m: Message):
    user = await _user(m)
    if user.role != "trainer":
        return await m.answer("Эта команда для тренера.")
    items = await service.queue(user)
    if not items:
        return await m.answer("Очередь пуста.")
    for p in items[:10]:
        await (_send_proposal(m.chat.id, p) if p["kind"] == "program" else _send_escalation(m.chat.id, p))


async def _run_turn(m: Message, user: User, **kw) -> None:
    await m.bot.send_chat_action(m.chat.id, "typing")
    try:
        r = await service.chat_turn(user, **kw)
    except Exception:
        log.exception("chat turn failed")
        return await m.answer("Не получилось обработать сообщение. Попробуйте ещё раз чуть позже.")
    kb = None
    if r.get("kind") == "meal_card" and r["meal"]["items"]:
        kb = _kb([("Записать в дневник", f"meal:{r['meal']['card_id']}")])
    await m.answer(_reply_text(r), reply_markup=kb)


@dp.message(F.photo)
async def on_photo(m: Message):
    user = await _user(m)
    if user.role == "trainer":
        return await m.answer("Фото еды присылают клиенты. Очередь тренера: /queue")
    buf = io.BytesIO()
    await m.bot.download(m.photo[-1], destination=buf)
    await _run_turn(m, user, text=m.caption or "", image=buf.getvalue())


@dp.message(F.voice | F.audio)
async def on_voice(m: Message):
    user = await _user(m)
    media = m.voice or m.audio
    buf = io.BytesIO()
    await m.bot.download(media, destination=buf)
    name = "voice.ogg" if m.voice else (m.audio.file_name or "audio.mp3")
    if user.role == "trainer":
        return await m.answer("Голосовые обрабатываются для клиентов. Очередь: /queue")
    await _run_turn(m, user, audio=buf.getvalue(), audio_name=name)


@dp.message(F.text)
async def on_text(m: Message):
    user = await _user(m)
    if user.role == "trainer":
        pending = _awaiting.pop(m.chat.id, None)
        if pending is None:
            return await m.answer("Решения по клиентам — кнопками под сообщениями. Очередь: /queue")
        action, pid = pending
        try:
            if action == "edit":
                await service.decide(user, pid, "edit", m.text)
                return await m.answer("Отправил AI на доработку. Новый черновик придёт сюда.")
            await service.resolve_escalation(user, pid, m.text)
            return await m.answer("Ответ отправлен клиенту.")
        except Exception as e:
            return await m.answer(f"Не получилось: {e}")
    await _run_turn(m, user, text=m.text)


# ── buttons ──────────────────────────────────────────────────────────────────
@dp.callback_query(F.data.startswith("meal:"))
async def on_meal(cb: CallbackQuery):
    user = await _user(cb)
    try:
        r = await service.confirm_meal(user, cb.data.split(":", 1)[1], meal=_meal_now())
        await cb.message.edit_reply_markup(reply_markup=None)
        await cb.message.answer("Уже записано." if r["already"] else f"Записал в дневник: ≈ {r['total_kcal']} ккал.")
    except Exception as e:
        await cb.message.answer(f"Не получилось записать: {e}")
    await cb.answer()


@dp.callback_query(F.data.regexp(r"^(acc|rej|edt|rep|cls):"))
async def on_decision(cb: CallbackQuery):
    user = await _user(cb)
    kind, pid = cb.data.split(":", 1)
    if user.role != "trainer":
        return await cb.answer("Только для тренера", show_alert=True)
    try:
        if kind == "acc":
            p = await service.decide(user, pid, "accept")
            await cb.message.edit_reply_markup(reply_markup=None)
            await cb.message.answer("Применено. Программа клиента обновлена." if p["status"] == "applied"
                                    else f"Статус: {p['status']}. {p.get('reply') or ''}")
        elif kind == "rej":
            await service.decide(user, pid, "reject")
            await cb.message.edit_reply_markup(reply_markup=None)
            await cb.message.answer("Отклонено. Программа не изменилась.")
        elif kind == "edt":
            _awaiting[cb.message.chat.id] = ("edit", pid)
            await cb.message.answer("Напишите одним сообщением, что поправить в черновике.")
        elif kind == "rep":
            _awaiting[cb.message.chat.id] = ("reply", pid)
            await cb.message.answer("Напишите ответ клиенту одним сообщением.")
        elif kind == "cls":
            await service.resolve_escalation(user, pid, None)
            await cb.message.edit_reply_markup(reply_markup=None)
            await cb.message.answer("Закрыто без ответа.")
    except Exception as e:
        await cb.message.answer(f"Не получилось: {e}")
    await cb.answer()


# ── outgoing notifications ───────────────────────────────────────────────────
def _proposal_text(p: dict) -> str:
    who = "запрос клиента" if p.get("source") == "client" else "запрос тренера"
    lines = [f"Черновик для {p.get('client_name', 'клиента')} ({who})", f"«{p.get('request', '')}»", "",
             (p.get("draft") or {}).get("summary", "")]
    for c in p.get("changes") or []:
        lines.append(f"• {c['day']}: {c['was'] or '—'} → {c['becomes']}" + (f" ({c['reason']})" if c.get("reason") else ""))
    warn = [v["message"] for v in (p.get("violations") or [])]
    if warn:
        lines += ["", "Проверка Skill:"] + [f"! {w}" for w in warn]
    return "\n".join(lines)


async def _send_proposal(chat_id: int, p: dict) -> None:
    await _bot.send_message(chat_id, _proposal_text(p), reply_markup=_kb(
        [("Принять", f"acc:{p['id']}"), ("Отклонить", f"rej:{p['id']}")], [("Поправить", f"edt:{p['id']}")]))


async def _send_escalation(chat_id: int, e: dict) -> None:
    reason = (e.get("draft") or {}).get("reason", "")
    await _bot.send_message(chat_id, f"Нужен тренер: {e.get('client_name', 'клиент')}\n«{e.get('request', '')}»\n"
                                     f"Причина: {reason}",
                            reply_markup=_kb([("Ответить", f"rep:{e['id']}"), ("Закрыть", f"cls:{e['id']}")]))


class BotSink:
    async def proposal_pending(self, p: dict) -> None:
        view = await service.proposal_view(p["id"])
        for ch in await get_store().tg_chats_for_user(p["trainer_id"]):
            await _send_proposal(ch["chat_id"], view)

    async def escalation(self, item: dict) -> None:
        if not item.get("trainer_id"):
            return
        view = await service.proposal_view(item["id"])
        for ch in await get_store().tg_chats_for_user(item["trainer_id"]):
            await _send_escalation(ch["chat_id"], view)

    async def to_client(self, client_id: str, text: str) -> None:
        for ch in await get_store().tg_chats_for_user(client_id):
            await _bot.send_message(ch["chat_id"], text)


async def start_bot() -> asyncio.Task:
    global _bot
    _bot = Bot(get_settings().telegram_bot_token)
    notify.add_sink(BotSink())

    async def run():
        try:
            await setup_menu_button(_bot)
            await dp.start_polling(_bot, handle_signals=False)
        finally:
            await _bot.session.close()

    log.info("Telegram bot polling started")
    return asyncio.create_task(run())
