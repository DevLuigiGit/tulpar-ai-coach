"""Telegram Mini App entry points: the chat menu button and an inline «Открыть приложение» button.

Both open the same web app; it signs in by itself with WebApp.initData (POST /api/auth/telegram).
Telegram accepts only https URLs, so locally (no WEBAPP_URL / RAILWAY_PUBLIC_DOMAIN) there are no buttons.
"""

from __future__ import annotations

import logging

from aiogram import Bot
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, MenuButtonWebApp, WebAppInfo

from ..config import get_settings

log = logging.getLogger("bot")


def open_app_kb() -> InlineKeyboardMarkup | None:
    url = get_settings().mini_app_url
    if not url:
        return None
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="Открыть приложение",
                                                                       web_app=WebAppInfo(url=url))]])


async def setup_menu_button(bot: Bot) -> None:
    url = get_settings().mini_app_url
    if not url:
        log.info("Mini App menu button skipped: no https WEBAPP_URL / RAILWAY_PUBLIC_DOMAIN")
        return
    try:
        await bot.set_chat_menu_button(menu_button=MenuButtonWebApp(text="Открыть", web_app=WebAppInfo(url=url)))
        log.info("Mini App menu button set to %s", url)
    except Exception:
        log.exception("could not set the Mini App menu button — the bot keeps working without it")
