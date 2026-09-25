"""Login from the Telegram Mini App: the bot opens the same web app, and it signs in with WebApp.initData.

Validation follows Telegram's docs («Validating data received via the Mini App»):
secret_key = HMAC_SHA256(key="WebAppData", msg=bot_token); the hash is HMAC_SHA256(secret_key, data_check_string).
The Telegram user is mapped exactly like the bot maps it, so the web session and the bot chat are the same account.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from urllib.parse import parse_qsl

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import get_settings
from ..gateway import get_gateway
from .auth import issue_token


class InitDataError(ValueError):
    pass


def validate_init_data(init_data: str, bot_token: str, max_age_s: int = 86400, now: float | None = None) -> dict:
    """Checks the signature and freshness of WebApp.initData; returns its fields with `user` parsed from JSON."""
    if not init_data or not bot_token:
        raise InitDataError("empty init data")
    try:
        pairs = parse_qsl(init_data, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        raise InitDataError("malformed init data")
    data: dict[str, str] = {}
    for key, value in pairs:
        if key in data:
            raise InitDataError(f"duplicate field {key}")
        data[key] = value
    received = data.pop("hash", "")
    if not received:
        raise InitDataError("missing hash")
    check = "\n".join(f"{k}={v}" for k, v in sorted(data.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    expected = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, received.lower()):
        raise InitDataError("bad hash")
    try:
        auth_date = int(data.get("auth_date", ""))
    except ValueError:
        raise InitDataError("missing auth_date")
    if (time.time() if now is None else now) - auth_date > max_age_s:
        raise InitDataError("init data expired")
    try:
        user = json.loads(data.get("user") or "null")
    except json.JSONDecodeError:
        raise InitDataError("bad user field")
    if not isinstance(user, dict) or not isinstance(user.get("id"), int):
        raise InitDataError("no user in init data")
    return {**data, "auth_date": auth_date, "user": user}


router = APIRouter()


class TelegramLogin(BaseModel):
    init_data: str


@router.post("/api/auth/telegram")
async def telegram_login(body: TelegramLogin):
    s = get_settings()
    if not s.telegram_bot_token:
        raise HTTPException(404, "Telegram login is not configured")
    try:
        tg = validate_init_data(body.init_data, s.telegram_bot_token, s.telegram_auth_max_age_s)
    except InitDataError as e:
        raise HTTPException(401, f"Telegram data rejected: {e}")
    u = tg["user"]
    tg_id = str(u["id"])
    try:
        # Same mapping as bot._user(): trainers are listed in TRAINER_TELEGRAM_IDS.
        user = await get_gateway().telegram_user(tg_id, u.get("first_name") or "Клиент",
                                                 as_trainer=tg_id in s.trainer_tg_ids)
    except LookupError:
        raise HTTPException(403, "this Telegram account is not linked to a user")
    return {"token": issue_token(user), "user": user}
