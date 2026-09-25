"""Telegram Mini App login: initData signature/freshness and POST /api/auth/telegram (no network)."""

import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient

from tulpar_ai.api.telegram_auth import InitDataError, validate_init_data

TOKEN = "123456:TEST-bot-token"
TRAINER_ID = 777001


def make_init_data(user_id=42, first_name="Айдос", token=TOKEN, auth_date=None, **extra) -> str:
    fields = {"auth_date": str(int(time.time()) if auth_date is None else auth_date),
              "query_id": "AAHdF6IQAAAAAN0XohDhrOrc",
              "user": json.dumps({"id": user_id, "first_name": first_name, "language_code": "ru"}, ensure_ascii=False),
              **extra}
    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields)


def test_valid_init_data():
    data = validate_init_data(make_init_data(signature="abc"), TOKEN, 3600)
    assert data["user"]["id"] == 42 and data["user"]["first_name"] == "Айдос"
    assert isinstance(data["auth_date"], int) and "hash" not in data


def test_tampered_hash():
    raw = make_init_data()
    bad = raw[:-1] + ("0" if raw[-1] != "0" else "1")
    with pytest.raises(InitDataError):
        validate_init_data(bad, TOKEN, 3600)


def test_tampered_field():
    raw = make_init_data(user_id=42)
    forged = raw.replace("%22id%22%3A+42", "%22id%22%3A+43")
    assert forged != raw
    with pytest.raises(InitDataError):
        validate_init_data(forged, TOKEN, 3600)


def test_wrong_token():
    with pytest.raises(InitDataError):
        validate_init_data(make_init_data(token="999:other"), TOKEN, 3600)


def test_stale_auth_date():
    old = int(time.time()) - 7200
    with pytest.raises(InitDataError, match="expired"):
        validate_init_data(make_init_data(auth_date=old), TOKEN, 3600)
    assert validate_init_data(make_init_data(auth_date=old), TOKEN, 86400)["user"]["id"] == 42


def test_missing_hash_and_garbage():
    raw = make_init_data()
    no_hash = "&".join(p for p in raw.split("&") if not p.startswith("hash="))
    for bad in (no_hash, "", "not-a-query", "hash="):
        with pytest.raises(InitDataError):
            validate_init_data(bad, TOKEN, 3600)


def _configure(monkeypatch, token=TOKEN):
    # Set after startup: a token present at lifespan time would start real bot polling.
    from tulpar_ai.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "telegram_bot_token", token)
    monkeypatch.setattr(s, "trainer_telegram_ids", f"{TRAINER_ID}, 5")
    monkeypatch.setattr(s, "allow_demo_login", False)


def test_telegram_login_client_and_trainer(env, fake_llm, monkeypatch):
    from tulpar_ai.api.app import app

    with TestClient(app) as c:
        _configure(monkeypatch)
        assert c.post("/api/auth/demo-login", json={"role": "client"}).status_code == 404  # demo off, Telegram still works

        r = c.post("/api/auth/telegram", json={"init_data": make_init_data(user_id=42, first_name="Айдос")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["user"]["role"] == "client" and body["user"]["name"] == "Айдос"
        h = {"Authorization": f"Bearer {body['token']}"}
        assert c.get("/api/me", headers=h).json()["id"] == body["user"]["id"]
        assert c.get("/api/my/plan", headers=h).status_code == 200

        again = c.post("/api/auth/telegram", json={"init_data": make_init_data(user_id=42)}).json()
        assert again["user"]["id"] == body["user"]["id"]  # same account on every open

        r = c.post("/api/auth/telegram", json={"init_data": make_init_data(user_id=TRAINER_ID, first_name="Тренер")})
        assert r.status_code == 200, r.text
        tr = r.json()
        assert tr["user"]["role"] == "trainer"
        assert c.get("/api/queue", headers={"Authorization": f"Bearer {tr['token']}"}).status_code == 200


def test_telegram_login_rejects(env, fake_llm, monkeypatch):
    from tulpar_ai.api.app import app

    with TestClient(app) as c:
        assert c.post("/api/auth/telegram", json={"init_data": make_init_data()}).status_code == 404  # no bot token
        _configure(monkeypatch)
        assert c.post("/api/auth/telegram", json={"init_data": make_init_data(token="999:other")}).status_code == 401
        stale = make_init_data(auth_date=int(time.time()) - 3 * 86400)
        assert c.post("/api/auth/telegram", json={"init_data": stale}).status_code == 401
        assert c.post("/api/auth/telegram", json={"init_data": ""}).status_code == 401


def test_mini_app_url(env, monkeypatch):
    from tulpar_ai.config import Settings

    monkeypatch.delenv("RAILWAY_PUBLIC_DOMAIN", raising=False)
    monkeypatch.delenv("WEBAPP_URL", raising=False)
    assert Settings(webapp_url="").mini_app_url == ""
    assert Settings(webapp_url="http://localhost:8089").mini_app_url == ""  # Telegram needs https
    assert Settings(webapp_url="https://coach.example.com").mini_app_url == "https://coach.example.com/"
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "coach-production.up.railway.app")
    assert Settings(webapp_url="").mini_app_url == "https://coach-production.up.railway.app/"


def test_non_hex_hash_is_rejected_not_crashing():
    import pytest
    from tulpar_ai.api.telegram_auth import validate_init_data
    with pytest.raises(Exception) as e:
        validate_init_data("auth_date=1&user=%7B%22id%22%3A1%7D&hash=%D0%B9", "123:ABC", 86400)
    assert not isinstance(e.value, TypeError)


def test_open_app_button_only_in_private_chats(monkeypatch):
    from tulpar_ai.bot import miniapp
    from tulpar_ai.config import get_settings
    monkeypatch.setattr(get_settings(), "webapp_url", "https://example.org")
    assert miniapp.open_app_kb("private") is not None
    assert miniapp.open_app_kb("group") is None
