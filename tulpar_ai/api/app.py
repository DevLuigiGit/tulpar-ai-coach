"""HTTP API + static web app + (optionally) the Telegram bot, in one process.

Contract used by web/, bot and the MCP server — see ARCHITECTURE.md «API».
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import service, tts
from ..config import ROOT, get_settings
from ..gateway import build_gateway, get_gateway, set_gateway
from ..gateway.base import User
from ..graph import runner
from ..rag.index import Index
from ..rag.retrieve import get_index, set_index
from ..store import Store, get_store, set_store
from .auth import client_user, current_user, issue_token, trainer_user
from .ratelimit import limit_chat, limit_login, limit_trainer
from .telegram_auth import router as telegram_auth_router

log = logging.getLogger("api")
MAX_UPLOAD = 8 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    store = await Store(s.data_path("app.sqlite")).open()
    set_store(store)
    gw = build_gateway(store)
    await gw.start()
    set_gateway(gw)
    await runner.open_graphs()
    set_index(Index())
    try:
        n = await get_index().build()
        log.info("RAG index ready: %s chunks (%s)", n, get_index().collection)
    except Exception:
        log.exception("RAG index build failed — questions will be escalated until it is fixed")
    await runner.recover_drafting()
    bot_task = None
    if s.telegram_bot_token:
        from ..bot.bot import start_bot

        bot_task = await start_bot()
    yield
    if bot_task is not None:
        bot_task.cancel()
    await runner.close_graphs()
    get_index().close()
    set_index(None)
    await gw.close()
    await store.close()
    set_store(None)
    set_gateway(None)


app = FastAPI(title="Tulpar AI Coach", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_list, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(telegram_auth_router)  # POST /api/auth/telegram — Mini App login


# ── health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"ok": True, "mode": get_gateway().mode}


@app.get("/health/ai")
async def health_ai():
    s = get_settings()
    import os

    return {"ollama": bool(s.ollama_api_key), "groq": bool(s.groq_api_key), "jina": bool(s.jina_api_key),
            "langsmith": bool(os.environ.get("LANGSMITH_API_KEY")),
            "langsmith_tracing": os.environ.get("LANGSMITH_TRACING", "").lower() == "true",
            "telegram": bool(s.telegram_bot_token),
            "rag_chunks": get_index().count(), "mode": s.tulpar_mode}


# ── auth ─────────────────────────────────────────────────────────────────────
class DemoLogin(BaseModel):
    role: str


@app.post("/api/auth/demo-login", dependencies=[Depends(limit_login)])
async def demo_login(body: DemoLogin):
    if not get_settings().allow_demo_login:
        raise HTTPException(404)
    if body.role not in ("client", "trainer"):
        raise HTTPException(422, "role must be client or trainer")
    user = await get_gateway().demo_user(body.role)
    return {"token": issue_token(user), "user": user}


@app.get("/api/me")
async def me(user: User = Depends(current_user)):
    return user


# ── client ───────────────────────────────────────────────────────────────────
async def _read(f: UploadFile | None) -> bytes | None:
    if f is None:
        return None
    data = await f.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file too large (max 8 MB)")
    return data or None


@app.post("/api/chat")
async def chat(text: str = Form(""), photo: UploadFile | None = File(None), audio: UploadFile | None = File(None),
               user: User = Depends(limit_chat)):
    image, voice = await _read(photo), await _read(audio)
    if not (text.strip() or image or voice):
        raise HTTPException(422, "send text, a photo or a voice message")
    return await service.chat_turn(user, text.strip(), image=image, audio=voice,
                                   audio_name=(audio.filename if audio else "voice.ogg") or "voice.ogg")


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


@app.post("/api/tts", response_class=Response)
async def speak(body: SpeakRequest, user: User = Depends(client_user)):
    """A coach reply as MP3 for the web play button. Only clients: the voice is part of their chat."""
    if not get_settings().tts_enabled:
        raise HTTPException(404, "Озвучка ответов отключена")
    try:
        audio = await tts.synthesize(body.text)
    except ValueError:
        raise HTTPException(422, "В ответе нечего озвучивать")
    except Exception:
        log.exception("tts failed")
        raise HTTPException(502, "Не получилось озвучить ответ, попробуйте ещё раз")
    return Response(audio, media_type="audio/mpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/chat/history")
async def history(limit: int = 50, user: User = Depends(client_user)):
    return await get_store().history(user.id, limit=min(limit, 200))


class ConfirmMeal(BaseModel):
    grams: dict[int, float] | None = None
    meal: str | None = None


@app.post("/api/meals/{card_id}/confirm")
async def confirm_meal(card_id: str, body: ConfirmMeal, user: User = Depends(client_user)):
    try:
        return await service.confirm_meal(user, card_id, body.grams, body.meal)
    except LookupError:
        raise HTTPException(404, "meal card not found")


@app.get("/api/my/proposals")
async def my_proposals(user: User = Depends(client_user)):
    return await service.my_proposals(user)


@app.get("/api/my/plan")
async def my_plan(user: User = Depends(client_user)):
    return await get_gateway().active_plan(user.id)


# ── trainer ──────────────────────────────────────────────────────────────────
@app.get("/api/trainer/clients")
async def clients(user: User = Depends(trainer_user)):
    return await get_gateway().list_clients(user.id)


async def _own_client(trainer: User, client_id: str) -> None:
    t = await get_gateway().trainer_of(client_id)
    if t is None or t.id != trainer.id:
        raise HTTPException(404, "client not found")


@app.get("/api/trainer/clients/{client_id}/context")
async def client_context(client_id: str, user: User = Depends(trainer_user)):
    await _own_client(user, client_id)
    return await get_gateway().client_context(client_id)


@app.get("/api/trainer/clients/{client_id}/chat")
async def client_chat(client_id: str, limit: int = 50, user: User = Depends(trainer_user)):
    await _own_client(user, client_id)
    return await get_store().history(client_id, limit=min(limit, 200))


class ChangeRequest(BaseModel):
    client_id: str
    request: str


@app.post("/api/trainer/proposals")
async def request_change(body: ChangeRequest, user: User = Depends(limit_trainer)):
    try:
        return await service.request_change(user, body.client_id, body.request.strip())
    except PermissionError:
        raise HTTPException(404, "client not found")
    except service.RejectedText as e:
        raise HTTPException(422, str(e))


@app.get("/api/queue")
async def queue(history: bool = False, user: User = Depends(trainer_user)):
    return await service.queue(user, history=history)


@app.get("/api/proposals/{pid}")
async def proposal(pid: str, user: User = Depends(current_user)):
    p = await service.proposal_view(pid)
    if p is None or user.id not in (p["trainer_id"], p["client_id"]):
        raise HTTPException(404, "not found")
    return p


class Decision(BaseModel):
    action: str
    comment: str | None = None


@app.post("/api/proposals/{pid}/decision")
async def decision(pid: str, body: Decision, user: User = Depends(limit_trainer)):
    try:
        return await service.decide(user, pid, body.action, body.comment)
    except LookupError:
        raise HTTPException(404, "proposal not found")
    except PermissionError:
        raise HTTPException(403, "not your proposal")
    except service.RejectedText as e:
        raise HTTPException(422, str(e))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(409, str(e))


class Resolve(BaseModel):
    reply: str | None = None


@app.post("/api/escalations/{eid}/resolve")
async def resolve(eid: str, body: Resolve, user: User = Depends(trainer_user)):
    try:
        return await service.resolve_escalation(user, eid, body.reply)
    except LookupError:
        raise HTTPException(404, "not found")
    except PermissionError:
        raise HTTPException(403, "not your client")


@app.get("/api/exercises")
async def exercises(q: str | None = None, muscle_group: str | None = None, equipment: str | None = None,
                    avoid: str | None = None, user: User = Depends(trainer_user)):
    return service.search_exercises(q, muscle_group, equipment, avoid)


# ── static web app (built web/dist), SPA fallback ───────────────────────────
DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        f = DIST / path
        if path and f.is_file() and DIST in f.resolve().parents:
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
