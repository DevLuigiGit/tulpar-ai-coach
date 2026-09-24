"""Lifecycle of both graphs: checkpointer, turn execution, proposal start/resume."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from ..config import get_settings
from ..store import get_store
from .chat import build_chat_graph
from .program import build_program_graph

log = logging.getLogger("runner")

_conn: aiosqlite.Connection | None = None
_saver: AsyncSqliteSaver | None = None
_chat = None
_program = None
_tasks: set[asyncio.Task] = set()


async def open_graphs(path: Path | None = None) -> None:
    """Open ONE long-lived connection for the checkpointer (a `with from_conn_string` block would close it
    as soon as the block ends — and the next resume would fail)."""
    global _conn, _saver, _chat, _program
    path = path or get_settings().data_path("graph.sqlite")
    _conn = await aiosqlite.connect(str(path))
    _saver = AsyncSqliteSaver(_conn)
    await _saver.setup()
    _chat = build_chat_graph().compile()
    _program = build_program_graph().compile(checkpointer=_saver)


async def close_graphs() -> None:
    global _conn, _saver, _chat, _program
    for t in list(_tasks):
        t.cancel()
    if _conn is not None:
        await _conn.close()
    _conn = _saver = _chat = _program = None


def program_graph():
    return _program


def chat_graph():
    return _chat


def _cfg(thread: str, run_name: str, **meta) -> dict:
    return {"configurable": {"thread_id": thread}, "run_name": run_name, "metadata": meta, "tags": [run_name]}


async def run_chat_turn(client_id: str, text: str = "", image: bytes | None = None, audio: bytes | None = None,
                        audio_name: str = "voice.ogg") -> dict:
    turn = str(uuid.uuid4())
    state: dict = {"client_id": client_id, "turn_id": turn, "text": text or ""}
    if image:
        p = get_settings().data_path("uploads", f"{turn}.jpg")
        p.write_bytes(image)
        state["image_path"] = str(p)
    if audio:
        ext = Path(audio_name).suffix or ".ogg"
        p = get_settings().data_path("uploads", f"{turn}{ext}")
        p.write_bytes(audio)
        state["audio_path"] = str(p)
    result = await _chat.ainvoke(state, _cfg(f"chat:{turn}", "coach_turn", client=client_id[:8]))
    for key in ("image_path", "audio_path"):
        if state.get(key):
            Path(state[key]).unlink(missing_ok=True)  # uploads are not kept
    return result


async def new_program_proposal(client_id: str, trainer_id: str, request: str, source: str) -> dict:
    p = await get_store().create_proposal(kind="program", client_id=client_id, trainer_id=trainer_id, source=source,
                                          request=request, status="drafting")
    spawn(start_program(p["id"]))
    return p


def spawn(coro) -> asyncio.Task:
    t = asyncio.create_task(coro)
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)
    return t


async def start_program(proposal_id: str) -> dict:
    p = await get_store().get_proposal(proposal_id)
    init = {"proposal_id": p["id"], "client_id": p["client_id"], "trainer_id": p["trainer_id"], "request": p["request"],
            "source": p["source"]}
    try:
        return await _program.ainvoke(init, _cfg(f"prop:{proposal_id}", "program_change", proposal=proposal_id[:8]))
    except Exception as e:
        log.exception("program graph failed")
        await get_store().update_proposal(proposal_id, status="failed", reply=f"{type(e).__name__}: {e}"[:300])
        raise


async def resume_program(proposal_id: str, action: str, comment: str | None = None) -> dict:
    cfg = _cfg(f"prop:{proposal_id}", "program_change", proposal=proposal_id[:8])
    snap = await _program.aget_state(cfg)
    if not snap.next or "review" not in snap.next:
        raise RuntimeError("proposal is not waiting for a decision")
    return await _program.ainvoke(Command(resume={"action": action, "comment": comment}), cfg)


async def recover_drafting() -> int:
    """After a restart, finish drafts that were interrupted mid-way (not the ones paused for review)."""
    n = 0
    for p in await get_store().list_proposals(statuses=["drafting"]):
        cfg = _cfg(f"prop:{p['id']}", "program_change")
        snap = await _program.aget_state(cfg)
        if snap.next and "review" not in snap.next:
            spawn(_program.ainvoke(None, cfg))
            n += 1
        elif not snap.next and not snap.values:
            spawn(start_program(p["id"]))
            n += 1
    return n
