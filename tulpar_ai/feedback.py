"""User feedback on coach answers (👍/👎): constants and the LangSmith side.

The vote is stored locally first (store.feedback) — that is the source of truth for the trainer view and
for tools/export_feedback.py. LangSmith gets a copy attached to the chat_turn run so a bad answer can be
opened with its whole trace. The copy is sent in the background: LangSmith being slow, down or not configured
never delays or fails the vote.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, AsyncIterator

from .pii import mask

log = logging.getLogger("feedback")

RATINGS = ("up", "down")
# Replies produced by the chat graph; trainer replies and «записал в дневник» notes are not the coach's answers.
RATEABLE_KINDS = {"answer", "info", "refusal", "meal_card", "escalated", "proposal"}
MAX_COMMENT = 500
LANGSMITH_KEY = "user_rating"
LANGSMITH_TIMEOUT_S = 8.0
_NS = uuid.UUID("0c7a6f1e-3b52-4f4e-9d0a-5a1f6e2b9c47")
_project_ids: dict[str, Any] = {}
_pending: set[asyncio.Task] = set()
_locks: dict[tuple[str, int], list] = {}  # (purpose, message id) → [lock, users]; gone when nobody holds it


@asynccontextmanager
async def _message_lock(purpose: str, message_id: int) -> AsyncIterator[None]:
    key = (purpose, message_id)
    entry = _locks.setdefault(key, [asyncio.Lock(), 0])
    entry[1] += 1
    try:
        async with entry[0]:
            yield
    finally:
        entry[1] -= 1
        if entry[1] == 0:
            _locks.pop(key, None)


def vote_lock(message_id: int) -> AbstractAsyncContextManager[None]:
    """Held around «store the vote + queue its LangSmith copy»: the copies are then queued in the order the votes
    were stored, so the last one sent is the vote that is kept locally. The bot and the web share this process."""
    return _message_lock("vote", message_id)


def score(rating: str) -> int:
    return 1 if rating == "up" else 0


def feedback_id(run_id: str, message_id: int) -> uuid.UUID:
    """Stable per answer: a changed vote updates the same LangSmith feedback instead of adding a second one.
    The run id is part of it because message ids restart at 1 in every app.sqlite (local, smoke, production)."""
    return uuid.uuid5(_NS, f"{run_id}:message:{message_id}")


def langsmith_client() -> Any:
    """Replaced in tests."""
    from langsmith import Client

    return Client()


def _project_id(client: Any, project: str | None) -> Any:
    """Run-level feedback without the project (session) id is deprecated in LangSmith; resolve it once per name."""
    if not project:
        return None
    if project not in _project_ids:
        try:
            _project_ids[project] = client.read_project(project_name=project).id
        except Exception:
            return None
    return _project_ids[project]


def _push(run_id: str, message_id: int, rating: str, comment: str | None, updated: bool, project: str | None) -> None:
    client, fid = langsmith_client(), feedback_id(run_id, message_id)
    if updated:
        try:
            client.update_feedback(fid, score=score(rating), comment=comment)
            return
        except Exception:  # the first vote may never have reached LangSmith → create it now
            pass
    client.create_feedback(run_id, key=LANGSMITH_KEY, score=score(rating), comment=comment, feedback_id=fid,
                           session_id=_project_id(client, project), stop_after_attempt=2)


async def send_to_langsmith(run_id: str | None, message_id: int, rating: str, comment: str | None,
                            updated: bool = False, project: str | None = None) -> str:
    """-> "sent" | "skipped" (turn was not traced / no key) | "failed" (logged, the vote is kept)."""
    if not run_id or not os.environ.get("LANGSMITH_API_KEY"):
        return "skipped"
    try:
        async with _message_lock("send", message_id):  # a quick 👍→👎 must not let the update overtake the create
            await asyncio.wait_for(asyncio.to_thread(_push, run_id, message_id, rating, mask(comment or "") or None,
                                                     updated, project), LANGSMITH_TIMEOUT_S)
        return "sent"
    except Exception as e:
        log.warning("LangSmith feedback not sent for message %s: %s: %s", message_id, type(e).__name__, str(e)[:200])
        return "failed"


def send_later(run_id: str | None, message_id: int, rating: str, comment: str | None, updated: bool = False,
               project: str | None = None) -> str:
    """-> "queued" | "skipped". The result of the send itself only goes to the log."""
    if not run_id or not os.environ.get("LANGSMITH_API_KEY"):
        return "skipped"
    t = asyncio.create_task(send_to_langsmith(run_id, message_id, rating, comment, updated, project))
    _pending.add(t)
    t.add_done_callback(_pending.discard)
    return "queued"


async def drain() -> list[str]:
    """Wait for queued sends (tests; graceful shutdown)."""
    return list(await asyncio.gather(*list(_pending))) if _pending else []
