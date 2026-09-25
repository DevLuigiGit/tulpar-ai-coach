"""User feedback on coach answers (👍/👎): constants and the LangSmith side.

The vote is stored locally first (store.feedback) — that is the source of truth for the trainer view and
for tools/export_feedback.py. LangSmith gets a copy attached to the chat_turn run so a bad answer can be
opened with its whole trace. LangSmith being down or not configured never fails the vote.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from .pii import mask

log = logging.getLogger("feedback")

RATINGS = ("up", "down")
# Replies produced by the chat graph; trainer replies and «записал в дневник» notes are not the coach's answers.
RATEABLE_KINDS = {"answer", "info", "refusal", "meal_card", "escalated", "proposal"}
MAX_COMMENT = 500
LANGSMITH_KEY = "user_rating"
LANGSMITH_TIMEOUT_S = 8.0
_NS = uuid.UUID("0c7a6f1e-3b52-4f4e-9d0a-5a1f6e2b9c47")


def score(rating: str) -> int:
    return 1 if rating == "up" else 0


def feedback_id(message_id: int) -> uuid.UUID:
    """Stable per message: a changed vote updates the same LangSmith feedback instead of adding a second one."""
    return uuid.uuid5(_NS, f"message:{message_id}")


def langsmith_client() -> Any:
    """Replaced in tests."""
    from langsmith import Client

    return Client()


def _push(run_id: str, message_id: int, rating: str, comment: str | None, updated: bool) -> None:
    client, fid = langsmith_client(), feedback_id(message_id)
    if updated:
        try:
            client.update_feedback(fid, score=score(rating), comment=comment)
            return
        except Exception:  # the first vote may never have reached LangSmith → create it now
            pass
    client.create_feedback(run_id, key=LANGSMITH_KEY, score=score(rating), comment=comment, feedback_id=fid,
                           stop_after_attempt=2)


async def send_to_langsmith(run_id: str | None, message_id: int, rating: str, comment: str | None,
                            updated: bool = False) -> str:
    """-> "sent" | "skipped" (turn was not traced / no key) | "failed" (logged, the vote is kept)."""
    if not run_id or not os.environ.get("LANGSMITH_API_KEY"):
        return "skipped"
    try:
        await asyncio.wait_for(asyncio.to_thread(_push, run_id, message_id, rating, mask(comment or "") or None,
                                                 updated), LANGSMITH_TIMEOUT_S)
        return "sent"
    except Exception as e:
        log.warning("LangSmith feedback not sent for message %s: %s: %s", message_id, type(e).__name__, str(e)[:200])
        return "failed"
