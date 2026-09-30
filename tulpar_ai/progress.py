"""Live progress of one chat turn for POST /api/chat/stream: the stage the graph is at, for the «Ищу в базе…» line.

Only stages: the reply itself still comes whole, exactly as from POST /api/chat. Streaming the answer text gave no
gain (EVALS finding 27: the first words came no earlier than the whole reply), while the stage label is instant.
Nodes call `stage(...)`; without a bound listener (POST /api/chat, the bot, evals) it does nothing.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from typing import Callable

log = logging.getLogger("progress")

STAGES = ("listen", "route", "search", "answer", "meal", "photo", "program")

_listener: ContextVar[Callable[[str], None] | None] = ContextVar("tulpar_progress", default=None)
_last: ContextVar[list | None] = ContextVar("tulpar_progress_last", default=None)


def bind(listener: Callable[[str], None]) -> Token:
    _last.set([None])
    return _listener.set(listener)


def unbind(token: Token) -> None:
    _listener.reset(token)


def stage(name: str) -> None:
    """Report a stage; a repeat of the same stage (the retrieve → rewrite loop) is sent once."""
    listener = _listener.get()
    if listener is None:
        return
    last = _last.get()
    if last is not None:
        if last[0] == name:
            return
        last[0] = name
    try:
        listener(name)
    except Exception:  # a broken listener must not fail the turn
        log.warning("progress listener failed", exc_info=True)
