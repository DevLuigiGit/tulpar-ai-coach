"""Fan-out of events to whoever listens (the Telegram bot when it runs; the web UI polls the store)."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Protocol

log = logging.getLogger("notify")


class Sink(Protocol):
    async def proposal_pending(self, proposal: dict) -> None: ...
    async def escalation(self, item: dict) -> None: ...
    async def to_client(self, client_id: str, text: str) -> None: ...


_sinks: list[Sink] = []


def add_sink(sink: Sink) -> None:
    _sinks.append(sink)


def clear_sinks() -> None:
    _sinks.clear()


async def _each(fn: Callable[[Sink], Awaitable[None]]) -> None:
    for s in list(_sinks):
        try:
            await fn(s)
        except Exception:  # a broken sink must never break the graph
            log.exception("notification sink failed")


async def proposal_pending(p: dict) -> None:
    await _each(lambda s: s.proposal_pending(p))


async def escalation(item: dict) -> None:
    await _each(lambda s: s.escalation(item))


async def to_client(client_id: str, text: str) -> None:
    await _each(lambda s: s.to_client(client_id, text))
