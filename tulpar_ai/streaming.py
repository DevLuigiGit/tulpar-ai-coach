"""Live progress of one chat turn for POST /api/chat/stream.

The graph does not know about HTTP. A turn started by the stream endpoint binds an event sink in a context variable;
nodes call `stage(...)` / `delta(...)`, which do nothing when no sink is bound (POST /api/chat, the bot, evals).
The final result still comes from `service.chat_turn`, so the stored message, the cache and the trace do not depend
on whether the turn was streamed.

Answer text reaches the client only through `GuardedText`: the model's JSON is decoded as it arrives
(`JsonStringField`), cut at sentence ends, and a piece is released only after the output guard has passed the whole
text up to that point. A blocked prefix stops the stream; the final `done` event then carries the guarded reply.
"""

from __future__ import annotations

import logging
import re
from contextvars import ContextVar, Token
from typing import Callable

from . import guardrails

log = logging.getLogger("streaming")

Sink = Callable[[dict], None]
_sink: ContextVar[Sink | None] = ContextVar("tulpar_stream_sink", default=None)
_last_stage: ContextVar[list | None] = ContextVar("tulpar_stream_last_stage", default=None)

STAGES = ("listen", "route", "search", "answer", "meal")


def bind(sink: Sink) -> Token:
    _last_stage.set([None])
    return _sink.set(sink)


def unbind(token: Token) -> None:
    _sink.reset(token)


def active() -> bool:
    return _sink.get() is not None


def emit(event: dict) -> None:
    sink = _sink.get()
    if sink is None:
        return
    try:
        sink(event)
    except Exception:  # a broken listener must not fail the turn
        log.warning("stream sink failed", exc_info=True)


def stage(name: str) -> None:
    """Progress marker; repeats of the same stage (the rewrite loop) are sent once."""
    last = _last_stage.get()
    if last is not None:
        if last[0] == name:
            return
        last[0] = name
    emit({"type": "stage", "stage": name})


def delta(text: str) -> None:
    if text:
        emit({"type": "delta", "text": text})


# ── incremental JSON: the value of one top-level string field ────────────────
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


class JsonStringField:
    """Feed raw model output piece by piece; `feed` returns the newly decoded characters of `field`'s string value.

    Tracks JSON structure outside strings, so the name inside another value or in a nested object does not match.
    Handles escapes split across pieces, \\uXXXX and surrogate pairs; text before the first `{` (``` fences, prose)
    is skipped. Anything it cannot follow simply yields nothing: the full reply is parsed with json afterwards anyway.
    """

    def __init__(self, field: str = "answer"):
        self.field = field
        self.stack: list[str] = []
        self.expect_key = False
        self.in_str = False
        self.is_key = False
        self.capture = False
        self.esc = ""
        self.buf: list[str] = []  # the key being read
        self.key: str | None = None  # last key at the top level
        self.high: int | None = None  # pending high surrogate
        self.done = False
        self.closed = False  # the field's string has ended: its text is complete

    def feed(self, chunk: str) -> str:
        out: list[str] = []
        for ch in chunk:
            if self.done:
                break
            if self.in_str:
                self._in_string(ch, out)
            else:
                self._structure(ch)
        return "".join(out)

    def _structure(self, ch: str) -> None:
        if not self.stack and ch != "{":
            return  # before the object: fences, prose
        if ch == '"':
            self.in_str = True
            self.is_key = self.stack[-1] == "{" and self.expect_key
            self.capture = not self.is_key and len(self.stack) == 1 and self.key == self.field
            self.buf = []
        elif ch in "{[":
            self.stack.append(ch)
            self.expect_key = ch == "{"
        elif ch in "}]":
            if self.stack:
                self.stack.pop()
            self.expect_key = False
            if not self.stack:
                self.done = True  # the top-level object is closed
        elif ch == ":":
            self.expect_key = False
        elif ch == ",":
            self.expect_key = self.stack[-1] == "{"
        elif not ch.isspace() and len(self.stack) == 1 and self.key == self.field and not self.expect_key:
            self.key = None  # the field is not a string (null, number…): nothing to stream

    def _in_string(self, ch: str, out: list[str]) -> None:
        if self.esc:
            self.esc += ch
            if self.esc[1] == "u":
                if len(self.esc) < 6:
                    return
                try:
                    code = int(self.esc[2:], 16)
                except ValueError:
                    code = 0xFFFD
                self.esc = ""
                self._code(code, out)
            else:
                self.esc = ""
                self._put(_ESCAPES.get(ch, ch), out)
            return
        if ch == "\\":
            self.esc = "\\"
            return
        if ch == '"':
            self._flush_high(out)
            self.in_str = False
            if self.is_key:
                if len(self.stack) == 1:
                    self.key = "".join(self.buf)
            elif self.capture:
                self.capture = False
                self.done = self.closed = True
            return
        self._put(ch, out)

    def _code(self, code: int, out: list[str]) -> None:
        if 0xD800 <= code <= 0xDBFF:
            self._flush_high(out)
            self.high = code
        elif 0xDC00 <= code <= 0xDFFF and self.high is not None:
            c = chr(0x10000 + ((self.high - 0xD800) << 10) + (code - 0xDC00))
            self.high = None
            self._put(c, out)
        else:
            self._put(chr(code), out)

    def _flush_high(self, out: list[str]) -> None:
        if self.high is not None:  # a lone surrogate cannot be sent as UTF-8
            self.high = None
            self._emit("\ufffd", out)

    def _put(self, c: str, out: list[str]) -> None:
        self._flush_high(out)
        self._emit(c, out)

    def _emit(self, c: str, out: list[str]) -> None:
        if self.is_key:
            self.buf.append(c)
        elif self.capture:
            out.append(c)


# ── sentence release behind the output guard ─────────────────────────────────
# The same boundary as guardrails.gives_dosage (`(?<=[.!?;\n])\s+`): a released prefix always ends between two of
# the guard's sentences, so the guard judges exactly the sentences the client will see.
_BOUNDARY = re.compile(r"[.!?;\n]\s+")


class GuardedText:
    """Answer text in, guarded pieces out through `send`.

    The raw text is kept; at each sentence end the whole raw prefix goes through `guardrails.guard_reply`. A blocked
    prefix (dosage, prompt leak) stops the stream for good; otherwise the masked prefix is compared with what was
    already sent and only the new part goes out. If masking ever rewrites text that was already sent (a phone number
    split by a sentence end), the stream stops too: the final message carries the masked text.
    """

    def __init__(self, send: Callable[[str], None]):
        self.send = send
        self.raw = ""
        self.cut = 0  # raw[:cut] has been judged and released
        self.shown = ""  # what the client has, after masking
        self.stopped = False
        self.closed = False
        self.blocked: str | None = None  # guard action that stopped the stream

    def push(self, text: str) -> None:
        if not text:
            return
        if not self.raw:
            text = text.lstrip()  # the final reply is stripped; keep the two comparable
        self.raw += text
        if self.stopped:
            return
        end = None
        for m in _BOUNDARY.finditer(self.raw, self.cut):
            end = m.end()
        if end is not None:
            self._release(self.raw[:end], end)

    def close(self, final_text: str) -> None:
        """The answer text is complete (its JSON string ended). `final_text` is the reply the service will judge
        (stripped answer + disclaimer): if the guard blocks it, nothing more goes out; otherwise the last sentence
        does, without waiting for the rest of the JSON."""
        if self.stopped or self.closed:
            return
        self.closed = True
        action = guardrails.guard_reply(final_text)[1]
        if action.startswith("blocked"):
            self.stopped, self.blocked = True, action
            return
        self._release(self.raw.rstrip(), len(self.raw))

    def finish(self, final_text: str) -> None:
        """The reply is final and is an answer: judge `final_text` like the service will and send the rest."""
        if not self.stopped:
            self._release(final_text, len(self.raw))
        self.stopped = True

    def _release(self, prefix: str, cut: int) -> None:
        text, action = guardrails.guard_reply(prefix)
        if action.startswith("blocked"):
            self.stopped, self.blocked = True, action
            return
        if not text.startswith(self.shown):
            self.stopped = True
            return
        piece, self.shown, self.cut = text[len(self.shown):], text, cut
        if piece:
            self.send(piece)
