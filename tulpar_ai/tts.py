"""Text-to-speech for voice replies (Microsoft Edge neural voices via `edge-tts`). Traced as its own LangSmith run.

Why edge-tts: Groq serves only English/Arabic TTS (Orpheus) and Ollama Cloud has none, while edge-tts has
ru-RU and kk-KZ neural voices, needs no key and returns MP3 — which Telegram accepts for send_voice as is.
The reply is written for reading, so it is cleaned for listening first: no [n] citations, no markdown,
no disclaimer line, units spelled out, and a length cap (the full answer stays in the text message).
"""

from __future__ import annotations

import asyncio
import re
from typing import Callable

from langsmith import traceable

from .config import get_settings
from .pii import mask

_fake: Callable[[str, str], bytes] | None = None

DISCLAIMER = "Это общая информация, а не медицинская консультация."
TRUNCATED_TAIL = "Остальное — в текстовом ответе."
# Letters that exist in Kazakh Cyrillic but not in Russian: enough of them means the Kazakh voice.
_KK_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")

_CITE = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_URL = re.compile(r"https?://\S+")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|`+|~~)(?=\S)(.+?)(?<=\S)\1")
_LINE_MARK = re.compile(r"^\s*(?:#{1,6}\s+|[-*•·]\s+|>\s*|\d+[.)]\s+)", re.M)
_UNITS = [
    (re.compile(r"≈\s*"), "примерно "),
    (re.compile(r"(\d)\s*ккал\b"), r"\1 килокалорий"),
    (re.compile(r"(\d)\s*г\b"), r"\1 грамм"),
    (re.compile(r"(\d)\s*мин\b\.?"), r"\1 минут"),
    (re.compile(r"\bстр\."), "страница"),
    (re.compile(r"\bпоз\."), "позиций"),
]


def set_fake(fn: Callable[[str, str], bytes] | None) -> None:
    global _fake
    _fake = fn


def speech_text(text: str, max_chars: int | None = None) -> str:
    """The reply as it should sound: plain sentences, one line per list item, capped on a sentence boundary."""
    limit = max_chars or get_settings().tts_max_chars
    out = text.replace(DISCLAIMER, "")
    out = _LINK.sub(r"\1", out)
    out = _URL.sub("", out)
    out = _CITE.sub("", out)
    out = _EMPHASIS.sub(r"\2", out)
    out = _LINE_MARK.sub("", out)
    for pattern, repl in _UNITS:
        out = pattern.sub(repl, out)
    # A list item without end punctuation would run into the next one when spoken.
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    out = " ".join(ln if ln[-1] in ".!?:;…" else ln + "." for ln in lines)
    out = re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s+", " ", out)).strip()
    if len(out) <= limit:
        return out
    cut = out[:limit]
    end = max(cut.rfind(p) for p in (". ", "! ", "? ", "… "))
    cut = cut[: end + 1] if end >= limit // 2 else cut[: cut.rfind(" ")].rstrip(",;:—- ") + "."
    return f"{cut} {TRUNCATED_TAIL}"


def pick_voice(text: str) -> str:
    s = get_settings()
    return s.tts_voice_kk if sum(ch in _KK_LETTERS for ch in text) >= 3 else s.tts_voice


def _trace_outputs(audio: bytes | None) -> dict:
    # The MP3 itself is useless in a trace and heavy; its size is what matters. None on error.
    return {"bytes": len(audio) if isinstance(audio, (bytes, bytearray)) else 0}


@traceable(run_type="tool", name="text_to_speech", process_outputs=_trace_outputs)
async def synthesize(text: str) -> bytes:
    """MP3 (24 kHz, 48 kbit/s, mono) of the cleaned reply. Raises on empty text, timeout or an empty stream."""
    s = get_settings()
    # The text leaves for a third-party service: same masking as for LLM calls.
    clean = mask(speech_text(text, s.tts_max_chars))
    if not clean:
        raise ValueError("nothing to speak")
    voice = pick_voice(clean)
    if _fake is not None:
        return _fake(clean, voice)
    import edge_tts  # imported lazily: only voice replies need it

    async def run() -> bytes:
        buf = bytearray()
        async for chunk in edge_tts.Communicate(clean, voice, rate=s.tts_rate).stream():
            if chunk["type"] == "audio":
                buf += chunk["data"]
        return bytes(buf)

    audio = await asyncio.wait_for(run(), timeout=s.tts_timeout_s)
    if not audio:
        raise RuntimeError("tts returned no audio")
    return audio
