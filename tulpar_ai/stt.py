"""Speech-to-text for voice notes (Groq Whisper). Traced as its own LangSmith run."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import httpx
from langsmith import traceable

from .config import get_settings

_fake: Callable[[bytes, str], str] | None = None


def set_fake(fn: Callable[[bytes, str], str] | None) -> None:
    global _fake
    _fake = fn


def trace_stt_inputs(inputs: dict) -> dict:
    """The voice note itself stays out of the trace (size and format only); the transcript is the output."""
    audio = inputs.get("audio") or b""
    fmt = Path(inputs.get("filename") or "").suffix.lstrip(".") or "ogg"
    return {"audio": {"bytes": len(audio), "format": fmt}, "language": inputs.get("language")}


@traceable(run_type="tool", name="speech_to_text", process_inputs=trace_stt_inputs)
async def transcribe(audio: bytes, filename: str = "voice.ogg", language: str = "ru") -> str:
    if _fake is not None:
        return _fake(audio, filename)
    s = get_settings()
    if not s.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set — voice input needs it")
    # Groq decides the codec from the extension; Telegram voice notes are OGG/Opus.
    if "." not in filename:
        filename += ".ogg"
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(
            f"{s.groq_url}/audio/transcriptions",
            headers={"Authorization": f"Bearer {s.groq_api_key}"},
            data={"model": s.stt_model, "language": language, "response_format": "json", "temperature": "0"},
            files={"file": (filename, audio)},
        )
    if r.status_code >= 400:
        raise RuntimeError(f"stt {r.status_code}: {r.text[:200]}")
    return (r.json().get("text") or "").strip()
