"""Speech-to-text for voice notes (Groq Whisper). Traced as its own LangSmith run.

The recognition language is a setting (STT_LANGUAGE, chosen by evals/voice_eval.py, see EVALS.md):
  ru   — always Russian (the original behaviour);
  kk   — always Kazakh;
  auto — no language hint, Whisper detects it itself; a language other than ru/kk is asked again in ru;
  hint — per client: Kazakh when the client's own recent text (or Telegram UI language) is Kazakh, else Russian.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import httpx
from langsmith import traceable

from .config import get_settings
from .lang import is_kazakh, kk_word_share  # noqa: F401  (kk_word_share: re-exported for evals)

# Tests: fn(audio, filename) → transcript (taken as Russian) or {"text", "language"}.
_fake: Callable[[bytes, str], str | dict] | None = None

LANGUAGES = ("ru", "kk", "auto", "hint")
# Auto-detect is trusted only for these; anything else (a short note heard as Tatar or Uzbek) is asked again in ru.
AUTO_ACCEPT = ("ru", "kk")
# verbose_json names the detected language in English; the codes that matter here.
_LANG_NAMES = {"russian": "ru", "kazakh": "kk", "english": "en", "ukrainian": "uk", "turkish": "tr", "uzbek": "uz",
               "kyrgyz": "ky", "tatar": "tt", "bashkir": "ba", "mongolian": "mn"}


def set_fake(fn: Callable[[bytes, str], str | dict] | None) -> None:
    global _fake
    _fake = fn


def choose_language(mode: str | None = None, *, recent_texts: Iterable[str] = (), tg_language: str | None = None) -> str | None:
    """Language for Whisper: "ru" / "kk", or None for auto-detect. `hint` looks at what the client wrote before
    (and the Telegram UI language): a Kazakh-writing client gets "kk", everyone else keeps "ru"."""
    mode = (mode or get_settings().stt_language).strip().lower()
    if mode in ("ru", "kk"):
        return mode
    if mode == "auto":
        return None
    if mode != "hint":
        raise ValueError(f"STT_LANGUAGE must be one of {LANGUAGES}, got {mode!r}")
    if (tg_language or "").lower().startswith("kk"):
        return "kk"
    joined = " ".join(t for t in recent_texts if t)
    return "kk" if joined and is_kazakh(joined) else "ru"


def trace_stt_inputs(inputs: dict) -> dict:
    """The voice note itself stays out of the trace (size and format only); the transcript is the output."""
    audio = inputs.get("audio") or b""
    fmt = Path(inputs.get("filename") or "").suffix.lstrip(".") or "ogg"
    return {"audio": {"bytes": len(audio), "format": fmt}, "language": inputs.get("language") or "auto"}


def _lang_code(name: str | None) -> str | None:
    n = (name or "").strip().lower()
    return _LANG_NAMES.get(n, n) or None


@traceable(run_type="tool", name="speech_to_text", process_inputs=trace_stt_inputs)
async def transcribe_detailed(audio: bytes, filename: str = "voice.ogg", language: str | None = "ru") -> dict:
    """{"text", "language"}: `language` is the requested one, or the one Whisper detected when None was passed."""
    if _fake is not None:
        out = _fake(audio, filename)
        return out if isinstance(out, dict) else {"text": out, "language": language or "ru"}
    s = get_settings()
    if not s.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set — voice input needs it")
    # Groq decides the codec from the extension; Telegram voice notes are OGG/Opus.
    if "." not in filename:
        filename += ".ogg"
    data = {"model": s.stt_model, "temperature": "0",
            # verbose_json only for auto-detect: it carries the detected language
            "response_format": "json" if language else "verbose_json"}
    if language:
        data["language"] = language
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(f"{s.groq_url}/audio/transcriptions",
                         headers={"Authorization": f"Bearer {s.groq_api_key}"},
                         data=data, files={"file": (filename, audio)})
    if r.status_code >= 400:
        raise RuntimeError(f"stt {r.status_code}: {r.text[:200]}")
    body = r.json()
    return {"text": (body.get("text") or "").strip(), "language": language or _lang_code(body.get("language"))}


async def transcribe(audio: bytes, filename: str = "voice.ogg", language: str | None = "ru") -> str:
    return (await transcribe_detailed(audio, filename, language))["text"]


async def recognize(audio: bytes, filename: str = "voice.ogg", language: str | None = "ru") -> dict:
    """What the chat graph calls: `transcribe_detailed`, plus a second call in Russian when auto-detect
    heard a language this club does not speak."""
    res = await transcribe_detailed(audio, filename, language)
    if language is None and res["language"] not in AUTO_ACCEPT:
        res = {**await transcribe_detailed(audio, filename, "ru"), "detected": res["language"]}
    return res
