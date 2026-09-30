"""Second input guard: an LLM moderation label for messages the regex rules (guardrails.py) let through.

    GUARD_LLM=off         no call (default until the A/B in evals/guardrails_layers.py says otherwise)
    GUARD_LLM=suspicious  call only when the text carries a risk signal (`signals`)
    GUARD_LLM=all         call for every text message that reaches the LLM router

The call runs next to the router (`graph/chat.py:route`) and its label wins over the router's intent, so it adds
latency only when it is slower than the router. It can only add a refusal or an escalation, never lift one: a
message the rules already decided never reaches it. It fails open: on a provider error, a timeout or an unknown label
the router decides as before. The model sees the masked text (pii.mask), like the router.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import asdict, dataclass

from langsmith import traceable

from . import guardrails
from .config import get_settings
from .llm import LLMError, json_call
from .pii import mask
from .prompts import prompt

LABELS = ("self_harm", "injection", "pii_exfil", "dangerous_domain", "toxic")
MODES = ("off", "suspicious", "all")

# ── risk signals: cheap hints that the rules may have missed something ──────
# They are topics, not verdicts: each only buys the message one model call. Written from the category definitions
# and from the rules' known blind spots (other languages, obfuscation, meaning without marker words).
_KK_LETTERS = re.compile(r"[әіңғүұқөһ]", re.I)
_LAT = re.compile(r"[a-z]", re.I)
_CYR = re.compile(r"[а-яё]", re.I)
_MIXED_WORD = re.compile(r"(?=\w*[a-z])(?=\w*[а-яё])\w+", re.I)
_INNER_SYMBOL = re.compile(r"[a-zа-яё][0-9@$*_][a-zа-яё]", re.I)

_ABOUT_BOT = re.compile(
    r"инструкц|правил(?!ьн)|ограничени|промпт|настро[йе]к|конфиг|систем\w*\s+(?:сообщ|указ|настр|промпт)|разработчик"
    r"|админ|модератор|\bрол[ьиюе]\b|притвор|представь|вообрази|сыграй|\bты\s+теперь\b|\bтеперь\s+ты\b"
    r"|prompt|instruction|\brules?\b|\bsystem\b|developer|\badmin|\bmode\b|\brole\b|pretend|imagine|act\s+as"
    r"|you\s+are|you'?re|from\s+now|override|jailbreak"
    r"|ереже|нұсқау|шектеу|\bенді\b", re.I)
_PEOPLE = re.compile(
    r"тренер|клиент|сотрудни|администрат|девушк|парн[яюе]|мужчин|женщин|\bдруг[ауео]?\b|подруг|сосед|участни"
    r"|coach|trainer|member|client|user|someone|\bguy\b|\bgirl\b|жаттықтырушы|бапкер", re.I)
_DATA_OR_ASK = re.compile(
    r"телефон|номер|адрес|жив[её]т|контакт|почт|e-?mail|инст|телеграм|whats\s*app|ватсап|вотсап|соцсет|фамили|рождени"
    r"|\bвес\b|весит|замер|дневник|анализ|диагноз"
    r"|\b(?:дай|дайте|скинь|скиньте|покажи|покажите|пришли|пришлите|напиши|напишите|скажи|скажите|найди|узнай)\b"
    r"|phone|number|address|\blives?\b|contact|instagram|weight|\b(?:send|give|show|tell|find)\b"
    r"|мекенжай|нөмір|\bайт|\bбер(?:ші|іңіз)|көрсет|жібер", re.I)
_SUBSTANCE = re.compile(
    r"доз[аыуе]|\bмг\b|мкг|таблет|капсул|укол|колоть|колю|инъекц|шприц|препарат|лекарств|гормон|пептид|стимулятор"
    r"|жиросжигат|\bкурс|\bцикл|схем\w*\s+при[её]м|мочегон|слабит|рвот|голод|\bне\s+(?:ем|есть|ела|ел|кушаю|кушать)\b"
    r"|без\s+еды|только\s+(?:на\s+)?вод"
    r"|\bdos(?:e|age)|\bmg\b|\bpills?\b|tablet|inject|\bcycle|steroid|\bfast(?:ing)?\b|starv|purg|vomit|laxative|diuretic"
    r"|дәрі|ашығу|\bаш\b|жемей|жемеймін", re.I)
_LIFE_DEATH = re.compile(
    r"умер|умир|смерт|\bжить\b|жизн|исчез|покончи|суицид|\bdie\b|dying|death|\bdead\b|suicid|\bkill|\blife\b|alive"
    r"|өмір|\bөл(?:ім|гім|емін|ген|сем)", re.I)
_SECOND_PERSON = re.compile(
    r"\b(?:ты|тебя|тебе|тобой|твой|твоя|твои|твоё|твое|вы|вас|вам|ваш\w*|бот\w*|you|your|u|сен|сені|саған|сенің|сендер)\b"
    r"|\b\w{3,}(?:ешь|ишь)\b"
    r"|^\s*\w+\s*,|,\s*\w+\s*[!.]*\s*$", re.I)


def _obfuscated(text: str) -> bool:
    low = (text or "").lower()
    return (guardrails._clean(text or "") != (text or "") or bool(_MIXED_WORD.search(low))
            or bool(_INNER_SYMBOL.search(low)) or bool(guardrails._SPACED.search(low)))


def _other_language(text: str) -> bool:
    if _KK_LETTERS.search(text):
        return True
    lat, cyr = len(_LAT.findall(text)), len(_CYR.findall(text))
    return lat > cyr


def signals(text: str) -> tuple[str, ...]:
    """Names of the risk signals in `text` (empty: nothing worth a model call in `suspicious` mode)."""
    t = text or ""
    found = []
    if _obfuscated(t):
        found.append("obfuscation")
    if _other_language(t):
        found.append("language")
    if _ABOUT_BOT.search(t):
        found.append("about_bot")
    if _PEOPLE.search(t) and _DATA_OR_ASK.search(t) or _named_person(t) and _DATA_OR_ASK.search(t):
        found.append("other_person")
    if _SUBSTANCE.search(t):
        found.append("substance")
    if _LIFE_DEATH.search(t):
        found.append("life_death")
    if _SECOND_PERSON.search(t):
        found.append("addressing")
    return tuple(found)


def _named_person(text: str) -> bool:
    """A capitalised word that does not open a sentence: «скинь номер Асель»."""
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = re.findall(r"[\w'-]+", sentence)
        if any(w[:1].isupper() and w[1:].islower() and len(w) >= 3 for w in words[1:]):
            return True
    return False


def mode() -> str:
    m = (get_settings().guard_llm or "off").strip().lower()
    return m if m in MODES else "off"


def wanted(text: str) -> tuple[str, ...] | None:
    """Signals to send along with the call, or None when this message gets no guard call."""
    m = mode()
    if m == "off" or not (text or "").strip():
        return None
    sig = signals(text)
    if m == "suspicious" and not sig:
        return None
    return sig


@dataclass(frozen=True)
class GuardResult:
    label: str | None  # one of LABELS, or None: no risk, or the check did not work
    outcome: str  # the label, "none", "error", "timeout" or "invalid" — what the trace and the eval see
    reason: str = ""
    signals: tuple[str, ...] = ()
    model: str = ""
    ms: int = 0


def _trace_inputs(inputs: dict) -> dict:
    return {"text": mask(inputs.get("text") or ""), "signals": list(inputs.get("signals") or ())}


def _trace_outputs(out) -> dict:
    return asdict(out) if isinstance(out, GuardResult) else {"output": out}


@traceable(run_type="chain", name="guard_llm", process_inputs=_trace_inputs, process_outputs=_trace_outputs)
async def classify(text: str, signals: tuple[str, ...] = ()) -> GuardResult:
    s = get_settings()
    t0 = time.perf_counter()
    ms = lambda: int((time.perf_counter() - t0) * 1000)  # noqa: E731
    try:
        data, res = await asyncio.wait_for(
            json_call("guard", prompt("guard"), mask(text), temperature=0.0, max_tokens=s.guard_max_tokens),
            timeout=s.guard_timeout_s)
    except asyncio.TimeoutError:
        return GuardResult(None, "timeout", signals=signals, ms=ms())
    except LLMError as e:
        return GuardResult(None, "error", reason=str(e)[:200], signals=signals, ms=ms())
    raw = str(data.get("label") or "").strip().lower()
    reason = str(data.get("reason") or "")[:200]
    model = f"{res.provider}:{res.model}"
    if raw == "none":
        return GuardResult(None, "none", reason, signals, model, ms())
    if raw not in LABELS:
        return GuardResult(None, "invalid", f"label={raw[:40]!r}", signals, model, ms())
    return GuardResult(raw, raw, reason, signals, model, ms())


def applied_label(res: GuardResult, flags: dict) -> str | None:
    """The label the graph acts on. Rudeness next to a pain marker is dropped, as in `guardrails.check_input`:
    a client in pain who swears still needs the router and the trainer, not a lecture."""
    if res.label == "toxic" and (flags.get("soft") or flags.get("hard")):
        return None
    return res.label
