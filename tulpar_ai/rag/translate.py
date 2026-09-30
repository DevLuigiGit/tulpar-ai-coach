"""English version of a Russian question for cross-lingual multi-query search (RAG_MULTI_QUERY=en).

Jina puts Russian and English into one vector space, but a Russian question still lands closer to Russian
chunks than to the English WHO page that answers it (cross-lingual is the weakest tag on the golden set). Searching
with an English translation as well and fusing both lists by RRF gives the WHO pages a second chance.

The translation is one small call to the cheap `route` chain at temperature 0 with a tiny prompt
(prompts/translate_en.v1.md). Without an LLM — no keys, all providers down — a small ru→en glossary of the
corpus's own terms gives an English keyword query instead. An English question is not translated at all.
Only the masked question leaves the process, as with every other LLM call.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import OrderedDict
from pathlib import Path

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from ..config import get_settings
from ..llm import LLMError, json_call
from ..pii import mask
from ..prompts import active_version, prompt

MEMO = 256
_CYR = re.compile(r"[а-яё]", re.I)
_LAT = re.compile(r"[a-z]", re.I)

# Stem → English terms, for the no-LLM fallback. Stems are matched at the start of a lower-cased word (ё → е).
GLOSSARY: dict[str, str] = {
    "воз": "WHO", "рекоменд": "recommendations", "минут": "minutes", "часов": "hours", "часа": "hours", "недел": "per week",
    "день": "per day", "дня": "per day", "ежеднев": "daily", "сколько": "how much",
    "взросл": "adults", "пожил": "older adults", "пенсионер": "older adults", "мам": "older adults",
    "бабушк": "older adults", "дедушк": "older adults", "ребен": "children", "дет": "children",
    "школьник": "children", "подрост": "adolescents", "беремен": "pregnant pregnancy", "родов": "postpartum",
    "после род": "postpartum", "кормящ": "postpartum", "диабет": "type 2 diabetes", "гипертон": "hypertension",
    "давлен": "blood pressure hypertension", "рак": "cancer", "онкол": "cancer", "вич": "HIV",
    "хронич": "chronic conditions", "инвалид": "disability", "умерен": "moderate-intensity",
    "интенсив": "vigorous-intensity", "высокой интенсив": "vigorous-intensity", "аэроб": "aerobic",
    "кардио": "aerobic", "силов": "muscle-strengthening", "мышц": "muscle-strengthening", "баланс": "balance",
    "равновес": "balance", "падени": "falls", "падать": "falls", "сидяч": "sedentary behaviour",
    "сиден": "sedentary behaviour", "сидет": "sedentary", "компьютер": "sedentary screen time",
    "экран": "screen time", "телевиз": "television", "сон": "sleep", "сна": "sleep", "активн": "physical activity",
    "движен": "physical activity", "двигат": "physical activity", "заним": "physical activity",
    "трениров": "exercise training", "упражнен": "exercise", "ходьб": "walking", "бег": "running",
    "вред": "health risks", "польз": "benefits", "здоров": "health", "смертн": "mortality",
    "лежа на спин": "supine position", "триместр": "trimester", "подряд": "bouts", "10 минут": "10 minutes",
    "засчит": "count", "легк": "light-intensity",
}


def is_russian(text: str) -> bool:
    """More Cyrillic than Latin letters: translate. An English (or empty) query is searched as is."""
    cyr, lat = len(_CYR.findall(text or "")), len(_LAT.findall(text or ""))
    return cyr > 0 and cyr >= lat


def glossary_translate(text: str) -> str:
    """English keywords for the Russian terms of the question, in order of appearance; "" if none is known."""
    t = " " + " ".join(re.findall(r"[0-9a-zа-яё]+", (text or "").lower().replace("ё", "е"))) + " "
    found: list[tuple[int, str]] = []
    for stem, en in GLOSSARY.items():
        m = re.search(r"(?<=\s)" + re.escape(stem.replace("ё", "е")), t)
        if m:
            found.append((m.start(), en))
    numbers = re.findall(r"\b\d+\b", t)
    words = list(dict.fromkeys(w for _, en in sorted(found) for w in en.split()))
    return " ".join(words + [n for n in numbers if n not in words]).strip()


class _DiskMemo:
    """Translations on disk next to the embedding cache (RAG_EMBED_CACHE=all only: evals, not client questions)."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.data: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def put(self, key: str, value: str) -> None:
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=0), encoding="utf-8")
        os.replace(tmp, self.path)


_memo: OrderedDict[str, tuple[str, str]] = OrderedDict()
_disk: dict[str, _DiskMemo] = {}


def _setup_key() -> str:
    s, v = get_settings(), active_version("translate_en")
    return hashlib.sha1(f"{v}|{prompt('translate_en', v)}|{s.route_models}".encode()).hexdigest()[:10]


def _disk_memo() -> _DiskMemo | None:
    s = get_settings()
    if str(s.rag_embed_cache).lower() != "all":
        return None
    path = Path(s.ai_data_dir) / "emb_cache" / f"translations_{_setup_key()}.json"
    if str(path) not in _disk:
        _disk[str(path)] = _DiskMemo(path)
    return _disk[str(path)]


def _only_query(inputs: dict) -> dict:
    return {"query": mask(inputs.get("query") or "")}


@traceable(run_type="chain", name="translate_query_en", process_inputs=_only_query)
async def to_english(query: str) -> str:
    """English search query for a Russian question; "" when the question is not Russian or nothing is known."""
    rt = get_current_run_tree()

    def note(source: str, out: str) -> str:
        if rt is not None:
            rt.metadata.update({"translation_source": source})
        return out

    if not is_russian(query):
        return note("skip", "")
    masked = mask(query)
    key = f"{_setup_key()}|{' '.join(masked.split())}"
    if key in _memo:
        _memo.move_to_end(key)
        return note("memo:" + _memo[key][0], _memo[key][1])
    disk = _disk_memo()
    cached = disk.get(key) if disk is not None else None
    if cached is not None:
        return note("disk", cached)
    try:
        data, _ = await json_call("route", prompt("translate_en"), masked, temperature=0.0, max_tokens=120)
        out = " ".join(str(data.get("query") or "").split())
        source = "llm"
    except LLMError:
        out, source = "", "llm_error"
    if not out or is_russian(out):  # no key, a failed call or an echo of the Russian text
        out, source = glossary_translate(masked), "glossary"
    _memo[key] = (source, out)
    if len(_memo) > MEMO:
        _memo.popitem(last=False)
    if disk is not None and source == "llm":
        disk.put(key, out)
    return note(source, out)


def reset_memo() -> None:
    _memo.clear()
    _disk.clear()
