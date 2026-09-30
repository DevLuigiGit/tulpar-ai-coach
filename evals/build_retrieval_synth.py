"""Build the SYNTHETIC retrieval set evals/golden/retrieval_synth.jsonl (+ .meta.json) from sampled corpus chunks.

    python evals/build_retrieval_synth.py --plan-only          # sampling plan, no LLM calls
    python evals/build_retrieval_synth.py [--limit 10]         # generate → check → verify → write
    python evals/build_retrieval_synth.py --refresh-stats      # recompute bias statistics of the set, no LLM
    python evals/build_retrieval_synth.py --audit evals/results/retrieval_baseline.json   # label completeness

1. Stratified sample of production chunks (load_chunks(pdf_chunk=400)), fixed seed:
   exercises  60 cards, allocated to muscle groups by share (largest remainder, at least 1 per group);
   nutrition  every chunk of nutrition.md except the HTML source note; there are only 7, so a chunk gets 1–5
              questions by length (20 in all), each about a different fact;
   who2020    120 contiguous strata of the WHO chunks in page order (so every part of the PDF is covered in
              proportion to its size), one question per stratum: its chunks are tried in a seeded order until
              one question passes, a stratum of methods text only can stay empty. Front matter (PDF pages 1–7: cover, licence,
              contents, acknowledgements) and bibliography / member-list chunks (a pattern check) are left out;
              the generator skips what is left of that kind.
   An exercise card gets an aspect (technique / benefit / contraindications, only aspects the card has) and a style
   (named: the client names the exercise; indirect: describes it); a WHO chunk gets a style (cites-who or not).
2. Generator (deepseek-v4.1-flash) writes ONE realistic Russian client question answerable from the chunk (Russian
   for the English WHO PDF too: the cross-lingual case) plus a verbatim evidence quote, or skips the chunk.
3. Code checks: Russian, 4–40 words, not a duplicate, no run of ≥ 5 words copied from a Russian chunk (the
   exercise name is not counted).
4. Verifier (kimi-k2.6) must say answerable by this chunk AND specific (not generic); otherwise the question is
   dropped. A dropped exercise/WHO question is replaced by the next candidate of the same stratum.
5. Relevant = the source chunk plus near-duplicates, one group each (see `relevant_groups`).
Every row has "synthetic": true. LLM replies are cached in AI_DATA_DIR/synth_llm_cache.jsonl, so a re-run with the
same seed makes no new calls.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.retrieval_eval import SPAN_MIN, norm_text, span_coverage, words  # noqa: E402

OUT = ROOT / "evals" / "golden" / "retrieval_synth.jsonl"
META = ROOT / "evals" / "golden" / "retrieval_synth.meta.json"
SEED = 20260929
GEN_MODEL = "deepseek-v4.1-flash"
VERIFY_MODEL = "kimi-k2.6"
GEN_TEMPERATURE = 0.7
MAX_TRIES = 4  # a muscle group may replace max(MAX_TRIES - 1, quota / 2) dropped cards
COPY_RUN_MAX = 4  # a question may repeat at most 4 consecutive words of a Russian chunk
EVIDENCE_SAME_PAGE_MIN = 30  # chars of normalised evidence for the «answer sits in the overlap» rule
EVIDENCE_REPEAT_MIN = 60  # chars for the «same quote on another page» rule
NEAR_DUP_MIN = 0.8  # word 5-gram containment for the near-duplicate rule
LLM_WAITS = (5.0, 15.0, 40.0)

ASPECTS = {"technique": ("Техника:", "техника выполнения: положение тела, движение, амплитуда, дыхание, время удержания"),
           "benefit": ("Польза:", "польза: что развивает упражнение, зачем его делать"),
           "contraindications": ("Противопоказания:", "противопоказания и ограничения: кому осторожно, что делать при проблеме")}
STYLES = {"named": "клиент называет упражнение (можно разговорно или сокращённо)",
          "indirect": "клиент НЕ называет упражнение (не знает названия), а описывает его своими словами: снаряд, "
                      "положение тела, движение или цель"}
WHO_STYLES = {"cites-who": "Клиент ссылается на ВОЗ («по рекомендациям ВОЗ», «что говорит ВОЗ»).",
              "no-source": "Не упоминай ВОЗ и рекомендации: клиент просто спрашивает о своей ситуации."}

GEN_SYSTEM = """Ты готовишь проверочный набор для поиска по базе знаний фитнес-клуба. Тебе дают один фрагмент базы. \
Придумай вопрос, который реальный клиент клуба мог бы написать тренеру в чат и ответ на который есть в этом фрагменте.

Требования к вопросу:
- по-русски, 6–25 слов, живым разговорным языком, как пишут в мессенджере;
- конкретный: про факт, число, рекомендацию, технику, ограничение или пользу из фрагмента, а не общий («как похудеть?», «полезен ли спорт?»);
- ответ следует из фрагмента;
- своими словами: не переписывай фразы фрагмента и не повторяй подряд больше трёх его слов (название упражнения повторять можно);
- без упоминания фрагмента, текста, документа, страниц, таблиц и ссылок на исследования.

Если во фрагменте нет ничего, о чём спросил бы клиент (список литературы, авторы и участники, выходные данные, \
оглавление, описание методики составления рекомендаций без самих выводов), верни {"skip": true, "reason": "..."}."""

GEN_ONE = """Верни JSON: {"skip": false, "question": "...", "evidence": "..."}, где evidence — дословная цитата из \
фрагмента (одно-два предложения без изменений), в которой есть ответ."""

VERIFY_SYSTEM = """Ты проверяешь вопросы для набора оценки поиска по базе знаний фитнес-клуба. Даны фрагмент базы \
(может быть на английском) и вопрос клиента на русском. Две проверки:
1. answerable — во фрагменте есть информация, из которой можно по существу ответить на этот вопрос без внешних \
знаний. Если фрагмент отвечает лишь частично или на другой, похожий вопрос — false.
2. specific — вопрос конкретный, и его естественно искать именно в таком фрагменте. Общий вопрос, к которому \
подходит почти любой текст о спорте или здоровье («полезны ли тренировки?», «как быть здоровым?»), — false.
Верни JSON: {"answerable": true|false, "specific": true|false, "reason": "одно короткое предложение"}"""


# ── sampling ─────────────────────────────────────────────────────────────────
_AUTH = re.compile(r"\b[A-Z][A-Za-z'’-]+ [A-Z]{1,3}(?:,|\.)")
_TITLES = re.compile(r"\b(?:Dr|Professor|Prof|Director|Department|University|Institute|Ministry)\b")


def not_client_content(text: str) -> bool:
    """Bibliography or a list of people/affiliations: nothing a client would ask about."""
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    refs = len(_AUTH.findall(text)) >= 3 or (years >= 2 and ("et al" in text or "doi" in text.lower() or "http" in text))
    return refs or len(_TITLES.findall(text)) >= 3


def largest_remainder(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Allocate `total` across groups by size, every group at least 1 (when total allows)."""
    n = sum(sizes.values())
    base = {k: max(1, math.floor(total * v / n)) for k, v in sizes.items()}
    rest = total - sum(base.values())
    order = sorted(sizes, key=lambda k: (-(total * sizes[k] / n - math.floor(total * sizes[k] / n)), k))
    for k in order[:max(0, rest)]:
        base[k] += 1
    return base


def who_band(page: int | None) -> str:
    """Part of the WHO PDF by its contents page (printed page 1 = PDF page 10)."""
    if page is None or page <= 7:
        return "who-front"  # cover, contents, acknowledgements, abbreviations
    if page <= 23:
        return "who-summary"  # glossary, executive summary: every recommendation «at a glance»
    if page <= 32:
        return "who-methods"  # background, methods
    if page <= 74:
        return "who-evidence"  # recommendations by population with supporting evidence and rationale
    return "who-back"  # evidence to recommendations, research needs, implementation, references, annexes


def card_aspects(text: str) -> list[str]:
    return [a for a, (marker, _) in ASPECTS.items() if marker in text]


@dataclass
class Stratum:
    key: str
    kind: str
    quota: int
    candidates: list[dict] = field(default_factory=list)  # {"chunk": Chunk, "aspect"?, "style", "n"?}


def sample_plan(chunks: list, seed: int = SEED, n_ex: int = 60, n_who: int = 120, n_nut: int = 20) -> list[Stratum]:
    rng = random.Random(seed)
    strata: list[Stratum] = []
    ex = [c for c in chunks if c.source == "exercises"]
    groups: dict[str, list] = defaultdict(list)
    for c in ex:
        groups[c.muscle_group or "?"].append(c)
    for g, q in sorted(largest_remainder({g: len(v) for g, v in groups.items()}, n_ex).items()):
        pool = sorted(groups[g], key=lambda c: c.id)
        rng.shuffle(pool)
        cands = []
        for c in pool:
            asp = card_aspects(c.text) or ["technique"]
            cands.append({"chunk": c, "aspect": rng.choice(asp), "style": "named" if rng.random() < 2 / 3 else "indirect"})
        strata.append(Stratum(f"ex-{g}", "exercises", q, cands))
    nut = [c for c in chunks if c.source == "nutrition" and not c.text.lstrip().startswith("<!--")]  # markup note
    if nut:
        weights = {c.id: len(c.text) for c in nut}
        alloc = {k: min(5, v) for k, v in largest_remainder(weights, n_nut).items()}
        for c in nut:
            strata.append(Stratum(f"nut-{c.id}", "nutrition", alloc[c.id], [{"chunk": c, "style": "rules", "n": alloc[c.id]}]))
    who = sorted((c for c in chunks if c.source == "who2020" and who_band(c.page) != "who-front"
                  and not not_client_content(c.text)),
                 key=lambda c: (c.page or 0, c.chunk_index))
    if who and n_who:
        bounds = [round(i * len(who) / n_who) for i in range(n_who + 1)]
        for i in range(n_who):
            pool = who[bounds[i]:bounds[i + 1]]
            rng.shuffle(pool)
            cands = [{"chunk": c, "style": "cites-who" if rng.random() < 0.5 else "no-source"} for c in pool]
            strata.append(Stratum(f"who-{i:03d}", "who", 1, cands))
    return strata


# ── checks ───────────────────────────────────────────────────────────────────
def cyrillic_share(s: str) -> float:
    letters = [ch for ch in s if ch.isalpha()]
    return sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in letters) / (len(letters) or 1)


def copy_run(question: str, text: str, ignore: str = "") -> int:
    """Longest run of consecutive question words that also appears consecutively in `text` (minus `ignore` words)."""
    q = words(question)
    t = words(text)
    skip = set(words(ignore))
    tj = " " + " ".join(t) + " "
    best = 0
    for i in range(len(q)):
        for j in range(i + 1, len(q) + 1):
            run = q[i:j]
            if " " + " ".join(run) + " " not in tj:
                break
            best = max(best, sum(1 for w in run if w not in skip))
    return best


def _stems(s: str) -> set[str]:
    return {w[:5] for w in words(s) if len(w) >= 4 and not w.isdigit()}


def lex_overlap(question: str, text: str) -> float | None:
    """Share of the question's content-word stems (first 5 letters, words of 4+ letters) present in `text`."""
    q = _stems(question)
    return round(len(q & _stems(text)) / len(q), 3) if q else None


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def containment(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def locate_evidence(evidence: str, text: str) -> tuple[str | None, bool]:
    """(evidence, verbatim?) — a quote the chunk does not contain (≥ 80% of 3-grams) is not used."""
    if not evidence:
        return None, False
    if norm_text(evidence) in norm_text(text):
        return evidence, True
    return (evidence, False) if span_coverage(evidence, text) >= 0.8 else (None, False)


def _page_key(c) -> tuple:
    return (c.source, c.page if c.page is not None else c.title)


def relevant_groups(src, evidence: str | None, chunks: list) -> list[dict]:
    """Relevance rule of the synthetic set; one group (interchangeable answer) per chunk:

      source          the chunk the question was generated from;
      evidence-overlap a chunk of the same page (same section for nutrition) that contains the whole evidence
                      quote (≥ 30 chars) — the quote sits in the 120-char sliding-window overlap;
      evidence-repeat a chunk of the same source on another page that contains the whole quote (≥ 60 chars) —
                      e.g. the WHO «at a glance» boxes repeated in the main text;
      near-duplicate  a chunk of the same source sharing ≥ 80% of the word 5-grams of the shorter of the two.
    Each group keeps id + source + page (+ title) + `text` (the quote, or the chunk text if it lacks the quote),
    so a config with another chunking can still resolve it by source/page/text coverage.
    """
    ev = norm_text(evidence) if evidence else ""

    def group(c, why: str) -> dict:
        g = {"id": c.id, "source": c.source, "page": c.page}
        if c.source != "who2020":
            g["title"] = c.title
        g["text"] = evidence if evidence and span_coverage(evidence, c.text) >= SPAN_MIN else c.text
        g["why"] = why
        return g

    out = [group(src, "source")]
    for c in chunks:
        if c.id == src.id or c.source != src.source:
            continue
        t = norm_text(c.text)
        if ev and len(ev) >= EVIDENCE_SAME_PAGE_MIN and _page_key(c) == _page_key(src) and ev in t:
            out.append(group(c, "evidence-overlap"))
        elif ev and len(ev) >= EVIDENCE_REPEAT_MIN and ev in t:
            out.append(group(c, "evidence-repeat"))
        elif containment(src.text, c.text) >= NEAR_DUP_MIN:
            out.append(group(c, "near-duplicate"))
    return out


# ── LLM ──────────────────────────────────────────────────────────────────────
class LLMCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.data[d["key"]] = d
        self.calls = 0
        self.used: set[str] = set()

    @staticmethod
    def key(model: str, system: str, user: str, temperature: float) -> str:
        return hashlib.sha1(json.dumps([model, system, user, temperature], ensure_ascii=False).encode()).hexdigest()

    def put(self, key: str, rec: dict) -> None:
        self.data[key] = {"key": key, **rec}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.data[key], ensure_ascii=False) + "\n")


async def llm_json(cache: LLMCache, sem: asyncio.Semaphore, model: str, system: str, user: str,
                   temperature: float, max_tokens: int) -> dict:
    from tulpar_ai import llm

    k = LLMCache.key(model, system, user, temperature)
    cache.used.add(k)
    if k in cache.data:
        return cache.data[k]["data"]
    err: Exception | None = None
    for attempt in range(len(LLM_WAITS) + 1):
        try:
            async with sem:
                cache.calls += 1
                res = await llm.chat("judge", system, user, json_mode=True, temperature=temperature,
                                     max_tokens=max_tokens, models=[("ollama", model)])
            if not isinstance(res.data, dict):
                raise llm.LLMError(f"expected a JSON object, got {type(res.data).__name__}")
            cache.put(k, {"model": model, "data": res.data, "in": res.input_tokens, "out": res.output_tokens})
            return res.data
        except (llm.LLMError, ValueError) as e:
            err = e
            if attempt < len(LLM_WAITS):
                await asyncio.sleep(LLM_WAITS[attempt])
    raise llm.LLMError(f"{model}: {err}")


def gen_user(cand: dict) -> str:
    c = cand["chunk"]
    if c.source == "exercises":
        head = (f"Фрагмент — карточка упражнения из каталога клуба.\nАспект вопроса: {ASPECTS[cand['aspect']][1]}.\n"
                f"Стиль: {STYLES[cand['style']]}.\n{GEN_ONE}")
    elif c.source == "nutrition":
        head = (f"Фрагмент — правила питания Tulpar, по которым приложение считает дневную норму.\n"
                f"Напиши {cand['n']} разных вопроса(ов), каждый про свой факт из фрагмента.\n"
                'Верни JSON: {"skip": false, "questions": [{"question": "...", "evidence": "дословная цитата с ответом"}]}.')
    else:
        head = ("Фрагмент — из рекомендаций ВОЗ по физической активности и сидячему образу жизни (2020), на английском. "
                "Вопрос всё равно пиши по-русски, а evidence — дословно по-английски, как во фрагменте.\n"
                f"{WHO_STYLES[cand['style']]}\n{GEN_ONE}")
    return f"{head}\n\nФрагмент:\n<<<\n{c.text}\n>>>"


def verify_user(chunk_text: str, question: str) -> str:
    return f"Фрагмент:\n<<<\n{chunk_text}\n>>>\n\nВопрос клиента: {question}"


# ── build ────────────────────────────────────────────────────────────────────
@dataclass
class Stats:
    tried: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)
    drops: Counter = field(default_factory=Counter)
    drops_by_kind: dict = field(default_factory=lambda: defaultdict(Counter))

    def drop(self, kind: str, reason: str) -> None:
        self.drops[reason] += 1
        self.drops_by_kind[kind][reason] += 1


def names_exercise(question: str, title: str) -> bool:
    """≥ 60% of the title's word stems (first 5 letters of words with 3+ letters) occur in the question."""
    stem = lambda s: {w[:5] for w in words(s) if len(w) >= 3}  # noqa: E731
    t = stem(title)
    return bool(t) and len(t & stem(question)) / len(t) >= 0.6


def tags_for(kind: str, cand: dict, question: str) -> list[str]:
    """Style tags describe the question as written, not the style asked for: the generator does not always obey."""
    c = cand["chunk"]
    if kind == "exercises":
        return ["exercises", cand["aspect"], "named" if names_exercise(question, c.title) else "indirect"]
    if kind == "nutrition":
        return ["nutrition"]
    return ["who", "cross-lingual", who_band(c.page), "cites-who" if "воз" in words(question) else "no-source"]


async def process(stratum: Stratum, chunks: list, cache: LLMCache, sem: asyncio.Semaphore, stats: Stats,
                  seen: set[str]) -> list[dict]:
    rows: list[dict] = []
    # nutrition: one call per chunk; WHO: every chunk of the stratum until one question passes (methods and
    # implementation text is mostly skipped by the generator); a muscle group: quota + max(3, quota/2) cards
    if stratum.kind == "nutrition":
        tries = 1
    elif stratum.kind == "who":
        tries = len(stratum.candidates)
    else:
        tries = stratum.quota + max(MAX_TRIES - 1, math.ceil(stratum.quota / 2))
    for cand in stratum.candidates[:tries]:
        if len(rows) >= stratum.quota:
            break
        c = cand["chunk"]
        stats.tried[stratum.kind] += 1
        try:
            g = await llm_json(cache, sem, GEN_MODEL, GEN_SYSTEM, gen_user(cand), GEN_TEMPERATURE, 700)
        except Exception:  # noqa: BLE001
            stats.drop(stratum.kind, "gen_error")
            continue
        if g.get("skip"):
            stats.skipped[stratum.kind] += 1
            continue
        items = g.get("questions") if stratum.kind == "nutrition" else [g]
        for it in (items or [])[: stratum.quota - len(rows)]:
            q = str((it or {}).get("question") or "").strip()
            reason = None
            if cyrillic_share(q) < 0.6:
                reason = "not_russian"
            elif not 4 <= len(q.split()) <= 40:
                reason = "length"
            elif norm_text(q) in seen:
                reason = "duplicate"
            elif c.source != "who2020" and copy_run(q, c.text, ignore=c.title if c.source == "exercises" else "") > COPY_RUN_MAX:
                reason = "verbatim_copy"
            if reason:
                stats.drop(stratum.kind, reason)
                continue
            try:
                v = await llm_json(cache, sem, VERIFY_MODEL, VERIFY_SYSTEM, verify_user(c.text, q), 0.0, 200)
            except Exception:  # noqa: BLE001
                stats.drop(stratum.kind, "verify_error")
                continue
            if v.get("answerable") is not True:
                stats.drop(stratum.kind, "verify_not_answerable")
                continue
            if v.get("specific") is not True:
                stats.drop(stratum.kind, "verify_not_specific")
                continue
            seen.add(norm_text(q))
            evidence, verbatim = locate_evidence(str(it.get("evidence") or ""), c.text)
            rows.append({
                "question": q, "relevant": relevant_groups(c, evidence, chunks), "tags": tags_for(stratum.kind, cand, q),
                "synthetic": True, "source_chunk": c.id, "source": c.source, "page": c.page, "title": c.title,
                "evidence": evidence, "evidence_verbatim": verbatim, "stratum": stratum.key, "style_requested": cand["style"],
                "gen_model": f"ollama:{GEN_MODEL}", "verify_model": f"ollama:{VERIFY_MODEL}",
                "verify_reason": str(v.get("reason") or "")[:200],
                "lex_overlap": lex_overlap(q, c.text) if c.source != "who2020" else None,
                "copy_run": copy_run(q, c.text, ignore=c.title if c.source == "exercises" else "")})
    return rows


async def build(chunks: list, strata: list[Stratum], cache: LLMCache, concurrency: int = 4) -> tuple[list[dict], Stats]:
    sem = asyncio.Semaphore(concurrency)
    stats = Stats()
    seen: set[str] = set()
    per = await asyncio.gather(*(process(s, chunks, cache, sem, stats, seen) for s in strata))
    rows, kept = [], set()
    for r in (r for rs in per for r in rs):  # strata run concurrently: a race past the `seen` check ends here
        if norm_text(r["question"]) in kept:
            stats.drop("who" if r["source"] == "who2020" else r["source"], "duplicate")
            continue
        kept.add(norm_text(r["question"]))
        rows.append(r)
    order = {"exercises": 0, "nutrition": 1, "who2020": 2}
    rows.sort(key=lambda r: (order[r["source"]], r["stratum"], r["source_chunk"]))
    for i, r in enumerate(rows, 1):
        r["id"] = f"s{i:03d}"
    return [{"id": r.pop("id"), **r} for r in rows], stats


def _numbers(s: str) -> set[str]:
    return set(re.findall(r"\d+", s or ""))


def _latin(s: str) -> set[str]:
    return set(re.findall(r"[a-z]{2,}", norm_text(s)))


def _share(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 3) if flags else None


def lexical_stats(rows: list[dict], chunks: list) -> dict:
    """How much of its source a question gives away, next to the same numbers for the hand-written qa.jsonl.

    Russian sources: share of the question's content-word stems (first 5 letters, the lexical reranker's stems)
    found in the chunk. WHO (English) sources: share of questions that repeat a number of the chunk, and share
    with a Latin-script term (MET, BMI...) — both are lexical anchors even across languages.
    """
    from evals.retrieval_eval import load_jsonl

    by_id = {c.id: c for c in chunks}
    who = [r for r in rows if r["source"] == "who2020"]
    out = {"synth_exercises": _mean([r["lex_overlap"] for r in rows if r["source"] == "exercises"]),
           "synth_nutrition": _mean([r["lex_overlap"] for r in rows if r["source"] == "nutrition"]),
           "synth_who_n": len(who),
           "synth_who_number_shared": _share([bool(_numbers(r["question"]) & _numbers(by_id[r["source_chunk"]].text))
                                              for r in who if r["source_chunk"] in by_id]),
           "synth_who_latin_term": _share([bool(_latin(r["question"])) for r in who])}
    cards = {c.title: c.text for c in chunks if c.source == "exercises"}
    nut = [c.text for c in chunks if c.source == "nutrition"]
    pages: dict[int, str] = defaultdict(str)
    for c in chunks:
        if c.source == "who2020":
            pages[c.page] += " " + c.text
    ex_vals, nut_vals, who_num, who_lat = [], [], [], []
    for r in load_jsonl(ROOT / "evals" / "golden" / "qa.jsonl"):
        srcs = r.get("expected_sources", [])
        for e in srcs:
            if e["source"] == "exercises" and e.get("title") in cards:
                ex_vals.append(lex_overlap(r["question"], cards[e["title"]]))
            elif e["source"] == "nutrition":
                nut_vals.append(max((lex_overlap(r["question"], t) or 0) for t in nut))
        cited = " ".join(pages.get(e.get("page"), "") for e in srcs if e["source"] == "who2020")
        if cited:
            who_num.append(bool(_numbers(r["question"]) & _numbers(cited)))
            who_lat.append(bool(_latin(r["question"])))
    ex_vals = [x for x in ex_vals if x is not None]
    out.update({"qa_exercises": _mean(ex_vals), "qa_exercises_n": len(ex_vals), "qa_nutrition": _mean(nut_vals),
                "qa_nutrition_n": len(nut_vals), "qa_who_n": len(who_num), "qa_who_number_shared": _share(who_num),
                "qa_who_latin_term": _share(who_lat)})
    return out


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def meta(rows: list[dict], strata: list[Stratum], stats: Stats, chunks: list, seed: int, cache: LLMCache) -> dict:
    from parsing.chunker import PARSING_VERSION
    from tulpar_ai.rag.index import corpus_fingerprint

    kinds = Counter(r["source"] for r in rows)
    why = Counter(g["why"] for r in rows for g in r["relevant"])
    return {
        "synthetic": True, "created": time.strftime("%Y-%m-%d %H:%M"), "seed": seed, "rows": len(rows),
        "gen_model": f"ollama:{GEN_MODEL}", "gen_temperature": GEN_TEMPERATURE, "verify_model": f"ollama:{VERIFY_MODEL}",
        "chunking": {"pdf_chunk": 400, "pdf_overlap": 120, "parsing_version": PARSING_VERSION,
                     "corpus_fingerprint": corpus_fingerprint(), "chunks": len(chunks)},
        "plan": {"strata": Counter(s.kind for s in strata), "quota": {k: sum(s.quota for s in strata if s.kind == k)
                                                                       for k in ("exercises", "nutrition", "who")}},
        "rows_by_source": dict(kinds), "candidates_tried": dict(stats.tried), "skipped_by_generator": dict(stats.skipped),
        "dropped": sum(stats.drops.values()), "drops": dict(stats.drops),
        "drops_by_kind": {k: dict(v) for k, v in stats.drops_by_kind.items()},
        "relevance_groups_by_rule": dict(why),
        "rows_with_extra_relevant": sum(1 for r in rows if len(r["relevant"]) > 1),
        "evidence_verbatim": sum(1 for r in rows if r["evidence_verbatim"]),
        "evidence_missing": sum(1 for r in rows if not r["evidence"]),
        "lexical_overlap": lexical_stats(rows, chunks),
        "llm_responses_used": len(cache.used),
        "llm_responses_used_by_model": dict(Counter(cache.data[k]["model"] for k in cache.used if k in cache.data)),
        "llm_calls_this_run": cache.calls,
        "prompts_sha1": hashlib.sha1((GEN_SYSTEM + GEN_ONE + VERIFY_SYSTEM).encode()).hexdigest()[:12],
    }


async def audit(result: dict, gold: list[dict], chunks: list, cache: LLMCache, *, config: str | None = None,
                k: int = 4, concurrency: int = 4) -> dict:
    """How incomplete are single-chunk labels? For every synthetic question the config missed at k, the verifier
    judges each of its top-k chunks with the same «answerable» check. The golden set is NOT changed: counting
    these chunks as relevant would favour the audited config (pooling bias); the result is a sensitivity bound."""
    name = config or next(iter(result["configs"]))
    rows = result["configs"][name]["sets"]["synth"]["rows"]
    questions = {r["id"]: r["question"] for r in gold}
    by_id = {c.id: c for c in chunks}
    sem = asyncio.Semaphore(concurrency)
    misses = [r for r in rows if not r["rank"] or r["rank"] > k]

    async def judge(r: dict) -> dict:
        found = []
        for cid in r["top"][:k]:
            v = await llm_json(cache, sem, VERIFY_MODEL, VERIFY_SYSTEM, verify_user(by_id[cid].text, questions[r["id"]]),
                               0.0, 200)
            if v.get("answerable") is True:
                found.append(cid)
        return {"id": r["id"], "tags": r["tags"], "answering_chunks": found}

    judged = await asyncio.gather(*(judge(r) for r in misses))
    hits = {r["id"] for r in rows if r["rank"] and r["rank"] <= k}
    rescued = {j["id"] for j in judged if j["answering_chunks"]}

    def part(sub: list[dict]) -> dict:
        n = len(sub)
        h = sum(1 for r in sub if r["id"] in hits)
        x = sum(1 for r in sub if r["id"] in rescued)
        return {"n": n, "misses": n - h, "misses_answered_by_retrieved": x, f"hit@{k}": round(h / n, 4) if n else None,
                f"hit@{k}_if_counted": round((h + x) / n, 4) if n else None}

    tags = sorted({t for r in rows for t in r["tags"]})
    return {"synthetic": True, "config": name, "k": k, "judge_model": f"ollama:{VERIFY_MODEL}", **part(rows),
            "by_tag": {t: part([r for r in rows if t in r["tags"]]) for t in tags}, "rows": judged}


async def main_async(args) -> int:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from tulpar_ai.config import get_settings
    from tulpar_ai.rag.index import load_chunks

    get_settings.cache_clear()
    chunks = load_chunks(pdf_chunk=400)
    strata = sample_plan(chunks, args.seed, args.n_ex, args.n_who, args.n_nut)
    if args.limit:
        by_kind: dict[str, list] = defaultdict(list)
        for s in strata:
            by_kind[s.kind].append(s)
        strata = [s for k in by_kind for s in by_kind[k][: args.limit]]
    if args.plan_only:
        plan = [{"key": s.key, "kind": s.kind, "quota": s.quota, "first": s.candidates[0]["chunk"].id,
                 "pool": len(s.candidates)} for s in strata]
        print(json.dumps({"strata": len(plan), "quota": sum(p["quota"] for p in plan), "plan": plan}, ensure_ascii=False, indent=1))
        return 0
    out, meta_path = Path(args.out), Path(args.out).with_suffix(".meta.json")
    if args.audit:
        from evals.retrieval_eval import load_jsonl

        cache = LLMCache(Path(get_settings().ai_data_dir) / "synth_llm_cache.jsonl")
        res = await audit(json.loads(Path(args.audit).read_text(encoding="utf-8")), load_jsonl(out), chunks, cache,
                          config=args.audit_config, k=args.audit_k, concurrency=args.concurrency)
        res["llm_calls_this_run"] = cache.calls
        Path(args.audit_out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
        return 0
    if args.refresh_stats:  # recompute the statistics of an existing set: no LLM calls, rows untouched
        from evals.retrieval_eval import load_jsonl

        m = json.loads(meta_path.read_text(encoding="utf-8"))
        m["lexical_overlap"] = lexical_stats(load_jsonl(out), chunks)
        meta_path.write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(m["lexical_overlap"], ensure_ascii=False, indent=1))
        return 0
    cache = LLMCache(Path(get_settings().ai_data_dir) / "synth_llm_cache.jsonl")
    rows, stats = await build(chunks, strata, cache, args.concurrency)
    m = meta(rows, strata, stats, chunks, args.seed, cache)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    meta_path.write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(m, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-ex", type=int, default=60)
    ap.add_argument("--n-who", type=int, default=120)
    ap.add_argument("--n-nut", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="first N strata of each kind (a trial run)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--refresh-stats", action="store_true", help="recompute lexical statistics of --out, no LLM")
    ap.add_argument("--audit", help="retrieval_eval result JSON: judge the retrieved chunks of synth misses")
    ap.add_argument("--audit-config")
    ap.add_argument("--audit-k", type=int, default=4)
    ap.add_argument("--audit-out", default=str(ROOT / "evals" / "results" / "retrieval_synth_audit.json"))
    ap.add_argument("--out", default=str(OUT))
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
