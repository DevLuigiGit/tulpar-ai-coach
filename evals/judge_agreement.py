"""How reliable is the LLM judge? Several judges from different model families score the same QA answers.

    python evals/judge_agreement.py                      # score answered rows, write results + human sample
    python evals/judge_agreement.py --stats-only         # recompute stats from the cache, no LLM calls
    python evals/judge_agreement.py --retest ollama:deepseek-v4.1-flash   # + a second pass of one judge
    python evals/judge_agreement.py --human              # human labels in evals/human/*.csv vs every judge

Inputs are rebuilt from a finished QA run (default: the current config, answer model minimax-m3): the stored
answer, the golden question/reference and the same top-4 sources, re-retrieved from the Jina index in
AI_DATA_DIR/qdrant (no re-embedding of the corpus). Judge prompts and the user message are exactly those of
evals/run.py. Results: evals/results/judge_agreement.json; raw calls are cached in AI_DATA_DIR so a crash or
a 429 storm never pays twice. A failed call (rate limit, timeout, empty reply) stays in the cache as a failure
and is asked again by the next scoring run; only a hard refusal (retired model, bad request) is final.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import itertools
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as ev  # noqa: E402  evals/run.py: golden loader, hit rank, settings boot

RESULTS = ROOT / "evals" / "results"
HUMAN_DIR = ROOT / "evals" / "human"
DEFAULT_ANSWERS = RESULTS / "qa_text_minimax-m3.json"
# The configured chain first (its primary may be gone: it is probed and reported), then other model families.
DEFAULT_JUDGES = "ollama:qwen3.5:397b,groq:openai/gpt-oss-120b,ollama:kimi-k2.6,ollama:deepseek-v4.1-flash"
STORED = "stored"  # scores written by the original QA run (config judge chain at that time)
PASS = 4
SCALE = [1, 2, 3, 4, 5]
METRICS = ("faithfulness", "correctness")
HUMAN_COLUMN = "human_faithfulness"


# ── agreement math (pure, unit-tested) ───────────────────────────────────────
def _pairs(a: list, b: list) -> list[tuple]:
    return [(x, y) for x, y in zip(a, b) if x is not None and y is not None]


def cohen_kappa(a: list, b: list, labels: list | None = None, weights: str | None = None) -> float | None:
    """Cohen's kappa; weights None | "linear" | "quadratic". None when chance agreement is total (one class only)."""
    pairs = _pairs(a, b)
    if not pairs:
        return None
    labels = sorted(set(labels or []) | {v for p in pairs for v in p})
    k, idx, n = len(labels), {v: i for i, v in enumerate(labels)}, len(pairs)
    obs = [[0] * k for _ in range(k)]
    for x, y in pairs:
        obs[idx[x]][idx[y]] += 1
    rows = [sum(r) for r in obs]
    cols = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:
        if weights is None or k == 1:
            return 0.0 if i == j else 1.0
        d = abs(i - j) / (k - 1)
        return d if weights == "linear" else d * d

    disagree_obs = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    disagree_exp = sum(w(i, j) * rows[i] * cols[j] for i in range(k) for j in range(k)) / (n * n)
    return None if disagree_exp == 0 else 1 - disagree_obs / disagree_exp


def fleiss_kappa(items: list[list]) -> float | None:
    """Fleiss' kappa for items rated by the same number of raters (each item: the list of its ratings)."""
    items = [it for it in items if it and all(v is not None for v in it)]
    if not items or len(items[0]) < 2:  # one rater (e.g. a run with a single live judge): no agreement to measure
        return None
    m = len(items[0])
    cats = sorted({v for it in items for v in it})
    p_items = [(sum(it.count(c) ** 2 for c in cats) - m) / (m * (m - 1)) for it in items]
    p_cats = [sum(it.count(c) for it in items) / (len(items) * m) for c in cats]
    p_bar, p_exp = statistics.mean(p_items), sum(p * p for p in p_cats)
    return None if p_exp == 1 else (p_bar - p_exp) / (1 - p_exp)


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1  # ties share the average rank
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    mx, my = statistics.mean(x), statistics.mean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    var = sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    return None if var == 0 else cov / var ** 0.5


def spearman(x: list, y: list) -> float | None:
    pairs = _pairs(x, y)
    if len(pairs) < 3:
        return None
    return _pearson(_ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs]))


def spearman_perm_p(x: list, y: list, shuffles: int = 5000, seed: int = 0) -> float | None:
    """Two-sided permutation p-value of Spearman's rho, seeded so reruns print the same number.

    Scores are mostly 5 with a few 3–4, so the textbook t-approximation is unreliable here; shuffling the
    lengths against the scores asks directly how often chance alone gives a correlation this strong.
    """
    pairs = _pairs(x, y)
    rho = spearman(x, y)
    if rho is None:
        return None
    rx, ry = _ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs])
    rng, hits = random.Random(seed), 0
    for _ in range(shuffles):
        rng.shuffle(ry)
        hits += abs(_pearson(rx, ry) or 0.0) >= abs(rho) - 1e-12
    return (hits + 1) / (shuffles + 1)


def krippendorff_alpha(units: list[list], level: str = "nominal") -> float | None:
    """Krippendorff's alpha, level "nominal" | "interval".

    Unlike Fleiss' kappa it needs no complete rows: a rating a judge failed to give (None) drops only that
    value, and every row with at least two ratings still counts. So a judge that fails exactly on the
    contested rows cannot hide them from the agreement figure.
    """
    units = [[v for v in u if v is not None] for u in units]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None

    def delta(a, b) -> float:
        return float((a - b) ** 2) if level == "interval" else float(a != b)

    values = [v for u in units for v in u]
    n = len(values)
    d_obs = sum(sum(delta(a, b) for a, b in itertools.permutations(u, 2)) / (len(u) - 1) for u in units) / n
    d_exp = sum(delta(a, b) for a, b in itertools.permutations(values, 2)) / (n * (n - 1))
    return None if d_exp == 0 else 1 - d_obs / d_exp


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(x, nd)


def pair_stats(a: list, b: list, threshold: int = PASS) -> dict:
    """Two raters on a 1–5 scale: exact and pass/fail agreement, kappa on pass/fail, weighted kappa, MAD."""
    pairs = _pairs(a, b)
    if not pairs:
        return {"n": 0}
    pa, pb = [x >= threshold for x, _ in pairs], [y >= threshold for _, y in pairs]
    xs, ys = [x for x, _ in pairs], [y for _, y in pairs]
    return {"n": len(pairs), "exact_pct": ev.pct(x == y for x, y in pairs),
            "pass_agree_pct": ev.pct(x == y for x, y in zip(pa, pb)),
            "kappa_pass": _r(cohen_kappa(pa, pb, labels=[False, True])),
            "kappa_quadratic": _r(cohen_kappa(xs, ys, labels=SCALE, weights="quadratic")),
            "mad": round(statistics.mean(abs(x - y) for x, y in pairs), 3)}


def judge_stats(scores: list, chars: list, tokens: list, cut_at: int = 400) -> dict:
    """Length bias: score vs characters the judge saw, vs full generated tokens, and on answers not cut short.

    Stored answers are cut at 400 chars, so a long answer may lose facts before the judge sees it; the
    `uncut` correlation removes that confound. `pass_pct` is on this judge's own scored rows; compare judges
    with the metric's `pass_pct_common` instead.
    """
    got = [s for s in scores if s is not None]
    uncut = [(s, c) for s, c in zip(scores, chars) if c < cut_at]
    us, uc = [s for s, _ in uncut], [c for _, c in uncut]
    return {"n_scored": len(got), "failed": len(scores) - len(got),
            "mean": round(statistics.mean(got), 2) if got else None,
            "pass_pct": ev.pct(s >= PASS for s in got) if got else None,
            "dist": {str(v): got.count(v) for v in SCALE},
            "spearman_len_chars": _r(spearman(scores, chars)),
            "perm_p_len_chars": _r(spearman_perm_p(scores, chars)),
            "spearman_len_tokens": _r(spearman(scores, tokens)),
            "spearman_len_chars_uncut": _r(spearman(us, uc)),
            "perm_p_len_chars_uncut": _r(spearman_perm_p(us, uc)),
            "n_uncut": sum(s is not None for s in us)}


# ── judge inputs ─────────────────────────────────────────────────────────────
def judge_user(kind: str, **kw) -> str:
    """The judge's user message, byte for byte what evals/run.py:judge sends (a test guards against drift)."""
    if kind == "faithfulness":
        return f"Источники:\n{kw['sources']}\n\nОтвет коуча:\n{kw['answer']}"
    return f"Вопрос: {kw['question']}\nЭталон: {kw['reference']}\n\nОтвет коуча:\n{kw['answer']}"


async def build_inputs(answers_file: Path, data_dir: Path, limit: int = 0) -> list[dict]:
    """Answered rows of a QA run + golden question/reference + the same sources, re-retrieved from the index."""
    from tulpar_ai.graph import chat as g
    from tulpar_ai.rag.index import Index
    from tulpar_ai.rag.retrieve import set_index

    run_rows = json.loads(answers_file.read_text(encoding="utf-8"))["rows"]
    answered = [r for r in run_rows if r.get("answered") and r.get("rep", 0) == 0][: limit or None]
    gold = {c["id"]: c for c in ev.load("qa.jsonl")}
    idx = Index(pdf_chunk=400, path=data_dir / "qdrant")
    if not idx.count():
        raise SystemExit(f"No index {idx.collection} in {data_dir / 'qdrant'}: copy data/evals/qdrant there first.")
    set_index(idx)
    out = []
    try:
        for r in answered:
            c = gold[r["id"]]
            st = await g.retrieve_node({"text": c["question"], "intent": "question"})
            out.append({"id": r["id"], "tags": c.get("tags", []), "question": c["question"],
                        "reference": c.get("reference_answer", ""), "answerable": c["answerable"],
                        "sources": g._sources(st["hits"]), "source_titles": [h["title"] for h in st["hits"]],
                        "sufficient": bool(st.get("sufficient")), "answer": r["answer"],
                        "answer_chars": len(r["answer"]), "out_tokens": r.get("out_tokens"),
                        "rank_stored": r.get("rank"), "rank_now": ev._hit_rank(st["hits"], c["expected_sources"]),
                        "stored": {m: r.get(m) for m in METRICS}})
    finally:
        idx.close()
        set_index(None)
    return out


# ── scoring ──────────────────────────────────────────────────────────────────
def label(spec: str) -> str:
    return spec.split(":", 1)[1].split("/")[-1]


# A hard HTTP refusal (401/404/410 retired, 400 bad request) will not change on a retry; 429 is a rate limit.
_HARD_REFUSAL = re.compile(r"(?:ollama|groq) 4(?!29)\d\d")
# Empty or non-JSON reply, a list instead of an object, a missing or out-of-range score. kimi returns an empty
# reply now and then at temperature 0 and answers the same input on the next call, so these are retried too,
# but only a couple of times: a judge that really cannot produce the format should not cost five calls a row.
_MALFORMED = re.compile(r"no JSON value|JSONDecodeError|expected a JSON object|score out of range|int\(\)|invalid literal")
MALFORMED_RETRIES = 2


def _retry_wait(err: str, attempt: int) -> float | None:
    """Seconds to wait before retrying, or None when a retry cannot help (410 retired, 400) or the budget is spent."""
    if re.search(r"(?:ollama|groq) (?:429|5\d\d)|Timeout|ConnectError|ReadError|RemoteProtocol", err):
        m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", err)
        hinted = (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else 0
        return min(90.0, max(hinted + 1, 10 * 2 ** attempt))
    if _MALFORMED.search(err) and attempt < MALFORMED_RETRIES:
        return 3.0 * (attempt + 1)
    return None


def is_final(res: dict) -> bool:
    """A cached result a later run must not ask again: a score, or a hard refusal that a retry cannot fix.

    Every other failure (rate limit, timeout, empty reply) is kept in the cache so `--stats-only` counts it,
    and the next scoring run asks that row again.
    """
    return res.get("score") is not None or bool(_HARD_REFUSAL.search(res.get("error", "")))


async def judge_once(spec: str, kind: str, user: str, retries: int = 5) -> dict:
    from tulpar_ai.llm import LLMError, json_call
    from tulpar_ai.prompts import prompt

    provider, model = spec.split(":", 1)
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            data, _ = await json_call("judge", prompt(f"judge_{kind}"), user, temperature=0.0, max_tokens=150,
                                      models=[(provider, model)])
            score = int(data.get("score"))
            if score not in SCALE:
                raise ValueError(f"score out of range: {score}")
            return {"score": score, "reason": str(data.get("reason", ""))[:200],
                    "ms": int((time.perf_counter() - t0) * 1000), "attempts": attempt + 1}
        except (LLMError, TypeError, ValueError) as e:
            err = str(e)
            wait = _retry_wait(err, attempt) if attempt < retries else None
            if wait is None:
                return {"score": None, "error": err[-300:], "ms": int((time.perf_counter() - t0) * 1000),
                        "attempts": attempt + 1}
            print(f"  {label(spec)} {kind}: {err[-60:]!r}, retry in {wait:.0f}s", flush=True)
            await asyncio.sleep(wait)
    return {"score": None, "error": "retries exhausted", "attempts": retries + 1}


class Cache:
    """Judge calls keyed by judge, metric, row, pass and a hash of the exact input."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @staticmethod
    def key(spec: str, kind: str, row_id: str, user: str, rep: int = 0) -> str:
        return f"{spec}|{kind}|{row_id}|{rep}|{hashlib.sha1(user.encode()).hexdigest()[:12]}"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=0), encoding="utf-8")


def tasks_for(inputs: list[dict]) -> list[tuple[str, dict, str]]:
    out = []
    for row in inputs:
        out.append(("faithfulness", row, judge_user("faithfulness", sources=row["sources"], answer=row["answer"])))
        if row["answerable"]:
            out.append(("correctness", row, judge_user("correctness", question=row["question"],
                                                       reference=row["reference"], answer=row["answer"])))
    return out


async def probe(spec: str, row_id: str, user: str, cache: Cache) -> str | None:
    """One real call before a full pass: a retired or misconfigured judge is reported, not scored 88 times.

    The probe is the first faithfulness task, so a successful answer goes to the cache and is not paid twice.
    """
    k = Cache.key(spec, "faithfulness", row_id, user)
    if cache.data.get(k, {}).get("score") is not None:
        return None
    res = await judge_once(spec, "faithfulness", user, retries=2)
    if res["score"] is not None:
        cache.data[k] = res
        cache.save()
        return None
    err = res.get("error", "")
    # only a hard HTTP refusal (401/404/410…) disqualifies a judge; one malformed reply is just a failed score
    return err if _HARD_REFUSAL.search(err) else None


async def score_all(specs: list[str], inputs: list[dict], cache: Cache, retest: str | None) -> None:
    """Sequential per provider, providers in parallel: at most 2 concurrent calls on the shared keys."""
    tasks = tasks_for(inputs)
    passes = [(s, 0) for s in specs] + ([(retest, 1)] if retest else [])
    by_provider: dict[str, list[tuple[str, int]]] = {}
    for spec, rep in passes:
        by_provider.setdefault(spec.split(":", 1)[0], []).append((spec, rep))

    async def worker(items: list[tuple[str, int]]) -> None:
        for spec, rep in items:
            done = 0
            for kind, row, user in tasks:
                k = Cache.key(spec, kind, row["id"], user, rep)
                prev = cache.data.get(k)
                if prev is not None and is_final(prev):
                    continue
                res = await judge_once(spec, kind, user)
                res["runs"] = prev.get("runs", 1) + 1 if prev else 1  # scoring runs that asked this row
                cache.data[k] = res
                cache.save()
                done += 1
                if done % 10 == 0:
                    print(f"  {label(spec)} rep{rep}: {done} new calls", flush=True)

    await asyncio.gather(*(worker(v) for v in by_provider.values()))


# ── assembling results ───────────────────────────────────────────────────────
def collect(specs: list[str], inputs: list[dict], cache: Cache, retest: str | None) -> list[dict]:
    rows = []
    for row in inputs:
        scores: dict = {m: {} for m in METRICS}
        reasons: dict = {m: {} for m in METRICS}
        errors: dict = {m: {} for m in METRICS}
        for kind, _, user in tasks_for([row]):
            for spec, rep in [(s, 0) for s in specs] + ([(retest, 1)] if retest else []):
                hit = cache.data.get(Cache.key(spec, kind, row["id"], user, rep), {})
                name = label(spec) + ("#retest" if rep else "")
                scores[kind][name] = hit.get("score")
                if hit.get("reason"):
                    reasons[kind][name] = hit["reason"]
                if hit.get("error"):
                    errors[kind][name] = hit["error"][-120:]
            scores[kind][STORED] = row["stored"].get(kind)
        rows.append({"id": row["id"], "tags": row["tags"], "answer_chars": row["answer_chars"],
                     "out_tokens": row["out_tokens"], "rank_stored": row["rank_stored"], "rank_now": row["rank_now"],
                     "sufficient": row["sufficient"], "source_titles": row["source_titles"],
                     "scores": scores, "reasons": reasons, "errors": errors})
    return rows


def summarize(rows: list[dict], live: list[str], retest: str | None) -> dict:
    names = [*live, *([label(retest) + "#retest"] if retest else []), STORED]
    out: dict = {"n_answered": len(rows), "live_judges": live, "pass_threshold": PASS}
    for m in METRICS:
        mrows = [r for r in rows if any(v is not None for v in r["scores"][m].values())]
        col = {n: [r["scores"][m].get(n) for r in mrows] for n in names}
        chars, toks = [r["answer_chars"] for r in mrows], [r["out_tokens"] for r in mrows]
        per_judge = {n: judge_stats(col[n], chars, toks) for n in names}
        pairs = {f"{a} vs {b}": pair_stats(col[a], col[b]) for a, b in itertools.combinations(names, 2)}
        out[m] = {"n_rows": len(mrows), "per_judge": per_judge, "pairs": pairs,
                  **panel_agreement([[r["id"], [r["scores"][m].get(n) for n in live]] for r in mrows], live)}
    out["length_effects_p_lt_0_05"] = [
        {"metric": m, "judge": n, "subset": sub, "rho": j[f"spearman_len_chars{suf}"], "p": j[f"perm_p_len_chars{suf}"],
         "n": j["n_uncut"] if suf else j["n_scored"]}
        for m in METRICS for n, j in out[m]["per_judge"].items() for sub, suf in (("all", ""), ("uncut", "_uncut"))
        if j[f"perm_p_len_chars{suf}"] is not None and j[f"perm_p_len_chars{suf}"] < 0.05]
    return out


def usage_by_model(calls: list[dict]) -> dict:
    by_model: dict = {}
    for c in calls:
        b = by_model.setdefault(c["model"], {"calls": 0, "tokens_in": 0, "tokens_out": 0, "ms": []})
        b["calls"] += 1
        b["tokens_in"] += c["in"]
        b["tokens_out"] += c["out"]
        b["ms"].append(c["ms"])
    return {k: {**{x: v[x] for x in ("calls", "tokens_in", "tokens_out")}, "p50_ms": statistics.median(v["ms"])}
            for k, v in by_model.items()}


def panel_agreement(units: list[list], live: list[str]) -> dict:
    """Agreement of the live judges as a panel; `units` is [[row_id, [score per live judge or None]], ...].

    Fleiss' kappa and the "all agree" shares need rows every judge scored, and pass% comparisons need the
    same rows for every judge, so those use the complete rows only. Rows a judge failed are listed with the
    scores they did get, and Krippendorff's alpha keeps them, so a failure on a contested row is visible
    instead of silently raising agreement.
    """
    full = [vals for _, vals in units if all(v is not None for v in vals)]
    partial = [(rid, vals) for rid, vals in units if any(v is None for v in vals)]
    common = {n: ev.pct(it[j] >= PASS for it in full) for j, n in enumerate(live)} if full else {}

    def split(vals: list) -> bool:
        got = {v >= PASS for v in vals if v is not None}
        return len(got) > 1

    return {"live_all_scored": len(full),
            "all_live_exact_pct": ev.pct(len(set(it)) == 1 for it in full) if full else None,
            "all_live_pass_agree_pct": ev.pct(len({v >= PASS for v in it}) == 1 for it in full) if full else None,
            "fleiss_kappa_pass": _r(fleiss_kappa([[v >= PASS for v in it] for it in full])),
            "krippendorff_alpha_pass": _r(krippendorff_alpha(
                [[None if v is None else v >= PASS for v in vals] for _, vals in units])),
            "krippendorff_alpha_interval": _r(krippendorff_alpha([vals for _, vals in units], "interval")),
            "pass_pct_common": common,
            "pass_pct_spread": round(max(common.values()) - min(common.values()), 1) if common else None,
            "incomplete_rows": [{"id": rid, "scores": dict(zip(live, vals))} for rid, vals in partial],
            "incomplete_pass_split": sum(split(vals) for _, vals in partial)}


def print_tables(summary: dict) -> None:
    for m in METRICS:
        s = summary[m]
        print(f"\n## {m}: n={s['n_rows']}, all live judges scored {s['live_all_scored']}, "
              f"all agree exact {s['all_live_exact_pct']}%, on pass {s['all_live_pass_agree_pct']}%, "
              f"Fleiss kappa(pass) {s['fleiss_kappa_pass']}, pass% on common rows {s['pass_pct_common']} "
              f"(spread {s['pass_pct_spread']}); all rows: Krippendorff alpha pass {s['krippendorff_alpha_pass']}, "
              f"interval {s['krippendorff_alpha_interval']}")
        for inc in s["incomplete_rows"]:
            print(f"  not scored by every judge: {inc['id']} {inc['scores']}")
        print("| judge | scored | failed | mean | pass% own rows | 1/2/3/4/5 | rho(len chars), p | rho(out tokens) "
              "| rho(uncut), p, n |")
        print("|---|---|---|---|---|---|---|---|---|")
        for n, j in s["per_judge"].items():
            dist = "/".join(str(j["dist"][str(v)]) for v in SCALE)
            print(f"| {n} | {j['n_scored']} | {j['failed']} | {j['mean']} | {j['pass_pct']} | {dist} | "
                  f"{j['spearman_len_chars']}, {j['perm_p_len_chars']} | {j['spearman_len_tokens']} | "
                  f"{j['spearman_len_chars_uncut']}, {j['perm_p_len_chars_uncut']}, {j['n_uncut']} |")
        print("\n| pair | n | exact% | pass agree% | kappa(pass) | kappa quadratic | MAD |")
        print("|---|---|---|---|---|---|---|")
        for p, v in s["pairs"].items():
            print(f"| {p} | {v['n']} | {v.get('exact_pct')} | {v.get('pass_agree_pct')} | {v.get('kappa_pass')} | "
                  f"{v.get('kappa_quadratic')} | {v.get('mad')} |")


# ── human check ──────────────────────────────────────────────────────────────
def pick_human_sample(rows: list[dict], live: list[str], k: int = 10, contested_max: int = 4) -> list[dict]:
    """Deterministic stratified sample: up to `contested_max` rows some judge failed (<4), the rest spread over tags.

    Contested rows are over-represented on purpose: they are where a human label decides who is right.
    """
    def lo(r: dict) -> int:
        vals = [r["scores"]["faithfulness"].get(n) for n in live]
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else PASS

    contested = sorted([r for r in rows if lo(r) < PASS], key=lambda r: (lo(r), r["id"]))[:contested_max]
    chosen = {r["id"] for r in contested}
    rest = sorted([r for r in rows if r["id"] not in chosen], key=lambda r: r["id"])
    by_tag: dict[str, list[dict]] = {}
    for r in rest:
        by_tag.setdefault((r["tags"] or ["-"])[0], []).append(r)
    picked = [{"id": r["id"], "stratum": "contested"} for r in contested]
    while len(picked) < k and any(by_tag.values()):
        for tag in sorted(by_tag):
            if by_tag[tag] and len(picked) < k:
                picked.append({"id": by_tag[tag].pop(0)["id"], "stratum": f"tag:{tag}"})
    return picked


def write_human_csv(path: Path, sample: list[dict], inputs: list[dict]) -> None:
    """Nothing in the file hints at what the judges said, so the labeller is not anchored by it.

    No scores, and rows are in id order: `pick_human_sample` puts contested rows (some judge failed them)
    first, and that order would tell the labeller which answers a judge failed.
    """
    by_id = {r["id"]: r for r in inputs}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:  # BOM: Excel opens Cyrillic correctly
        w = csv.writer(fh)
        w.writerow(["id", "question", "sources", "answer", HUMAN_COLUMN, "comment"])
        for rid in sorted(s["id"] for s in sample):
            r = by_id[rid]
            w.writerow([r["id"], r["question"], r["sources"], r["answer"], "", ""])


def read_human_labels(path: Path) -> dict[str, int]:
    labels = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for rec in csv.DictReader(fh):
            v = (rec.get(HUMAN_COLUMN) or "").strip()
            if not v:
                continue
            if v not in {str(x) for x in SCALE}:
                raise SystemExit(f"{rec['id']}: {HUMAN_COLUMN} must be 1..5, got {v!r}")
            labels[rec["id"]] = int(v)
    return labels


def human_agreement(labels: dict[str, int], results: dict) -> dict:
    rows = {r["id"]: r for r in results["rows"] if r["id"] in labels}
    live = results["summary"]["live_judges"]
    ids = sorted(rows)
    human = [labels[i] for i in ids]
    out: dict = {"n_labelled": len(ids), "human_mean": round(statistics.mean(human), 2) if human else None,
                 "human_pass_pct": ev.pct(h >= PASS for h in human) if human else None, "vs": {}}
    for n in [*live, STORED]:
        out["vs"][n] = pair_stats(human, [rows[i]["scores"]["faithfulness"].get(n) for i in ids])
    medians = []
    for i in ids:
        vals = [rows[i]["scores"]["faithfulness"].get(n) for n in live]
        vals = [v for v in vals if v is not None]
        medians.append(round(statistics.median(vals)) if vals else None)  # judge panel = median of live judges
    out["vs"]["median_of_live_judges"] = pair_stats(human, medians)
    return out


def scoring_history(prev: dict) -> list[dict]:
    """Per-run token usage of earlier scoring runs; older result files kept one run in `usage_scoring_run`."""
    if "scoring_runs" in prev:
        return list(prev["scoring_runs"])
    if "usage_scoring_run" in prev:
        return [{"at": prev.get("measured_at"), "usage": prev["usage_scoring_run"]}]
    return []


# ── CLI ──────────────────────────────────────────────────────────────────────
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    ap.add_argument("--judges", default=DEFAULT_JUDGES)
    ap.add_argument("--retest", help="score a second pass with this judge (test-retest at temperature 0)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--rebuild-inputs", action="store_true")
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--human-csv", type=Path, default=HUMAN_DIR / "faithfulness_sample.csv")
    ap.add_argument("--out", type=Path, default=RESULTS / "judge_agreement.json")
    return ap


def run_human(args) -> int:
    if not args.out.exists():
        raise SystemExit(f"{args.out} not found: score the judges first.")
    labels = read_human_labels(args.human_csv)
    if not labels:
        print(f"No labels yet: fill the {HUMAN_COLUMN} column in {args.human_csv} (see evals/human/README.md).")
        return 1
    res = human_agreement(labels, json.loads(args.out.read_text(encoding="utf-8")))
    dst = args.out.with_name(args.out.stem + "_human.json")
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("| judge | n | exact% | pass agree% | kappa(pass) | kappa quadratic | MAD |\n|---|---|---|---|---|---|---|")
    for n, v in res["vs"].items():
        print(f"| {n} | {v['n']} | {v.get('exact_pct')} | {v.get('pass_agree_pct')} | {v.get('kappa_pass')} | "
              f"{v.get('kappa_quadratic')} | {v.get('mad')} |")
    print(f"→ {dst}")
    return 0


async def main() -> int:
    args = parser().parse_args()
    if args.human:
        return run_human(args)
    os.environ.setdefault("AI_DATA_DIR", str(ROOT / "data" / "judge-agreement"))
    ev.boot_settings(argparse.Namespace(trace=False))
    data_dir = Path(os.environ["AI_DATA_DIR"])
    inputs_file, cache = data_dir / "judge_inputs.json", Cache(data_dir / "judge_cache.json")
    if not inputs_file.exists() or args.rebuild_inputs:
        inputs = await build_inputs(args.answers, data_dir)
        inputs_file.write_text(json.dumps(inputs, ensure_ascii=False, indent=1), encoding="utf-8")
    inputs = json.loads(inputs_file.read_text(encoding="utf-8"))[: args.limit or None]
    specs = [s.strip() for s in args.judges.split(",") if s.strip()]
    from tulpar_ai import llm

    # what only scoring runs know (probe verdicts, time, tokens) survives --stats-only and later re-runs
    prev = json.loads(args.out.read_text(encoding="utf-8"))["summary"] if args.out.exists() else {}
    history = scoring_history(prev)
    run_meta: dict = {k: prev[k] for k in ("unavailable", "measured_at") if k in prev}
    with llm.record() as calls:
        if not args.stats_only:
            run_meta = {"unavailable": {}, "measured_at": time.strftime("%Y-%m-%d %H:%M")}
            _, first_row, first_user = tasks_for(inputs[:1])[0]
            for spec in specs:
                err = await probe(spec, first_row["id"], first_user, cache)
                if err:
                    run_meta["unavailable"][spec] = err
                    print(f"judge {spec} unavailable: {err[-160:]}")
            live_specs = [s for s in specs if s not in run_meta["unavailable"]]
            await score_all(live_specs, inputs, cache, args.retest)
        else:
            live_specs = [s for s in specs if any(k.startswith(s + "|") for k in cache.data)]
    rows = collect(live_specs, inputs, cache, args.retest)
    summary = summarize(rows, [label(s) for s in live_specs], args.retest)
    if not args.stats_only:
        history.append({"at": run_meta["measured_at"], "usage": usage_by_model(calls)})
    summary.update({"scoring_runs": history,
                    "answers_file": str(args.answers.relative_to(ROOT)) if args.answers.is_relative_to(ROOT) else str(args.answers),
                    "judge_specs": live_specs, "retest": args.retest,
                    "stored_judge": "scores saved by the original QA run (JUDGE_MODELS chain of that day)",
                    "answers_cut_at_400_chars": sum(r["answer_chars"] >= 400 for r in rows),
                    "sources_rank_match": sum(r["rank_now"] == r["rank_stored"] for r in rows),
                    "sources_sufficient": sum(r["sufficient"] for r in rows), **run_meta})
    if not args.limit:
        summary["human_sample"] = pick_human_sample(rows, summary["live_judges"])
        # never overwrite labels someone already typed in
        if not args.human_csv.exists() or not read_human_labels(args.human_csv):
            write_human_csv(args.human_csv, summary["human_sample"], inputs)
            print(f"human sample: {len(summary['human_sample'])} rows → {args.human_csv}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print_tables(summary)
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
