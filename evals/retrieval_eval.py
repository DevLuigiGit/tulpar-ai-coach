"""Retrieval-only evaluation of a named retrieval config: hit@1, hit@4, recall@10, MRR, nDCG@10 with bootstrap CIs.

    python evals/retrieval_eval.py run                                   # prod_jina_dense on synth + qa
    python evals/retrieval_eval.py run --config prod_jina_dense --config prod_local_fallback --out FILE
    python evals/retrieval_eval.py run --config jina_dense_np:256        # Matryoshka from cached 1024-d vectors
    python evals/retrieval_eval.py run --plugin evals.variants.hybrid --config hybrid   # a variant from another module
    python evals/retrieval_eval.py compare A.json B.json [--a NAME --b NAME] [--tag T]  # paired, from saved rows
    python evals/retrieval_eval.py configs

A config is anything that answers `retrieve(query, k) -> ranked chunk ids` (RetrievalConfig below). It does not
touch the chat graph: no router, no rewrite loop, no answer. Two golden sets run by default:
  synth  evals/golden/retrieval_synth.jsonl — SYNTHETIC questions generated from sampled chunks (see its README)
  qa     evals/golden/qa.jsonl, answerable rows, expected sources turned into matchers with the same rules as
         evals/run.py (exercise by title, nutrition by source, WHO page ±1)
Any other JSONL file in the retrieval format can be passed to --sets by path.

Relevance. A row has `relevant`: a list of matchers, each one a GROUP of interchangeable chunks:
  "who:400:12:3"                                   one chunk id
  {"id": ..., "source": ..., "page": ..., "text": ...}   the id if the evaluated chunking has it and that chunk
                                                    still covers `text`; otherwise every chunk of that
                                                    source/page covering ≥ 50% of the word 3-grams of `text`
  {"source": "who2020", "pages": [11, 12, 13]}     every chunk of the source on those pages
  {"source": "exercises", "title": "Жим штанги лёжа"}   every chunk with that title
Metrics per question (ranked list = the first DEPTH=20 ids, the production dense top_k):
  hit@k      1 if a chunk of any group is in the top k
  MRR        1 / rank of the first chunk of any group (0 if none in the top 20)
  recall@10  share of the row's groups with at least one chunk in the top 10
  nDCG@10    binary gain 1 at a rank whose chunk covers a group not covered higher up;
             ideal DCG = min(number of groups, 10) gains at the top
The mean over questions gets a percentile bootstrap 95% CI (1000 resamples of questions, fixed seed). Per tag the
same. A paired comparison of two configs on one set resamples the same question ids for both (CI and two-sided
p of the mean difference) and adds an exact McNemar test on hit@4.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import inspect
import json
import math
import os
import re
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.emb_cache import CachedEmbedder, EmbeddingCache, truncate  # noqa: E402

GOLDEN = ROOT / "evals" / "golden"
RESULTS = ROOT / "evals" / "results"
DEPTH = 20
METRICS = ("hit@1", "hit@4", "recall@10", "mrr", "ndcg@10")
N_BOOT = 1000
SEED = 20260929
SPAN_MIN = 0.5
QA_PAGE_TOLERANCE = 1


# ── text helpers ─────────────────────────────────────────────────────────────
_WORD = re.compile(r"[0-9a-zа-я]+")
_DASH = re.compile(r"[‐-―−]")


def norm_text(s: str) -> str:
    s = (s or "").lower().replace("ё", "е").replace("­", "")
    s = _DASH.sub("-", s).replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", s).strip()


def words(s: str) -> list[str]:
    return _WORD.findall(norm_text(s))


def _grams(ws: list[str], n: int = 3) -> set[tuple[str, ...]]:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def span_coverage(span: str, text: str) -> float:
    """Share of the span's word 3-grams found in `text` (a span under 3 words: 1.0 if contained, else 0.0)."""
    sw = words(span)
    if len(sw) < 3:
        return 1.0 if sw and " ".join(sw) in " ".join(words(text)) else 0.0
    g = _grams(sw)
    return len(g & _grams(words(text))) / len(g)


# ── relevance ────────────────────────────────────────────────────────────────
def _as_dict(c: Any) -> dict:
    return c if isinstance(c, dict) else dict(c.__dict__)


@dataclass
class Universe:
    """The chunks a config's ids refer to; matchers are resolved against it."""

    chunks: list[dict]
    by_id: dict[str, dict] = field(default_factory=dict)
    by_source: dict[str, list[dict]] = field(default_factory=dict)

    @classmethod
    def of(cls, chunks: Iterable[Any]) -> "Universe":
        u = cls([_as_dict(c) for c in chunks])
        for c in u.chunks:
            u.by_id[c["id"]] = c
            u.by_source.setdefault(c["source"], []).append(c)
        return u


def resolve(matcher: str | dict, uni: Universe) -> frozenset[str]:
    m = {"id": matcher} if isinstance(matcher, str) else matcher
    text = m.get("text")
    cid = m.get("id")
    if cid and cid in uni.by_id and (not text or span_coverage(text, uni.by_id[cid]["text"]) >= SPAN_MIN):
        return frozenset({cid})
    src = m.get("source")
    if not src:
        return frozenset()
    pages = set(m["pages"]) if m.get("pages") else ({m["page"]} if m.get("page") is not None else None)
    out = []
    for c in uni.by_source.get(src, []):
        if pages is not None and c.get("page") not in pages:
            continue
        if m.get("title") and c.get("title") != m["title"]:
            continue
        if text and span_coverage(text, c["text"]) < SPAN_MIN:
            continue
        out.append(c["id"])
    return frozenset(out)


def groups_for(row: dict, uni: Universe) -> list[frozenset[str]]:
    seen, out = set(), []
    for m in row.get("relevant", []):
        g = resolve(m, uni)
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out


# ── metrics ──────────────────────────────────────────────────────────────────
def score_row(ranked: list[str], groups: list[frozenset[str]], depth: int = DEPTH) -> dict:
    ranked = list(ranked)[:depth]
    first = None
    covered: set[int] = set()
    covered10 = 0
    dcg = 0.0
    for r, cid in enumerate(ranked, 1):
        new = {i for i, g in enumerate(groups) if cid in g and i not in covered}
        if any(cid in g for g in groups) and first is None:
            first = r
        if new:
            covered |= new
            if r <= 10:
                dcg += 1 / math.log2(r + 1)
                covered10 = len(covered)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(groups), 10) + 1))
    return {"rank": first,
            "hit@1": float(first is not None and first <= 1), "hit@4": float(first is not None and first <= 4),
            "recall@10": covered10 / len(groups) if groups else 0.0,
            "mrr": 1 / first if first else 0.0, "ndcg@10": dcg / ideal if ideal else 0.0}


# ── statistics ───────────────────────────────────────────────────────────────
def _seed(seed: int, label: str) -> int:
    return seed + zlib.crc32(label.encode("utf-8"))


def bootstrap(values: dict[str, np.ndarray], n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, dict]:
    """Mean and percentile 95% CI per metric; one set of resampled question indices for all metrics."""
    n = len(next(iter(values.values()))) if values else 0
    if n == 0:
        return {k: {"mean": None, "lo": None, "hi": None} for k in values}
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    out = {}
    for k, v in values.items():
        v = np.asarray(v, dtype=float)
        means = v[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        out[k] = {"mean": round(float(v.mean()), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: b = only A hits, c = only B hits (binomial test of b vs c at p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_compare(rows_a: list[dict], rows_b: list[dict], n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """B minus A on the questions both have, matched by id."""
    a = {r["id"]: r for r in rows_a}
    b = {r["id"]: r for r in rows_b}
    ids = [i for i in a if i in b]
    out: dict[str, Any] = {"n": len(ids), "n_boot": n_boot, "metrics": {}}
    if not ids:
        return out
    idx = np.random.default_rng(seed).integers(0, len(ids), size=(n_boot, len(ids)))
    for m in METRICS:
        va = np.array([a[i][m] for i in ids], dtype=float)
        vb = np.array([b[i][m] for i in ids], dtype=float)
        d = vb - va
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = min(1.0, 2 * min(float((boot <= 0).mean()), float((boot >= 0).mean())))
        out["metrics"][m] = {"a": round(float(va.mean()), 4), "b": round(float(vb.mean()), 4),
                             "diff": round(float(d.mean()), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                             "p_boot": round(p, 4)}
    only_a = sum(1 for i in ids if a[i]["hit@4"] and not b[i]["hit@4"])
    only_b = sum(1 for i in ids if b[i]["hit@4"] and not a[i]["hit@4"])
    out["mcnemar_hit@4"] = {"only_a": only_a, "only_b": only_b, "p": float(f"{mcnemar_exact(only_a, only_b):.3g}")}
    return out


# ── golden sets ──────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def qa_rows(path: Path | None = None, page_tolerance: int = QA_PAGE_TOLERANCE) -> list[dict]:
    """evals/golden/qa.jsonl in the retrieval format; unanswerable rows have nothing to find and are left out.

    One group per expected source, the same acceptance rules as `_hit_rank` in evals/run.py.
    """
    rows = []
    for r in load_jsonl(path or GOLDEN / "qa.jsonl"):
        if not r.get("answerable"):
            continue
        rel = []
        for e in r["expected_sources"]:
            if e["source"] == "exercises":
                rel.append({"source": "exercises", "title": e["title"]})
            elif e["source"] == "who2020" and e.get("page"):
                p = int(e["page"])
                rel.append({"source": "who2020", "pages": list(range(p - page_tolerance, p + page_tolerance + 1))})
            else:
                rel.append({"source": e["source"]})
        rows.append({"id": r["id"], "question": r["question"], "relevant": rel, "tags": list(r.get("tags", [])),
                     "synthetic": False})
    return rows


def synth_rows(path: Path | None = None) -> list[dict]:
    return load_jsonl(path or GOLDEN / "retrieval_synth.jsonl")


SETS: dict[str, tuple[Callable[[], list[dict]], str, bool]] = {
    "synth": (synth_rows, "evals/golden/retrieval_synth.jsonl", True),
    "qa": (qa_rows, "evals/golden/qa.jsonl (answerable rows)", False),
}


def load_set(spec: str) -> tuple[str, list[dict], str, bool]:
    if spec in SETS:
        fn, label, synthetic = SETS[spec]
        return spec, fn(), label, synthetic
    rows = load_jsonl(Path(spec))
    return Path(spec).stem, rows, spec, all(r.get("synthetic") for r in rows)


# ── configs ──────────────────────────────────────────────────────────────────
RetrieveFn = Callable[[str, int], "Awaitable[list[str]] | list[str]"]


@dataclass
class RetrievalConfig:
    """What the evaluator needs from a retrieval variant.

    retrieve  (query, k) -> chunk ids, best first (sync or async)
    chunks    the chunk universe those ids come from, if not the production chunking (load_chunks(400))
    warmup    optional: called once with every question of a set (embed all queries in a few batched API calls)
    """

    name: str
    retrieve: RetrieveFn
    meta: dict = field(default_factory=dict)
    chunks: list | None = None
    warmup: Callable[[list[str]], Awaitable[None]] | None = None
    close: Callable[[], Any] | None = None


Factory = Callable[["EvalContext", "str | None"], Awaitable[RetrievalConfig]]
CONFIGS: dict[str, tuple[Factory, str]] = {}


def register(name: str, help: str = "") -> Callable[[Factory], Factory]:
    def deco(fn: Factory) -> Factory:
        CONFIGS[name] = (fn, help)
        return fn
    return deco


class EvalContext:
    """Shared by the configs of one run: data dir, the embedding cache, the production chunks, API usage."""

    def __init__(self, data_dir: Path | None = None, cache: EmbeddingCache | None = None):
        from tulpar_ai.config import get_settings

        self.data_dir = Path(data_dir or get_settings().ai_data_dir)
        self.cache = cache or EmbeddingCache()
        self._chunks: dict[int, list] = {}
        self.embedders: list[CachedEmbedder] = []

    def chunks(self, pdf_chunk: int = 400) -> list:
        if pdf_chunk not in self._chunks:
            from tulpar_ai.rag.index import load_chunks

            self._chunks[pdf_chunk] = load_chunks(pdf_chunk=pdf_chunk)
        return self._chunks[pdf_chunk]

    def jina(self, dim: int | None = None) -> CachedEmbedder:
        from tulpar_ai.rag.embed import JinaEmbedder

        emb = CachedEmbedder(JinaEmbedder(), self.cache, dim=dim)
        self.embedders.append(emb)
        return emb

    def api_usage(self) -> dict:
        return {"api_texts": sum(e.api_texts for e in self.embedders),
                "api_calls": sum(e.api_calls for e in self.embedders)}


async def index_config(ctx: EvalContext, name: str, embedder, *, rerank: bool, pdf_chunk: int = 400,
                       help: str = "") -> RetrievalConfig:
    """A config that goes through the production code: rag.index.Index (Qdrant) + rag.retrieve.retrieve."""
    from tulpar_ai.config import get_settings
    from tulpar_ai.rag import qdrant
    from tulpar_ai.rag.index import Index
    from tulpar_ai.rag.retrieve import retrieve

    s = get_settings()
    idx = Index(embedder=embedder, path=ctx.data_dir / "qdrant", pdf_chunk=pdf_chunk)
    points = await idx.build()

    async def run(query: str, k: int) -> list[str]:
        hits = await retrieve(query, top_k=max(k, s.rag_top_k), top_n=k, rerank=rerank, index=idx)
        return [h["id"] for h in hits]

    async def warmup(queries: list[str]) -> None:
        if isinstance(embedder, CachedEmbedder):
            await embedder.matrix(queries, "retrieval.query")

    meta = {"description": help, "code_path": "tulpar_ai.rag.retrieve.retrieve", "embedder": embedder.id,
            "embed_model": getattr(embedder, "model", embedder.id), "rerank": rerank,
            "reranker": ("jina" if "jina" in embedder.id else "lexical") if rerank else None,
            "dense_top_k": max(DEPTH, s.rag_top_k), "production_top_k": s.rag_top_k, "production_top_n": s.rag_top_n,
            "pdf_chunk": pdf_chunk, "collection": idx.collection, "qdrant_mode": qdrant.mode(), "points": points}
    return RetrievalConfig(name, run, meta, warmup=warmup, close=idx.close)


class DenseMatrix:
    """Exact cosine search over unit vectors in memory: a base for variants that do not need Qdrant."""

    def __init__(self, ids: list[str], matrix: np.ndarray):
        self.ids = ids
        self.m = np.asarray(matrix, dtype=np.float32)

    def search(self, vec: np.ndarray, k: int) -> list[str]:
        scores = self.m @ np.asarray(vec, dtype=np.float32)
        top = np.argsort(-scores, kind="stable")[:k]
        return [self.ids[i] for i in top]


@register("prod_jina_dense", "production: Jina v3 dense (Qdrant, cosine), top_k 20, no rerank — RAG_RERANK=auto is off "
                             "for Jina by the earlier A/B")
async def _prod_jina(ctx: EvalContext, arg: str | None) -> RetrievalConfig:
    return await index_config(ctx, "prod_jina_dense", ctx.jina(), rerank=False, help=CONFIGS["prod_jina_dense"][1])


@register("prod_local_fallback", "production without a Jina key: local hashing embedder + lexical rerank of the "
                                 "top 20 (RAG_RERANK=auto is on for it)")
async def _prod_local(ctx: EvalContext, arg: str | None) -> RetrievalConfig:
    from tulpar_ai.rag.embed import LocalHashEmbedder

    return await index_config(ctx, "prod_local_fallback", LocalHashEmbedder(), rerank=True,
                              help=CONFIGS["prod_local_fallback"][1])


@register("jina_dense_np", "exact cosine in numpy over cached Jina vectors; jina_dense_np:256 = Matryoshka 256-d "
                           "(truncated + renormalised cached 1024-d vectors, no new API calls)")
async def _jina_np(ctx: EvalContext, arg: str | None) -> RetrievalConfig:
    emb = ctx.jina(int(arg) if arg else None)
    chunks = ctx.chunks()
    dense = DenseMatrix([c.id for c in chunks],
                        truncate(await emb.matrix([c.text for c in chunks], "retrieval.passage"), emb.dim))

    async def run(query: str, k: int) -> list[str]:
        return dense.search(truncate(await emb.matrix([query], "retrieval.query"), emb.dim)[0], k)

    async def warmup(queries: list[str]) -> None:
        await emb.matrix(queries, "retrieval.query")

    return RetrievalConfig(f"jina_dense_np{':' + arg if arg else ''}", run,
                           {"description": CONFIGS["jina_dense_np"][1], "embedder": emb.id, "dim": emb.dim,
                            "embed_model": emb.model}, warmup=warmup)


async def make_config(spec: str, ctx: EvalContext) -> RetrievalConfig:
    name, _, arg = spec.partition(":")
    if name not in CONFIGS:
        raise SystemExit(f"unknown config {name!r}; known: {', '.join(sorted(CONFIGS))} (plug-ins: --plugin module)")
    return await CONFIGS[name][0](ctx, arg or None)


# ── evaluation ───────────────────────────────────────────────────────────────
def summarize(scored: list[dict], n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    vals = {m: np.array([r[m] for r in scored], dtype=float) for m in METRICS}
    by_tag = {}
    for tag in sorted({t for r in scored for t in r.get("tags", [])}):
        sub = [r for r in scored if tag in r.get("tags", [])]
        by_tag[tag] = {"n": len(sub), **bootstrap({m: np.array([r[m] for r in sub], dtype=float) for m in METRICS},
                                                  n_boot, _seed(seed, tag))}
    return {"n": len(scored), "metrics": bootstrap(vals, n_boot, seed), "by_tag": by_tag}


async def evaluate(config: RetrievalConfig, rows: list[dict], *, universe: Universe, depth: int = DEPTH,
                   n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    uni = Universe.of(config.chunks) if config.chunks is not None else universe
    work, unresolved = [], []
    for r in rows:
        g = groups_for(r, uni)
        if g:
            work.append((r, g))
        else:
            unresolved.append(r["id"])
    if config.warmup:
        await config.warmup([r["question"] for r, _ in work])
    scored, t0 = [], time.perf_counter()
    for r, g in work:
        got = config.retrieve(r["question"], depth)
        ranked = list(await got if inspect.isawaitable(got) else got)
        s = score_row(ranked, g, depth)
        scored.append({"id": r["id"], "tags": r.get("tags", []), **s, "n_groups": len(g),
                       "n_relevant_chunks": len(frozenset().union(*g)), "top": ranked[:10]})
    ms = (time.perf_counter() - t0) * 1000 / (len(work) or 1)
    return {**summarize(scored, n_boot, seed), "unresolved": unresolved, "ms_per_query": round(ms, 1), "rows": scored}


async def evaluate_callable(retrieve: RetrieveFn, sets: Iterable[str] = ("synth", "qa"), *, name: str = "variant",
                            chunks: list | None = None, n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, dict]:
    """Shortest path for a notebook or another script: a bare retrieve(query, k) function → results per set."""
    from tulpar_ai.rag.index import load_chunks

    universe = Universe.of(chunks if chunks is not None else load_chunks(pdf_chunk=400))
    cfg = RetrievalConfig(name, retrieve, chunks=chunks)
    out = {}
    for spec in sets:
        set_name, rows, _, _ = load_set(spec)
        out[set_name] = await evaluate(cfg, rows, universe=universe, n_boot=n_boot, seed=seed)
    return out


def fmt_metric(m: str, x: dict) -> str:
    if x["mean"] is None:
        return "—"
    if m.startswith(("hit", "recall")):
        return f"{100 * x['mean']:.1f} [{100 * x['lo']:.1f}–{100 * x['hi']:.1f}]"
    return f"{x['mean']:.3f} [{x['lo']:.3f}–{x['hi']:.3f}]"


def table(res: dict, label: str) -> str:
    head = "| " + label + " | n | " + " | ".join(METRICS) + " |\n|---|---|" + "---|" * len(METRICS)
    lines = [head, f"| all | {res['n']} | " + " | ".join(fmt_metric(m, res["metrics"][m]) for m in METRICS) + " |"]
    for tag, t in res["by_tag"].items():
        lines.append(f"| {tag} | {t['n']} | " + " | ".join(fmt_metric(m, t[m]) for m in METRICS) + " |")
    return "\n".join(lines)


def compare_table(cmp: dict, a: str, b: str) -> str:
    """p_boot = 0 means no resample crossed zero: shown as < 1/n_boot."""
    floor = f"< {1 / cmp.get('n_boot', N_BOOT):g}"
    lines = [f"| {b} − {a} (n={cmp['n']}) | A | B | diff | 95% CI | p (bootstrap) |", "|---|---|---|---|---|---|"]
    for m, x in cmp["metrics"].items():
        p = floor if x["p_boot"] == 0 else x["p_boot"]
        lines.append(f"| {m} | {x['a']:.4f} | {x['b']:.4f} | {x['diff']:+.4f} | [{x['lo']:+.4f}, {x['hi']:+.4f}] | {p} |")
    mc = cmp.get("mcnemar_hit@4")
    if mc:
        lines.append(f"| McNemar hit@4 | only A: {mc['only_a']} | only B: {mc['only_b']} | | | {mc['p']} |")
    return "\n".join(lines)


def load_plugin(module: str) -> None:
    """Import a module whose @register(...) functions add configs, e.g. evals.variants.hybrid or a file path."""
    if module.endswith(".py"):
        spec = importlib.util.spec_from_file_location(Path(module).stem, module)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[Path(module).stem] = mod
        spec.loader.exec_module(mod)
    else:
        importlib.import_module(module)


def boot_env(trace: bool = False) -> None:
    if not trace:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from tulpar_ai.config import get_settings

    get_settings.cache_clear()


async def cmd_run(args) -> dict:
    boot_env(args.trace)
    for p in args.plugin or []:
        load_plugin(p)
    ctx = EvalContext()
    universe = Universe.of(ctx.chunks())
    sets = [load_set(s.strip()) for s in args.sets.split(",") if s.strip()]
    out: dict[str, Any] = {
        "kind": "retrieval_eval", "created": time.strftime("%Y-%m-%d %H:%M"),
        "evaluator": {"depth": args.depth, "metrics": list(METRICS), "bootstrap": args.n_boot, "seed": args.seed,
                      "ci": "percentile 95%", "span_min": SPAN_MIN, "qa_page_tolerance": QA_PAGE_TOLERANCE},
        "sets": {name: {"golden": label, "synthetic": synthetic, "rows": len(rows)} for name, rows, label, synthetic in sets},
        "configs": {}, "comparisons": []}
    for spec in args.config or ["prod_jina_dense"]:
        cfg = await make_config(spec, ctx)
        before = ctx.api_usage()
        entry: dict[str, Any] = {"meta": cfg.meta, "sets": {}}
        try:
            for name, rows, label, synthetic in sets:
                res = await evaluate(cfg, rows, universe=universe, depth=args.depth, n_boot=args.n_boot, seed=args.seed)
                entry["sets"][name] = res
                tag = " (SYNTHETIC)" if synthetic else ""
                print(f"\n### {cfg.name} — {name}{tag}: {res['n']} questions, unresolved {len(res['unresolved'])}\n")
                print(table(res, "tag"))
        finally:
            if cfg.close:
                cfg.close()
        after = ctx.api_usage()
        entry["embedding_api"] = {k: after[k] - before[k] for k in after}
        out["configs"][cfg.name] = entry
    names = list(out["configs"])
    for other in names[1:]:
        for s in out["sets"]:
            cmp = paired_compare(out["configs"][names[0]]["sets"][s]["rows"], out["configs"][other]["sets"][s]["rows"],
                                 args.n_boot, args.seed)
            out["comparisons"].append({"set": s, "a": names[0], "b": other, **cmp})
            print(f"\n### paired: {s}\n\n{compare_table(cmp, names[0], other)}")
    out["embedding_cache"] = {"dir": str(ctx.cache.root), "hits": ctx.cache.hits, "misses": ctx.cache.misses,
                              **ctx.api_usage()}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nsaved {args.out}")
    return out


def cmd_compare(args) -> dict:
    configs: dict[str, dict] = {}
    for f in args.files:
        for name, entry in json.loads(Path(f).read_text(encoding="utf-8"))["configs"].items():
            configs.setdefault(name, entry)
    names = list(configs)
    a, b = args.a or names[0], args.b or (names[1] if len(names) > 1 else None)
    if not b:
        raise SystemExit("need two configs: two result files or one file with two configs (--a/--b)")
    out = []
    for s in configs[a]["sets"]:
        if s not in configs[b]["sets"] or (args.set and s != args.set):
            continue
        ra, rb = configs[a]["sets"][s]["rows"], configs[b]["sets"][s]["rows"]
        if args.tag:  # a paired test on one slice, e.g. --tag cross-lingual
            ra, rb = ([r for r in rows if args.tag in r.get("tags", [])] for rows in (ra, rb))
        cmp = paired_compare(ra, rb, args.n_boot, args.seed)
        if not cmp["n"]:
            print(f"\n### paired: {s}: no common questions{f' with tag {args.tag}' if args.tag else ''}")
            continue
        out.append({"set": s, "tag": args.tag, "a": a, "b": b, **cmp})
        print(f"\n### paired: {s}{f' [{args.tag}]' if args.tag else ''}\n\n{compare_table(cmp, a, b)}")
    return {"comparisons": out}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", action="append", help="registered config, optionally name:arg; repeat to compare")
    r.add_argument("--plugin", action="append", help="module or .py file that registers more configs")
    r.add_argument("--sets", default="synth,qa", help="synth, qa or paths to retrieval-format JSONL, comma separated")
    r.add_argument("--out", help="write the full result (summary, per-tag, per-row) to this JSON")
    r.add_argument("--depth", type=int, default=DEPTH)
    r.add_argument("--n-boot", type=int, default=N_BOOT)
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--trace", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+")
    c.add_argument("--a")
    c.add_argument("--b")
    c.add_argument("--set")
    c.add_argument("--tag", help="only questions with this tag")
    c.add_argument("--n-boot", type=int, default=N_BOOT)
    c.add_argument("--seed", type=int, default=SEED)
    sub.add_parser("configs")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.cmd == "configs":
        for name, (_, help) in sorted(CONFIGS.items()):
            print(f"{name:22} {help}")
        return 0
    if args.cmd == "compare":
        cmd_compare(args)
        return 0
    asyncio.run(cmd_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
