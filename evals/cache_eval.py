"""Semantic answer cache: where to put the threshold, and what a hit saves.

    python evals/cache_eval.py sweep [--embedder jina|local]   # embeddings only, no LLM calls
    python evals/cache_eval.py latency [--n 10]                # live: full question pipeline vs cache hit
    python evals/cache_eval.py all

sweep: every `a` of evals/golden/cache_pairs.jsonl is stored in the cache, every `b` is looked up and its nearest
stored question decides. A paraphrase must hit its own `a`; a near-miss must not hit anything; a paraphrase landing
on someone else's `a` is a wrong answer too. The chosen threshold is the lowest one with zero wrong hits — a wrong
cached answer is worse than a miss.

latency: for N paraphrase pairs, `a` goes through the real nodes (cache miss → retrieve → answer → store) and then
`b` is looked up. The router call happens before the cache on both paths, so it is left out of both.

Results are merged into evals/results/cache.json, so the two parts can run separately.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

from run import RESULTS, load, p95, pct, price  # noqa: E402

COARSE = [0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
FINE = [round(0.80 + i / 100, 2) for i in range(20)]
OUT = RESULTS / "cache.json"


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def nearest(rows: list[dict], vecs: dict[str, list[float]]) -> list[dict]:
    """For every `b`: the closest stored `a` (all `a` of the file are in the cache at once) and the pair's own score."""
    stored = list(dict.fromkeys(r["a"] for r in rows))
    out = []
    for r in rows:
        scores = [(_cos(vecs[r["b"]], vecs[a]), a) for a in stored]
        top, top_a = max(scores)
        out.append({"id": r["id"], "kind": r["kind"], "a": r["a"], "b": r["b"], "pair_score": round(_cos(vecs[r["b"]], vecs[r["a"]]), 4),
                    "top_score": round(top, 4), "top_is_own": top_a == r["a"], "top_a": top_a})
    return out


def at_threshold(scored: list[dict], thr: float) -> dict:
    para = [x for x in scored if x["kind"] == "paraphrase"]
    near = [x for x in scored if x["kind"] == "near_miss"]
    right = [x for x in para if x["top_score"] >= thr and x["top_is_own"]]
    wrong_target = [x for x in para if x["top_score"] >= thr and not x["top_is_own"]]
    false_near = [x for x in near if x["top_score"] >= thr]
    wrong = len(wrong_target) + len(false_near)
    return {"threshold": thr, "paraphrase_hit_rate": pct([x in right for x in para]),
            "near_miss_false_hit_rate": pct([x in false_near for x in near]),
            "paraphrase_wrong_target": len(wrong_target), "false_hits": wrong,
            "false_hit_rate": round(100 * wrong / (len(scored) or 1), 1),
            "false_hit_ids": [x["id"] for x in false_near + wrong_target]}


def choose(scored: list[dict]) -> float | None:
    """Lowest threshold on the fine grid with no wrong answer at all."""
    for thr in FINE:
        if at_threshold(scored, thr)["false_hits"] == 0:
            return thr
    return None


async def sweep(kind: str, task: str = "retrieval.query") -> dict:
    from tulpar_ai.rag.embed import get_embedder

    rows = load("cache_pairs.jsonl")
    emb = get_embedder(kind)
    texts = list(dict.fromkeys([r["a"] for r in rows] + [r["b"] for r in rows]))
    t0 = time.perf_counter()
    vectors = await emb.embed(texts, task=task)
    ms = int((time.perf_counter() - t0) * 1000)
    vecs = {t: _unit(v) for t, v in zip(texts, vectors)}
    scored = nearest(rows, vecs)
    para = [x["pair_score"] for x in scored if x["kind"] == "paraphrase"]
    near = [x["pair_score"] for x in scored if x["kind"] == "near_miss"]
    chosen = choose(scored)
    summary = {
        "embedder": emb.id, "task": task, "n_paraphrase": len(para), "n_near_miss": len(near), "stored_questions": len({r["a"] for r in rows}),
        "embed_ms_all_texts": ms, "paraphrase_score": {"min": min(para), "p50": round(statistics.median(para), 4), "max": max(para)},
        "near_miss_score": {"min": min(near), "p50": round(statistics.median(near), 4), "max": max(near)},
        "coarse": [at_threshold(scored, t) for t in COARSE], "fine": [at_threshold(scored, t) for t in FINE],
        "chosen_threshold": chosen, "at_chosen": at_threshold(scored, chosen) if chosen else None,
    }
    return {"summary": summary, "rows": scored}


async def latency(n: int, min_score: float | None) -> dict:
    from tulpar_ai import llm
    from tulpar_ai.config import get_settings
    from tulpar_ai.graph import chat as g
    from tulpar_ai.rag.answer_cache import get_answer_cache
    from tulpar_ai.rag.index import Index
    from tulpar_ai.rag.retrieve import set_index

    s = get_settings()
    if min_score is not None:
        s.answer_cache_min_score = min_score
    idx = Index(path=Path(s.ai_data_dir) / "qdrant")
    await idx.build()
    set_index(idx)
    cache = get_answer_cache()
    if cache.client.collection_exists(cache.collection):  # measure from an empty cache
        cache.client.delete_collection(cache.collection)
    cache._created.clear()

    paras = [r for r in load("cache_pairs.jsonl") if r["kind"] == "paraphrase"]
    step = max(1, len(paras) // n)
    picked = list({r["a"]: r for r in paras[::step]}.values())[:n]
    rows = []
    for r in picked:
        with llm.record() as calls:
            t0 = time.perf_counter()
            st: dict = {"text": r["a"], "intent": "question"}
            t1 = time.perf_counter()
            st.update(await g.cache_lookup(st))
            miss_ms = (time.perf_counter() - t1) * 1000
            while not st.get("reply"):
                st.update(await g.retrieve_node(st))
                nxt = g.after_retrieve(st)
                if nxt == "rewrite":
                    st.update(await g.rewrite(st))
                    continue
                if nxt == "answer":
                    st.update(await g.answer(st))
                break
            t2 = time.perf_counter()
            if st.get("reply") and st.get("kind") == "answer":
                await g.cache_store(st)
            store_ms = (time.perf_counter() - t2) * 1000
            full_ms = (time.perf_counter() - t0) * 1000
        vec = await idx.embed_query(r["a"])  # memoized: times the Qdrant part of a lookup alone
        t4 = time.perf_counter()
        cache._nearest(cache.collection, vec, time.time())
        qdrant_ms = (time.perf_counter() - t4) * 1000
        idx._qvecs.clear()  # a repeat from another client arrives later: pay for the embedding again
        with llm.record() as hit_calls:
            t3 = time.perf_counter()
            exact = await g.cache_lookup({"text": r["a"], "intent": "question"})
            exact_ms = (time.perf_counter() - t3) * 1000
            t5 = time.perf_counter()
            para = await g.cache_lookup({"text": r["b"], "intent": "question"})
            para_ms = (time.perf_counter() - t5) * 1000
        rows.append({"id": r["id"], "answered": st.get("kind") == "answer", "stored": bool(st.get("citations")),
                     "full_ms": round(full_ms), "miss_lookup_ms": round(miss_ms, 1), "qdrant_ms": round(qdrant_ms, 1),
                     "store_ms": round(store_ms, 1), "exact_hit": bool(exact.get("reply")), "exact_ms": round(exact_ms, 1),
                     "exact_same_reply": exact.get("reply") == st.get("reply"),
                     "hit": bool(para.get("reply")), "hit_score": para.get("cache_score"), "hit_ms": round(para_ms, 1),
                     "hit_same_reply": para.get("reply") == st.get("reply"), "hit_llm_calls": len(hit_calls),
                     "llm_calls": len(calls), "tokens_in": sum(c["in"] for c in calls), "tokens_out": sum(c["out"] for c in calls),
                     "cost_usd": round(sum(price(c["provider"], c["model"], c["in"], c["out"]) for c in calls), 6),
                     "models": sorted({f"{c['provider']}:{c['model']}" for c in calls})})
        print(json.dumps(rows[-1], ensure_ascii=False))
    hits = [x for x in rows if x["hit"]]
    exact = [x for x in rows if x["exact_hit"]]
    summary = {
        "n": len(rows), "min_score": cache.min_score, "embedder": idx.embedder.id,
        "answered": sum(x["answered"] for x in rows), "stored": sum(x["stored"] for x in rows),
        "exact_repeat_hits": len(exact), "exact_same_reply": sum(x["exact_same_reply"] for x in exact),
        "paraphrase_hits": len(hits), "paraphrase_same_reply": sum(x["hit_same_reply"] for x in hits),
        "full_pipeline_p50_ms": round(statistics.median(x["full_ms"] for x in rows)),
        "full_pipeline_p95_ms": p95([x["full_ms"] for x in rows]),
        "exact_hit_p50_ms": round(statistics.median(x["exact_ms"] for x in exact)) if exact else None,
        "paraphrase_hit_p50_ms": round(statistics.median(x["hit_ms"] for x in hits)) if hits else None,
        "miss_lookup_p50_ms": round(statistics.median(x["miss_lookup_ms"] for x in rows), 1),
        "qdrant_lookup_p50_ms": round(statistics.median(x["qdrant_ms"] for x in rows), 1),
        "store_p50_ms": round(statistics.median(x["store_ms"] for x in rows), 1),
        "llm_calls_per_full": round(statistics.mean(x["llm_calls"] for x in rows), 2),
        "llm_calls_per_hit": max((x["hit_llm_calls"] for x in rows), default=0),
        "tokens_per_full": {"in": round(statistics.mean(x["tokens_in"] for x in rows)),
                            "out": round(statistics.mean(x["tokens_out"] for x in rows))},
        "cost_per_full_usd": round(statistics.mean(x["cost_usd"] for x in rows), 6),
    }
    idx.close()
    return {"summary": summary, "rows": rows}


def merge(section: str, payload) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    data[section] = payload
    data["updated"] = time.strftime("%Y-%m-%d %H:%M")
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def emb_key(summary: dict) -> str:
    """`jina3` for the production task (the cache reuses the retrieval vector), `jina3:text-matching` for an arm."""
    return summary["embedder"] + ("" if summary["task"] == "retrieval.query" else f":{summary['task']}")


def table(summary: dict) -> str:
    out = ["| threshold | paraphrase hit % | near-miss false hit % | wrong target | false hits |", "|---|---|---|---|---|"]
    for r in summary["coarse"]:
        out.append(f"| {r['threshold']} | {r['paraphrase_hit_rate']} | {r['near_miss_false_hit_rate']} | "
                   f"{r['paraphrase_wrong_target']} | {r['false_hits']} |")
    return "\n".join(out)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["sweep", "latency", "all"])
    ap.add_argument("--embedder", choices=["jina", "local", "both"], default="both")
    ap.add_argument("--task", default="retrieval.query", help="Jina task; the cache itself uses retrieval.query")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--min-score", type=float, help="latency: override ANSWER_CACHE_MIN_SCORE")
    args = ap.parse_args()
    os.environ.setdefault("AI_DATA_DIR", str(ROOT / "data" / "evals"))
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from tulpar_ai.config import get_settings

    get_settings.cache_clear()
    if args.part in ("sweep", "all"):
        kinds = ["jina", "local"] if args.embedder == "both" else [args.embedder]
        prev = json.loads(OUT.read_text(encoding="utf-8")).get("sweep", {}) if OUT.exists() else {}
        for kind in kinds:
            res = await sweep(kind, args.task)
            prev[emb_key(res["summary"])] = res
            print(f"\n{emb_key(res['summary'])}: chosen {res['summary']['chosen_threshold']}\n{table(res['summary'])}")
        merge("sweep", prev)
    if args.part in ("latency", "all"):
        res = await latency(args.n, args.min_score)
        merge("latency", res)
        print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
