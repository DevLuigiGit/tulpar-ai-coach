"""Automated evals and experiments for Tulpar AI Coach.

    python evals/run.py router [--repeats 3] [--route-temperature 0.0]
    python evals/run.py qa [--rerank on|off] [--temperature 0.2] [--top-p 0.9] [--max-tokens 700] [--repeats 1]
    python evals/run.py program                   # deterministic: validator vs golden error codes
    python evals/run.py draft                     # LLM drafts for demo clients → validator pass rate
    python evals/run.py vision --images-dir PATH --manifest PATH [--models ollama:kimi-k2.7-code]
    python evals/run.py experiment NAME           # rag_rerank | answer_temperature | answer_top_p |
                                                  # route_temperature | chunk_size | embedder
Every run writes evals/results/<name>.json (summary + per-item rows) and appends a table to
evals/results/SUMMARY.md. LangSmith tracing is OFF during evals unless --trace (free tier: 5k traces/mo).
Exit code 1 if a suite's gate fails (used by CI).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GOLDEN = ROOT / "evals" / "golden"
RESULTS = ROOT / "evals" / "results"

# Rough $ per 1M tokens (input, output) for cost-per-request estimates. Check prices on the day of the defense.
# Ollama Cloud is a flat subscription: its tokens are priced as the closest Groq-hosted equivalent for comparison.
PRICES = {"groq:llama-3.3-70b-versatile": (0.59, 0.79), "groq:llama-3.1-8b-instant": (0.05, 0.08),
          "ollama:*": (0.59, 0.79), "*": (0.5, 1.0)}


def price(provider: str, model: str, tin: int, tout: int) -> float:
    p = PRICES.get(f"{provider}:{model}") or PRICES.get(f"{provider}:*") or PRICES["*"]
    return (tin * p[0] + tout * p[1]) / 1e6


def load(name: str) -> list[dict]:
    path = GOLDEN / name
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def pct(xs) -> float:
    xs = list(xs)
    return round(100 * sum(xs) / len(xs), 1) if xs else 0.0


def p95(xs) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.95 * len(xs)))] if xs else 0


def usage_stats(calls: list[dict]) -> dict:
    return {"llm_calls": len(calls), "tokens_in": sum(c["in"] for c in calls), "tokens_out": sum(c["out"] for c in calls),
            "cost_usd": round(sum(price(c["provider"], c["model"], c["in"], c["out"]) for c in calls), 6),
            "fallbacks": sum(1 for c in calls if c["fallback"])}


def boot_settings(args) -> None:
    os.environ.setdefault("AI_DATA_DIR", str(ROOT / "data" / "evals"))
    if not getattr(args, "trace", False):
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from tulpar_ai.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    for k in ("route_temperature", "answer_temperature", "answer_top_p", "answer_max_tokens", "rag_rerank"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(s, k, v)


# ── router ───────────────────────────────────────────────────────────────────
async def suite_router(args) -> dict:
    from tulpar_ai import llm
    from tulpar_ai.graph.chat import precheck, route

    cases = load("router.jsonl")[: args.limit or None]
    rows, calls_all = [], []
    for c in cases:
        preds = []
        for _ in range(args.repeats):
            with llm.record() as calls:
                t0 = time.perf_counter()
                st = {"text": c["text"]}
                st.update(await precheck(st))
                out = await route(st)
                ms = int((time.perf_counter() - t0) * 1000)
            calls_all += calls
            preds.append(out["intent"])
        pred = max(set(preds), key=preds.count)
        rows.append({"id": c["id"], "expected": c["expected_intent"], "pred": pred, "all": preds,
                     "stable": len(set(preds)) == 1, "ms": ms, "tags": c.get("tags", []), "text": c["text"][:80]})
    esc = [r for r in rows if r["expected"] == "escalate"]
    non = [r for r in rows if r["expected"] != "escalate"]
    by_tag = {}
    for tag in sorted({t for r in rows for t in r["tags"]}):
        rr = [r for r in rows if tag in r["tags"]]
        by_tag[tag] = {"n": len(rr), "accuracy": pct(r["pred"] == r["expected"] for r in rr)}
    summary = {"n": len(rows), "intent_accuracy": pct(r["pred"] == r["expected"] for r in rows),
               "escalation_recall": pct(r["pred"] == "escalate" for r in esc),
               "false_escalation_rate": pct(r["pred"] == "escalate" for r in non),
               "stability": pct(r["stable"] for r in rows), "p50_ms": statistics.median([r["ms"] for r in rows]),
               "by_tag": by_tag, **usage_stats(calls_all)}
    summary["gate"] = summary["escalation_recall"] >= 90 and summary["intent_accuracy"] >= 75
    return {"summary": summary, "rows": rows}


# ── qa (RAG) ─────────────────────────────────────────────────────────────────
def _hit_rank(hits: list[dict], expected: list[dict]) -> int | None:
    for rank, h in enumerate(hits, 1):
        for e in expected:
            if h["source"] != e["source"]:
                continue
            if e["source"] == "exercises" and h["title"] != e.get("title"):
                continue
            if e["source"] == "who2020" and e.get("page") and abs((h.get("page") or 0) - e["page"]) > 1:
                continue
            return rank
    return None


async def judge(kind: str, **kw) -> int | None:
    from tulpar_ai.llm import LLMError, json_call
    from tulpar_ai.prompts import prompt

    if kind == "faithfulness":
        user = f"Источники:\n{kw['sources']}\n\nОтвет коуча:\n{kw['answer']}"
    else:
        user = f"Вопрос: {kw['question']}\nЭталон: {kw['reference']}\n\nОтвет коуча:\n{kw['answer']}"
    try:
        data, _ = await json_call("judge", prompt(f"judge_{kind}"), user, temperature=0.0, max_tokens=150)
        return int(data.get("score"))
    except (LLMError, TypeError, ValueError):
        return None


async def suite_qa(args) -> dict:
    from tulpar_ai import llm
    from tulpar_ai.config import get_settings
    from tulpar_ai.graph import chat as g
    from tulpar_ai.rag.index import Index
    from tulpar_ai.rag.retrieve import set_index

    s = get_settings()
    idx = Index(pdf_chunk=args.pdf_chunk, path=Path(s.ai_data_dir) / "qdrant")
    await idx.build()
    set_index(idx)
    rows, calls_all = [], []
    for c in load("qa.jsonl")[: args.limit or None]:
        for rep in range(args.repeats):
            with llm.record() as calls:
                t0 = time.perf_counter()
                st: dict = {"text": c["question"], "intent": "question"}
                while True:
                    st.update(await g.retrieve_node(st))
                    nxt = g.after_retrieve(st)
                    if nxt == "rewrite":
                        st.update(await g.rewrite(st))
                        continue
                    break
                first_hits = st.get("hits", [])
                answered = False
                if nxt == "answer":
                    st.update(await g.answer(st))
                    answered = bool(st.get("reply"))
                ms = int((time.perf_counter() - t0) * 1000)
            calls_all += calls
            answer = st.get("reply") or ""
            rank = _hit_rank(first_hits, c["expected_sources"]) if c["answerable"] else None
            low = answer.lower().replace("–", "-")
            keyfacts = all(tok.lower() in low for tok in c["must_include"]) if c["answerable"] and answered else None
            row = {"id": c["id"], "rep": rep, "answerable": c["answerable"], "answered": answered, "rank": rank,
                   "keyfacts": keyfacts, "ms": ms, "rewrites": st.get("rewrites", 0),
                   "out_tokens": sum(x["out"] for x in calls if x["role"] == "text"), "tags": c.get("tags", []),
                   "answer": answer[:400]}
            if answered and args.judge:
                row["faithfulness"] = await judge("faithfulness", sources=g._sources(st["hits"]), answer=answer)
                if c["answerable"]:
                    row["correctness"] = await judge("correctness", question=c["question"],
                                                     reference=c["reference_answer"], answer=answer)
            rows.append(row)
    idx.close()
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    faith = [r["faithfulness"] for r in rows if r.get("faithfulness")]
    corr = [r["correctness"] for r in rows if r.get("correctness")]
    outs = [r["out_tokens"] for r in rows if r["out_tokens"]]
    summary = {
        "n": len(rows), "rerank": s.rag_rerank, "temperature": s.answer_temperature, "top_p": s.answer_top_p,
        "max_tokens": s.answer_max_tokens, "pdf_chunk": args.pdf_chunk, "embedder": idx.embedder.id,
        "hit_at_4": pct(r["rank"] is not None and r["rank"] <= 4 for r in ans),
        "mrr": round(sum(1 / r["rank"] for r in ans if r["rank"]) / (len(ans) or 1), 3),
        "answered_rate": pct(r["answered"] for r in ans),
        "keyfact_accuracy": pct(bool(r["keyfacts"]) for r in ans),
        "correct_refusal_rate": pct(not r["answered"] for r in una),
        "faithfulness_avg": round(statistics.mean(faith), 2) if faith else None,
        "faithfulness_pass": pct(f >= 4 for f in faith) if faith else None,
        "correctness_avg": round(statistics.mean(corr), 2) if corr else None,
        "p50_ms": statistics.median([r["ms"] for r in rows]) if rows else 0, "p95_ms": p95([r["ms"] for r in rows]),
        "out_tokens_p95": p95(outs), **usage_stats(calls_all)}
    summary["cost_per_question_usd"] = round(summary["cost_usd"] / (len(rows) or 1), 6)
    summary["hit_at_4_by_tag"] = {t: pct(r["rank"] is not None and r["rank"] <= 4 for r in ans if t in r["tags"])
                                  for t in sorted({t for r in ans for t in r["tags"]})}
    summary["gate"] = summary["hit_at_4"] >= 70
    return {"summary": summary, "rows": rows}


# ── program (deterministic validator) ────────────────────────────────────────
async def suite_program(args) -> dict:
    from tulpar_ai.skill import catalog, validator

    rows = []
    for c in load("program.jsonl"):
        got = validator().validate(c["plan"], c["ops"], c["client"], catalog())
        codes = sorted({v["code"] for v in got if v["severity"] == "error"})
        rows.append({"id": c["id"], "expected": sorted(c["expect_errors"]), "got": codes,
                     "ok": codes == sorted(c["expect_errors"]), "note": c.get("note")})
    summary = {"n": len(rows), "exact_match": pct(r["ok"] for r in rows)}
    summary["gate"] = summary["exact_match"] == 100
    return {"summary": summary, "rows": rows}


# ── draft (LLM program changes on demo clients) ──────────────────────────────
DRAFT_REQUESTS = [
    ("demo-student", "Убери из программы упражнения, опасные для больного колена, и замени на щадящие"),
    ("demo-student", "Добавь кардио в конец дня ног, 1 упражнение"),
    ("demo-student", "Клиент устаёт: уменьши объём в тяговом дне на один подход"),
    ("demo-student", "Замени жим штанги лёжа на вариант с гантелями"),
    ("demo-student-2", "Тренируется дома: замени всё, что требует тренажёров или штанги"),
    ("demo-student-2", "Добавь упражнение на пресс во второй день"),
    ("demo-student-2", "Хочет больше кардио: добавь по одному кардио-упражнению в каждый день"),
    ("demo-student-2", "Сделай первый день чуть легче для новичка"),
]


async def suite_draft(args) -> dict:
    from tulpar_ai import llm
    from tulpar_ai.gateway import build_gateway, set_gateway
    from tulpar_ai.graph import program as pg
    from tulpar_ai.store import Store, set_store

    data = Path(os.environ["AI_DATA_DIR"])
    (data / "draft.sqlite").unlink(missing_ok=True)
    store = await Store(data / "draft.sqlite").open()
    set_store(store)
    gw = build_gateway(store)
    await gw.start()
    set_gateway(gw)
    trainer = await gw.demo_user("trainer")
    rows, calls_all = [], []
    for tg, req in DRAFT_REQUESTS[: args.limit or None]:
        client = await store.demo_user_by_tg(tg)
        p = await store.create_proposal(kind="program", client_id=client["id"], trainer_id=trainer.id, source="trainer",
                                        request=req, status="drafting")
        with llm.record() as calls:
            t0 = time.perf_counter()
            st = {"proposal_id": p["id"], "client_id": client["id"], "trainer_id": trainer.id, "request": req, "source": "trainer"}
            st.update(await pg.load_context(st))
            first_ok = None
            while True:
                st.update(await pg.draft(st))
                st.update(await pg.validate(st))
                errs = [v for v in st["violations"] if v["severity"] == "error"]
                if first_ok is None:
                    first_ok = not errs
                if pg.after_validate(st) == "publish":
                    break
            ms = int((time.perf_counter() - t0) * 1000)
        calls_all += calls
        errs = [v for v in st["violations"] if v["severity"] == "error"]
        rows.append({"client": tg, "request": req, "ops": len(st["draft"]["ops"]), "drafts": st["draft_attempts"],
                     "first_try_valid": first_ok, "final_valid": not errs and bool(st["draft"]["ops"]),
                     "errors": [e["code"] for e in errs], "warnings": [v["code"] for v in st["violations"] if v["severity"] == "warning"],
                     "summary": st["draft"]["summary"][:200], "ms": ms})
    await store.close()
    summary = {"n": len(rows), "first_try_valid": pct(r["first_try_valid"] for r in rows),
               "final_valid": pct(r["final_valid"] for r in rows),
               "avg_drafts": round(statistics.mean([r["drafts"] for r in rows]), 2) if rows else 0,
               "p50_ms": statistics.median([r["ms"] for r in rows]) if rows else 0, **usage_stats(calls_all)}
    summary["gate"] = summary["final_valid"] >= 75
    return {"summary": summary, "rows": rows}


# ── vision (food photos from Tulpar's labelled bench) ────────────────────────
def _norm(t: str) -> set[str]:
    return {w[:6] for w in re.sub(r"[^0-9a-zа-я]+", " ", (t or "").lower().replace("ё", "е")).split() if len(w) > 1}


def _match(name: str, truths: list[str]) -> bool:
    a = _norm(name)
    for t in truths:
        b = _norm(t)
        if a and b and (a <= b or b <= a or len(a & b) / len(a | b) >= 0.5):
            return True
    return False


async def suite_vision(args) -> dict:
    import base64

    import yaml

    from tulpar_ai import llm
    from tulpar_ai.config import get_settings
    from tulpar_ai.prompts import prompt

    s = get_settings()
    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))["photos"]
    models = [tuple(m.split(":", 1)) for m in args.models.split(",")] if args.models else None
    rows, calls_all = [], []
    for ph in manifest[: args.limit or None]:
        f = Path(args.images_dir) / ph["file"]
        if not f.exists():
            continue
        truths = [ph["truth"], *ph.get("synonyms", [])]
        unrec = ph.get("group") in ("g5", "unrecognisable") or ph["truth"] == "не определено"
        with llm.record() as calls:
            t0 = time.perf_counter()
            try:
                res = await llm.chat("vision", prompt("vision"), "Что на фото?", json_mode=True,
                                     images=[base64.b64encode(f.read_bytes()).decode()], temperature=s.vision_temperature,
                                     max_tokens=s.vision_max_tokens, models=models)
                items = (res.data or {}).get("items", [])
            except llm.LLMError:
                items = None
            ms = int((time.perf_counter() - t0) * 1000)
        calls_all += calls
        names = [i.get("name", "") for i in (items or [])]
        alts = [a for i in (items or []) for a in i.get("alternatives", [])]
        rows.append({"file": ph["file"], "group": ph.get("group"), "truth": ph["truth"], "pred": names[:3],
                     "error": items is None, "refused": items == [],
                     "top1": (items == []) if unrec else bool(names) and _match(names[0], truths),
                     "top3": (items == []) if unrec else any(_match(n, truths) for n in names + alts), "ms": ms})
    groups = sorted({r["group"] for r in rows})
    summary = {"n": len(rows), "models": args.models or s.vision_models, "top1": pct(r["top1"] for r in rows),
               "top3": pct(r["top3"] for r in rows), "errors": sum(r["error"] for r in rows),
               "p50_ms": statistics.median([r["ms"] for r in rows]) if rows else 0,
               "by_group": {g: {"n": len([r for r in rows if r["group"] == g]),
                                "top1": pct(r["top1"] for r in rows if r["group"] == g)} for g in groups},
               **usage_stats(calls_all)}
    summary["gate"] = True
    return {"summary": summary, "rows": rows}


SUITES = {"router": suite_router, "qa": suite_qa, "program": suite_program, "draft": suite_draft, "vision": suite_vision}

EXPERIMENTS = {
    "rag_rerank": ("qa", [{"rag_rerank": True}, {"rag_rerank": False}]),
    "answer_temperature": ("qa", [{"answer_temperature": 0.0, "repeats": 3}, {"answer_temperature": 0.2, "repeats": 3},
                                  {"answer_temperature": 0.7, "repeats": 3}]),
    "answer_top_p": ("qa", [{"answer_top_p": 0.9}, {"answer_top_p": 1.0}]),
    "route_temperature": ("router", [{"route_temperature": 0.0, "repeats": 3}, {"route_temperature": 0.7, "repeats": 3}]),
    "chunk_size": ("qa", [{"pdf_chunk": 400, "judge": False}, {"pdf_chunk": 800, "judge": False}, {"pdf_chunk": 1200, "judge": False}]),
}

KEYS = {"router": ["n", "intent_accuracy", "escalation_recall", "false_escalation_rate", "stability", "p50_ms", "cost_usd"],
        "qa": ["n", "embedder", "rerank", "temperature", "top_p", "pdf_chunk", "hit_at_4", "mrr", "keyfact_accuracy",
               "correct_refusal_rate", "faithfulness_avg", "correctness_avg", "p50_ms", "out_tokens_p95", "cost_per_question_usd"],
        "program": ["n", "exact_match"], "draft": ["n", "first_try_valid", "final_valid", "avg_drafts", "p50_ms", "cost_usd"],
        "vision": ["n", "models", "top1", "top3", "errors", "p50_ms"]}


def table(suite: str, labelled: list[tuple[str, dict]]) -> str:
    keys = KEYS[suite]
    out = ["| arm | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for label, s in labelled:
        out.append(f"| {label} | " + " | ".join(str(s.get(k, "")) for k in keys) + " |")
    return "\n".join(out)


def save(name: str, payload: dict, suite: str, labelled: list[tuple[str, dict]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    stamp = time.strftime("%Y-%m-%d %H:%M")
    with (RESULTS / "SUMMARY.md").open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {name} — {stamp}\n\n{table(suite, labelled)}\n")
    print(table(suite, labelled))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=[*SUITES, "experiment"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rerank", choices=["on", "off"])
    ap.add_argument("--temperature", type=float, dest="answer_temperature")
    ap.add_argument("--top-p", type=float, dest="answer_top_p")
    ap.add_argument("--max-tokens", type=int, dest="answer_max_tokens")
    ap.add_argument("--route-temperature", type=float)
    ap.add_argument("--pdf-chunk", type=int, default=800)
    ap.add_argument("--no-judge", dest="judge", action="store_false")
    ap.add_argument("--images-dir")
    ap.add_argument("--manifest")
    ap.add_argument("--models")
    ap.add_argument("--tag", default="")
    ap.add_argument("--trace", action="store_true")
    return ap


async def main() -> int:
    args = parser().parse_args()
    args.rag_rerank = None if args.rerank is None else args.rerank == "on"
    if args.suite != "experiment":
        boot_settings(args)
        res = await SUITES[args.suite](args)
        name = args.suite + (f"_{args.tag}" if args.tag else "")
        save(name, res, args.suite, [(args.tag or "default", res["summary"])])
        return 0 if res["summary"].get("gate", True) else 1
    suite, arms = EXPERIMENTS[args.name]
    labelled, payload = [], {"experiment": args.name, "arms": []}
    for arm in arms:
        a = parser().parse_args([suite])
        for k in ("limit", "judge", "trace", "images_dir", "manifest", "models", "repeats"):
            setattr(a, k, getattr(args, k))
        a.rag_rerank = None
        for k, v in arm.items():
            setattr(a, k, v)
        boot_settings(a)  # fresh settings per arm: nothing leaks from the previous arm
        res = await SUITES[suite](a)
        label = ", ".join(f"{k}={v}" for k, v in arm.items())
        labelled.append((label, res["summary"]))
        payload["arms"].append({"arm": arm, **res})
    save(f"exp_{args.name}" + (f"_{args.tag}" if args.tag else ""), payload, suite, labelled)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
