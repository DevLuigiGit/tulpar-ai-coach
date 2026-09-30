"""Streaming benchmark: how soon the first words of an answer reach the client, against a running service.

    python evals/stream_bench.py --url http://127.0.0.1:PORT [--n 10] [--tag NAME]
Start the service yourself (`python -m tulpar_ai` with ANSWER_CACHE=false, RATE_LIMIT_ENABLED=false,
TELEGRAM_BOT_TOKEN= empty): with the cache on, the second arm would get the first arm's answer from the cache.
Each question from evals/golden/qa.jsonl (first N answerable, file order) goes once through POST /api/chat/stream and
once through POST /api/chat, alternating which goes first. Measured on the client, in ms:
  stream: first stage event, first answer text (delta), full reply (done); plain: full reply.
The user of /api/chat sees nothing until the full reply, so its total is the time to the first words there.
Writes evals/results/stream[_tag].json and appends a table to evals/results/SUMMARY.md, like evals/run.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import run  # noqa: E402

run.KEYS["stream"] = ["n", "answers", "stream_first_stage_p50_ms", "stream_first_delta_p50_ms",
                      "stream_first_delta_p95_ms", "stream_total_p50_ms", "plain_total_p50_ms",
                      "wait_saved_p50_ms", "seamless", "same_kind", "errors"]
TIMEOUT = httpx.Timeout(120.0, connect=10.0)


def pick_questions(n: int) -> list[dict]:
    rows = [r for r in run.load("qa.jsonl") if r.get("answerable")]
    return [{"id": r["id"], "question": r["question"]} for r in rows[:n]]


async def stream_turn(c: httpx.AsyncClient, h: dict, text: str) -> dict:
    t0 = time.perf_counter()
    ms = lambda: round((time.perf_counter() - t0) * 1000, 1)  # noqa: E731
    first_stage = first_delta = None
    deltas: list[str] = []
    stages: list[str] = []
    done = None
    async with c.stream("POST", "/api/chat/stream", data={"text": text}, headers=h) as r:
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {(await r.aread()).decode()[:200]}", "total_ms": ms()}
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            if ev["type"] == "stage":
                stages.append(ev["stage"])
                first_stage = first_stage if first_stage is not None else ms()
            elif ev["type"] == "delta":
                deltas.append(ev["text"])
                first_delta = first_delta if first_delta is not None else ms()
            elif ev["type"] == "done":
                done = ev
            elif ev["type"] == "error":
                return {"error": ev.get("message"), "total_ms": ms()}
    total = ms()
    if done is None:
        return {"error": "stream ended without done", "total_ms": total}
    return {"first_stage_ms": first_stage, "first_delta_ms": first_delta, "total_ms": total, "kind": done["kind"],
            "stages": stages, "deltas": len(deltas), "reply_chars": len(done["reply"]),
            "first_delta_chars": len(deltas[0]) if deltas else 0,
            "seamless": "".join(deltas) == done["reply"] if deltas else None, "guard": done.get("guard")}


async def plain_turn(c: httpx.AsyncClient, h: dict, text: str) -> dict:
    t0 = time.perf_counter()
    r = await c.post("/api/chat", data={"text": text}, headers=h)
    total = round((time.perf_counter() - t0) * 1000, 1)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}", "total_ms": total}
    d = r.json()
    return {"total_ms": total, "kind": d["kind"], "reply_chars": len(d["reply"]), "guard": d.get("guard")}


def med(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 1) if xs else None


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["stream"].get("error") and not r["plain"].get("error")]
    answered = [r for r in ok if r["stream"]["first_delta_ms"] is not None]
    fd = [r["stream"]["first_delta_ms"] for r in answered]
    saved = [r["plain"]["total_ms"] - r["stream"]["first_delta_ms"] for r in answered]
    return {
        "n": len(rows), "answers": len(answered),
        "errors": sum(bool(r["stream"].get("error")) + bool(r["plain"].get("error")) for r in rows),
        "stream_first_stage_p50_ms": med([r["stream"]["first_stage_ms"] for r in ok]),
        "stream_first_delta_p50_ms": med(fd), "stream_first_delta_p95_ms": run.p95(fd) if fd else None,
        "stream_first_delta_min_ms": min(fd) if fd else None, "stream_first_delta_max_ms": max(fd) if fd else None,
        "stream_total_p50_ms": med([r["stream"]["total_ms"] for r in ok]),
        "stream_total_answers_p50_ms": med([r["stream"]["total_ms"] for r in answered]),
        "plain_total_p50_ms": med([r["plain"]["total_ms"] for r in ok]),
        "plain_total_answers_p50_ms": med([r["plain"]["total_ms"] for r in answered]),
        "plain_total_p95_ms": run.p95([r["plain"]["total_ms"] for r in ok]) if ok else None,
        "stream_total_p95_ms": run.p95([r["stream"]["total_ms"] for r in ok]) if ok else None,
        "wait_saved_p50_ms": med(saved),
        "seamless": run.pct(r["stream"]["seamless"] for r in answered),
        "same_kind": run.pct(r["stream"]["kind"] == r["plain"]["kind"] for r in ok),
        "kinds_stream": dict(Counter(r["stream"]["kind"] for r in ok)),
        "kinds_plain": dict(Counter(r["plain"]["kind"] for r in ok)),
    }


async def bench(url: str, n: int) -> dict:
    questions = pick_questions(n)
    async with httpx.AsyncClient(base_url=url, timeout=TIMEOUT) as c:
        token = (await c.post("/api/auth/demo-login", json={"role": "client"})).json()["token"]
        h = {"Authorization": f"Bearer {token}"}
        await plain_turn(c, h, "Привет")  # warm-up: connection pools, embedder, index
        rows = []
        for i, q in enumerate(questions):
            arms = [("stream", stream_turn), ("plain", plain_turn)]
            if i % 2:
                arms.reverse()  # alternate the order so provider warm-up and drift hit both arms equally
            row = {"id": q["id"], "first": arms[0][0]}
            for name, fn in arms:
                try:
                    row[name] = await fn(c, h, q["question"])
                except Exception as e:  # a failed call is a data point, not the end of the run
                    row[name] = {"error": f"{type(e).__name__}: {e}"[:200], "total_ms": None}
            s, p = row["stream"], row["plain"]
            print(f"{q['id']}: stream first stage {s.get('first_stage_ms')} ms, first words {s.get('first_delta_ms')} ms, "
                  f"done {s.get('total_ms')} ms ({s.get('kind') or s.get('error')}); "
                  f"plain {p.get('total_ms')} ms ({p.get('kind') or p.get('error')})", flush=True)
            rows.append(row)
    return {"summary": summarize(rows), "rows": rows}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tag", default="")
    return ap


async def main() -> int:
    args = parser().parse_args()
    res = await bench(args.url, args.n)
    name = "stream" + (f"_{args.tag}" if args.tag else "")
    run.save(name, res, "stream", [(args.tag or "default", res["summary"])])
    return 0 if res["summary"]["answers"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
