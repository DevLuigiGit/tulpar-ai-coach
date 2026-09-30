"""Warm the semantic answer cache of a running service with the questions of the defense, then check the hits.

    python tools/warm_cache.py --base https://coach-production-18ca.up.railway.app [--question "..."]...

The cache is the product's own (rag/answer_cache.py, TTL 72 h): a generic question answered once is answered again
from the cache in ~0.5 s. Personal answers (the «Ваш профиль» source) are never cached, so «сколько калорий мне есть»
stays a live answer. The questions go through the web demo client (demo-login), like a visitor of the demo would ask
them, and stay in its chat history. Two passes: the first one answers and stores, the second one shows the hit.
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

QUESTIONS = [
    "Сколько минут активности в неделю рекомендует ВОЗ?",
    "Как правильно делать приседания со штангой?",
    "Сколько отдыхать между подходами?",
    "Сколько повторений делать для роста мышц?",
    "Нужно ли проходить 10 000 шагов в день?",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--question", action="append", help="instead of the default defense questions")
    args = ap.parse_args()
    questions = args.question or QUESTIONS
    with httpx.Client(base_url=args.base.rstrip("/"), timeout=120) as c:
        token = c.post("/api/auth/demo-login", json={"role": "client"}).raise_for_status().json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        for rnd in ("answer", "cache"):
            print(f"── {rnd} ──")
            for q in questions:
                t0 = time.perf_counter()
                r = c.post("/api/chat", data={"text": q}, headers=headers).raise_for_status().json()
                ms = (time.perf_counter() - t0) * 1000
                cites = ", ".join(x["title"][:30] for x in r.get("citations") or [])
                hit = "" if r.get("cache_score") is None else f" cache {r['cache_score']:.3f}"
                print(f"{ms:7.0f} мс  {r['kind']:9s}{hit:13s} {q[:45]:45s} | {cites}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
