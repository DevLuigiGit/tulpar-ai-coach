"""Meal text → diary card: is the product found and are the grams plausible, especially for household measures.

    python evals/meal_eval.py [--tag NAME] [--repeats 1] [--trace]
    python evals/meal_eval.py --rescore evals/results/meal_text_X.json [--golden F]   # no calls: saved cards vs golden

Every message of evals/golden/meal_text.jsonl goes the chat graph's own way: chat.extract_meal (the meal_text prompt,
units → grams) → chat._resolve_items (catalog search of the demo gateway, 1305 Tulpar products). A golden item is
found when some card item resolves to a name from `any_of` (or starting with `prefix`); its grams are right when they
fall into `grams` [lo, hi]: ranges, because «пара печенек» has no single true weight.

  product_found   golden items whose product is on the card
  grams_ok        golden items found AND inside their grams range
  grams_informed  the same, but not by luck: the 150 g default that happens to fall into a range does not count
  default_share   card items that fell back to the 150 g default
  extra_items     card items that match no golden item (a second «сахар», a «салат» from «в салат»)

The golden messages were written during development from the user-simulation findings; see evals/golden/README.md.
Writes evals/results/meal_text[_tag].json and appends a table to evals/results/SUMMARY.md, like evals/run.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

import run  # noqa: E402

KEYS = ["n", "items", "product_found", "grams_ok", "grams_informed", "default_share", "estimate_share", "extra_items",
        "errors", "p50_ms"]


def _hit(name: str, exp: dict) -> bool:
    if "any_of" in exp:
        return name in exp["any_of"]
    return name.lower().startswith(exp["prefix"].lower())


def score(case: dict, items: list[dict]) -> dict:
    used: set[int] = set()
    per = []
    for exp in case["expected"]:
        k = next((i for i, it in enumerate(items) if i not in used and _hit(it["name"], exp)), None)
        if k is None:
            per.append({"expected": exp, "found": None, "grams": None, "ok": False})
            continue
        used.add(k)
        g = items[k]["grams"]
        lo, hi = exp["grams"]
        src = items[k].get("grams_source")
        per.append({"expected": exp, "found": items[k]["name"], "grams": g, "source": src, "ok": lo <= g <= hi,
                    "informed": lo <= g <= hi and src != "default"})
    return {"per_item": per, "extra": [items[i]["name"] for i in range(len(items)) if i not in used]}


def summarize(rows: list[dict], errors: int, times: list[float], prompt_version: str) -> dict:
    scored = [r for r in rows if "per_item" in r]
    per = [p for r in scored for p in r["per_item"]]
    card = [it for r in scored for it in r["card"]]
    return {
        "n": len(rows), "items": len(per),
        "product_found": run.pct(p["found"] is not None for p in per),
        "grams_ok": run.pct(p["ok"] for p in per),
        "grams_informed": run.pct(p.get("informed", False) for p in per),
        "default_share": run.pct(it["grams_source"] == "default" for it in card),
        "estimate_share": run.pct(it["grams_source"] == "estimate" for it in card),
        "extra_items": sum(len(r["extra"]) for r in scored),
        "errors": errors,
        "p50_ms": round(sorted(times)[len(times) // 2]) if times else None,
        "prompt": prompt_version,
        "by_tag": {t: run.pct(p["ok"] for r in scored if t in r["tags"] for p in r["per_item"])
                   for t in sorted({t for r in scored for t in r["tags"]})},
    }


def rescore(path: Path, golden: str) -> int:
    """Saved cards against the current golden file: a relabelled range changes the numbers, not the model output."""
    old = json.loads(path.read_text(encoding="utf-8"))
    gold = {c["id"]: c for c in run.load(golden)}
    rows = []
    for r in old["rows"]:
        if "card" in r and r["id"] in gold:
            r = {**r, "tags": gold[r["id"]]["tags"], **score(gold[r["id"]], r["card"])}
        rows.append(r)
    summary = {**summarize(rows, old["summary"].get("errors", 0), [], old["summary"].get("prompt", "")),
               "p50_ms": old["summary"].get("p50_ms"), "rescored_from": str(path.relative_to(ROOT))}
    path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in KEYS}, ensure_ascii=False))
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--rescore", help="saved result file to re-score without calls")
    ap.add_argument("--golden", default="meal_text.jsonl", help="golden file in evals/golden (meal_text_holdout.jsonl: "
                    "written after the prompt change, never used to tune it)")
    args = ap.parse_args()
    if args.rescore:
        return rescore(ROOT / args.rescore, args.golden)
    run.boot_settings(Namespace(trace=args.trace))

    from tulpar_ai.gateway import set_gateway
    from tulpar_ai.gateway.demo import DemoGateway
    from tulpar_ai.graph import chat
    from tulpar_ai.prompts import active_version

    set_gateway(DemoGateway(store=None))  # the catalog search reads fixtures only
    cases = run.load(args.golden)
    rows, times, errors = [], [], 0
    for rep in range(args.repeats):
        for c in cases:
            t0 = time.perf_counter()
            try:
                wanted = await chat.extract_meal(c["text"])
                items, unknown = await chat._resolve_items("eval", wanted)
            except Exception as e:  # noqa: BLE001 — a failed case is a result, not a crash
                errors += 1
                rows.append({"id": c["id"], "rep": rep, "text": c["text"], "error": repr(e)[:200]})
                continue
            times.append((time.perf_counter() - t0) * 1000)
            sc = score(c, items)
            rows.append({"id": c["id"], "rep": rep, "text": c["text"], "tags": c["tags"], "wanted": wanted,
                         "card": [{k: it.get(k) for k in ("name", "grams", "grams_source", "measure")} for it in items],
                         "unknown": [u["name"] for u in unknown], **sc})
            ok = sum(p["ok"] for p in sc["per_item"])
            print(f"{c['id']} {ok}/{len(sc['per_item'])}  {c['text'][:40]:40s} → "
                  + ", ".join(f"{it['name']} {it['grams']:g} г ({it.get('grams_source')})" for it in items))

    summary = summarize(rows, errors, times, f"meal_text {active_version('meal_text')}")
    name = "meal_text" + (f"_{args.tag}" if args.tag else "")
    run.KEYS["meal_text"] = KEYS
    run.save(name, {"summary": summary, "rows": rows}, "meal_text", [(args.tag or "meal_text", summary)])
    print(json.dumps(summary["by_tag"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
