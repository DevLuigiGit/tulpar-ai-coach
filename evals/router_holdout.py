"""Router on the held-out set evals/golden/router_holdout_llm.jsonl (built by evals/build_router_holdout.py).

The set is SYNTHETIC: messages written by deepseek-v4.1-flash, labels confirmed blind by kimi-k2.6. No real clients.
It exists because route.v2 was tuned on router.jsonl, so 100% there is an upper bound, not an estimate.

Reuses evals/run.py unchanged: the same precheck → route path and the same metrics (`suite_router`). run.py's router
suite reads router.jsonl by name, so this script points run.py's `load` at the held-out file for the run.
Two independent passes (repeats=1 each, live LLM): pass 1 is the headline, pass 2 measures stability.

    .venv/bin/python evals/router_holdout.py run          # live: 2 passes (~74 LLM calls), AI_DATA_DIR=data/holdout
    .venv/bin/python evals/router_holdout.py recompute    # no LLM: re-score the saved routings against current labels

Writes evals/results/router_holdout_llm.json (does not append to SUMMARY.md). Next to each headline rate: an exact
(Clopper–Pearson) 95% interval — with 45 rows and 11 escalations a 100% is a range, not a point. The rows the build
dropped (labels disputed or near-duplicates) are routed once as a side diagnostic and never enter the metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HOLDOUT = "router_holdout_llm.jsonl"
OUT = ROOT / "evals" / "results" / "router_holdout_llm.json"
BUILD_LOG = ROOT / "evals" / "results" / "router_holdout_build.json"


def _binom_cdf(k: int, n: int, p: float) -> float:
    from math import comb

    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def _solve(f, target: float) -> float:
    lo, hi = 0.0, 1.0  # f is decreasing in p on [0, 1]
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
    return (lo + hi) / 2


def ci95(k: int, n: int) -> list[float]:
    """Exact Clopper–Pearson 95% interval for k successes out of n, in percent."""
    if n == 0:
        return [0.0, 100.0]
    lower = 0.0 if k == 0 else _solve(lambda p: _binom_cdf(k - 1, n, p), 0.975)  # P(X >= k | p) = 2.5%
    upper = 1.0 if k == n else _solve(lambda p: _binom_cdf(k, n, p), 0.025)  # P(X <= k | p) = 2.5%
    return [round(100 * lower, 1), round(100 * upper, 1)]


def _runner():
    spec = importlib.util.spec_from_file_location("evals_run", ROOT / "evals" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evals_run"] = mod
    spec.loader.exec_module(mod)
    return mod


async def decided_by(text: str) -> tuple[str, str | None]:
    """'rules' when route() answers before calling the model (regex markers, guardrails), else 'llm'.
    Found by running route() with a fake model and checking whether it was called — no routing logic is copied here."""
    from tulpar_ai import llm
    from tulpar_ai.graph.chat import precheck, route

    called = []

    def fake(**kw):
        called.append(1)
        return '{"intent": "other", "red_flag": false, "reason": "probe"}'

    llm.set_fake(fake)
    try:
        st = {"text": text}
        st.update(await precheck(st))
        out = await route(st)
    finally:
        llm.set_fake(None)
    return ("llm", None) if called else ("rules", out.get("intent"))


def confusion(rows: list[dict], key: str) -> dict:
    return {exp: dict(Counter(r[key] for r in rows if r["expected"] == exp)) for exp in sorted({r["expected"] for r in rows})}


def reference() -> dict:
    out = {}
    for name in ("router_llm_v2", "router_final_merged"):
        p = ROOT / "evals" / "results" / f"{name}.json"
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))["summary"]
            out[name] = {k: s.get(k) for k in ("n", "intent_accuracy", "escalation_recall", "false_escalation_rate", "stability")}
    return out


def _rates(rows: list[dict], key: str) -> dict:
    esc = [r for r in rows if r["expected"] == "escalate"]
    non = [r for r in rows if r["expected"] != "escalate"]
    k_acc = sum(r[key] == r["expected"] for r in rows)
    k_rec = sum(r[key] == "escalate" for r in esc)
    k_fer = sum(r[key] == "escalate" for r in non)
    pct = lambda k, n: round(100 * k / n, 1) if n else None  # noqa: E731
    return {"n": len(rows), "intent_accuracy": pct(k_acc, len(rows)), "escalation_recall": pct(k_rec, len(esc)),
            "false_escalation_rate": pct(k_fer, len(non)),
            "counts": {"correct": k_acc, "escalations_caught": k_rec, "n_escalate": len(esc),
                       "false_escalations": k_fer, "n_non_escalate": len(non)},
            "ci95_clopper_pearson": {"intent_accuracy": ci95(k_acc, len(rows)), "escalation_recall": ci95(k_rec, len(esc)),
                                     "false_escalation_rate": ci95(k_fer, len(non))}}


def recompute(out: Path) -> int:
    """Re-score the saved pass-1/pass-2 routings against the CURRENT labels (after a relabel), no model calls."""
    data = json.loads(out.read_text(encoding="utf-8"))
    gold = {json.loads(l)["id"]: json.loads(l) for l in
            (ROOT / "evals" / "golden" / HOLDOUT).read_text(encoding="utf-8").splitlines() if l.strip()}
    changed = []
    for r in data["rows"]:
        g = gold[r["id"]]
        if r["expected"] != g["expected_intent"]:
            changed.append({"id": r["id"], "was": r["expected"], "now": g["expected_intent"]})
        r["expected"], r["tags"] = g["expected_intent"], g["tags"]
    rows = data["rows"]
    fresh = [r for r in rows if not gold[r["id"]].get("paraphrase_of_tuning")]
    esc = [r for r in rows if r["expected"] == "escalate"]
    data["summary"].update({
        **{k: v for k, v in _rates(rows, "pass1").items() if k != "counts"},
        "counts_pass1": _rates(rows, "pass1")["counts"],
        "pass2": {k: v for k, v in _rates(rows, "pass2").items() if k not in ("counts", "ci95_clopper_pearson")},
        "escalation_recall_both_passes": round(100 * sum(r["pass1"] == "escalate" and r["pass2"] == "escalate"
                                                         for r in esc) / len(esc), 1),
        "escalations_missed_in_any_pass": [r["id"] for r in esc if r["pass1"] != "escalate" or r["pass2"] != "escalate"],
        "n_escalate": len(esc),
        "without_tuning_paraphrases": {"pass1": _rates(fresh, "pass1"), "pass2": _rates(fresh, "pass2"),
                                       "excluded": sorted(r["id"] for r in rows if r not in fresh)},
    })
    data["misrouted"] = [{k: r[k] for k in ("id", "expected", "pass1", "pass2", "decided_by", "tags", "text")}
                         for r in rows if r["pass1"] != r["expected"] or r["pass2"] != r["expected"]]
    data.setdefault("relabels", []).extend(c | {"at": time.strftime("%Y-%m-%d %H:%M")} for c in changed)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    sm = data["summary"]
    print(json.dumps({k: sm[k] for k in ("n", "intent_accuracy", "escalation_recall", "false_escalation_rate",
                                          "ci95_clopper_pearson", "pass2", "escalations_missed_in_any_pass")},
                     ensure_ascii=False))
    print("without tuning paraphrases:", json.dumps(sm["without_tuning_paraphrases"]["pass1"], ensure_ascii=False))
    print("relabelled:", changed)
    return 0


async def main() -> int:
    os.environ["AI_DATA_DIR"] = str(ROOT / "data" / "holdout")
    run = _runner()
    args = run.parser().parse_args(["router"])  # repeats=1, default route temperature from settings
    run.boot_settings(args)
    orig_load = run.load
    run.load = lambda name: orig_load(HOLDOUT if name == "router.jsonl" else name)

    from tulpar_ai.config import get_settings
    from tulpar_ai.prompts import active_version

    s = get_settings()
    gold = {c["id"]: c for c in orig_load(HOLDOUT)}
    t0 = time.time()
    res1 = await run.suite_router(args)
    res2 = await run.suite_router(args)
    minutes = round((time.time() - t0) / 60, 1)

    p2 = {r["id"]: r for r in res2["rows"]}
    rows = []
    for r in res1["rows"]:
        who, rule_intent = await decided_by(gold[r["id"]]["text"])
        rows.append({"id": r["id"], "expected": r["expected"], "pass1": r["pred"], "pass2": p2[r["id"]]["pred"],
                     "stable": r["pred"] == p2[r["id"]]["pred"], "decided_by": who, "rule_intent": rule_intent,
                     "tags": r["tags"], "text": gold[r["id"]]["text"]})
    esc = [r for r in rows if r["expected"] == "escalate"]
    missed_esc_any = [r["id"] for r in esc if r["pass1"] != "escalate" or r["pass2"] != "escalate"]
    s1, s2 = res1["summary"], res2["summary"]
    by_decider = {w: {"n": len(rr), "accuracy_pass1": run.pct(r["pass1"] == r["expected"] for r in rr)}
                  for w in ("rules", "llm") for rr in [[r for r in rows if r["decided_by"] == w]]}
    non = [r for r in rows if r["expected"] != "escalate"]
    k_acc = sum(r["pass1"] == r["expected"] for r in rows)
    k_rec = sum(r["pass1"] == "escalate" for r in esc)
    k_fer = sum(r["pass1"] == "escalate" for r in non)
    esc_llm = [r for r in esc if r["decided_by"] == "llm"]
    summary = {
        "n": s1["n"],
        "intent_accuracy": s1["intent_accuracy"],
        "escalation_recall": s1["escalation_recall"],
        "false_escalation_rate": s1["false_escalation_rate"],
        "counts_pass1": {"correct": k_acc, "escalations_caught": k_rec, "n_escalate": len(esc),
                         "false_escalations": k_fer, "n_non_escalate": len(non)},
        "ci95_clopper_pearson": {"intent_accuracy": ci95(k_acc, len(rows)), "escalation_recall": ci95(k_rec, len(esc)),
                                 "false_escalation_rate": ci95(k_fer, len(non))},
        "escalation_recall_llm_decided": {"n": len(esc_llm), "recall": run.pct(r["pass1"] == "escalate" for r in esc_llm)},
        "stability_between_passes": run.pct(r["stable"] for r in rows),
        "pass2": {k: s2[k] for k in ("intent_accuracy", "escalation_recall", "false_escalation_rate")},
        "escalation_recall_both_passes": run.pct(r["pass1"] == "escalate" and r["pass2"] == "escalate" for r in esc),
        "escalations_missed_in_any_pass": missed_esc_any,
        "n_escalate": len(esc),
        "by_tag_pass1": s1["by_tag"],
        "by_decider": by_decider,
        "confusion_pass1": confusion(rows, "pass1"),
        "gate_pass1": s1["gate"],
        "p50_ms": s1["p50_ms"],
        "llm_calls": s1["llm_calls"] + s2["llm_calls"], "fallbacks": s1["fallbacks"] + s2["fallbacks"],
        "cost_usd_estimate": round(s1["cost_usd"] + s2["cost_usd"], 6), "wall_minutes": minutes,
    }
    dropped = []
    if BUILD_LOG.exists():
        from tulpar_ai.graph.chat import precheck, route

        for d in json.loads(BUILD_LOG.read_text(encoding="utf-8"))["rows"]:
            if d.get("kept"):
                continue
            st = {"text": d["text"]}
            st.update(await precheck(st))
            out = await route(st)
            dropped.append({"id": d["id"], "generator_label": d["expected_intent"], "verifier_label": d["verifier"]["intent"],
                            "router": out["intent"], "drop_reason": d["drop_reason"], "text": d["text"]})
    payload = {
        "synthetic": True,
        "disclaimer": "Набор сгенерирован LLM (deepseek-v4.1-flash), метки подтверждены kimi-k2.6. Это не реальные "
                      "сообщения клиентов и не реальные отзывы.",
        "golden": f"evals/golden/{HOLDOUT}",
        "config": {"route_chain": [f"{p}:{m}" for p, m in s.chain("route")], "route_prompt": f"route.{active_version('route')}",
                   "route_temperature": s.route_temperature, "passes": 2, "ai_data_dir": "data/holdout",
                   "run_at": time.strftime("%Y-%m-%d %H:%M")},
        "summary": summary,
        "misrouted": [{k: r[k] for k in ("id", "expected", "pass1", "pass2", "decided_by", "tags", "text")}
                      for r in rows if r["pass1"] != r["expected"] or r["pass2"] != r["expected"]],
        "dropped_rows_diagnostic": {"note": "Не входят в метрики: генератор и проверяющий разошлись в метке или строка — "
                                            "почти дубликат. Один проход маршрутизатора, только для разбора.",
                                    "rows": dropped},
        "reference_tuned_set": reference(),
        "rows": rows,
        "pass1_summary": s1, "pass2_summary": s2,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n", "intent_accuracy", "escalation_recall", "false_escalation_rate",
                                                "ci95_clopper_pearson", "stability_between_passes", "pass2",
                                                "escalation_recall_both_passes", "escalation_recall_llm_decided",
                                                "by_decider")}, ensure_ascii=False))
    for d in dropped:
        print(f"  dropped {d['id']}: gen={d['generator_label']} ver={d['verifier_label']} router={d['router']}")
    for m in payload["misrouted"]:
        print(f"  {m['id']} expected={m['expected']} pass1={m['pass1']} pass2={m['pass2']} [{m['decided_by']}] {m['text'][:90]}")
    print(f"→ {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["run", "recompute"], help="run = live two passes; recompute = re-score saved routings")
    ap.add_argument("--out", help="result file, relative to the repo root (default evals/results/router_holdout_llm.json)")
    a = ap.parse_args()
    if a.out:
        OUT = ROOT / a.out
    raise SystemExit(asyncio.run(main()) if a.cmd == "run" else recompute(OUT))
