"""Golden sets as LangSmith datasets and eval runs as LangSmith experiments.

    python evals/langsmith_sync.py datasets          # create the datasets from evals/golden (adds missing examples)
    python evals/langsmith_sync.py router            # experiment on tulpar-router and tulpar-router-holdout
    python evals/langsmith_sync.py meal              # experiment on tulpar-meal-text
    python evals/langsmith_sync.py qa                # experiment on tulpar-qa: retrieval + answer, no LLM judge

The same code paths and the same scores as evals/run.py, evals/router_holdout.py and evals/meal_eval.py — the local
JSON files stay the record of the numbers in EVALS.md. Here every example is a traced run you can open, and two
experiments on one dataset can be compared side by side in the LangSmith UI (prompt versions, settings).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

import run  # noqa: E402

DATASETS = {
    "tulpar-router": ("router.jsonl", "Маршрутизатор: куда отправить сообщение клиента и ловит ли он опасные. Настроечный набор."),
    "tulpar-router-holdout": ("router_holdout_llm.jsonl", "Маршрутизатор на сообщениях, которых не было при настройке. "
                                                          "Синтетика: написала LLM, метки перепроверила вторая модель."),
    "tulpar-qa": ("qa.jsonl", "Вопросы к базе знаний: находится ли источник, отвечает ли коуч, передаёт ли тренеру "
                              "вопросы без ответа в корпусе."),
    "tulpar-meal-text": ("meal_text.jsonl", "Еда текстом: найден ли продукт и правдоподобен ли вес «двух ложек»."),
}


def _example(name: str, row: dict) -> tuple[dict, dict, dict]:
    meta = {"id": row["id"], "tags": row.get("tags", [])}
    if name.startswith("tulpar-router"):
        return {"text": row["text"]}, {"intent": row["expected_intent"], "red_flag": bool(row.get("red_flag"))}, meta
    if name == "tulpar-qa":
        return ({"question": row["question"]},
                {k: row.get(k) for k in ("answerable", "reference_answer", "must_include", "expected_sources")}, meta)
    return {"text": row["text"]}, {"expected": row["expected"]}, meta


def sync_datasets(client) -> None:
    existing = {d.name: d for d in client.list_datasets()}
    for name, (file, description) in DATASETS.items():
        ds = existing.get(name) or client.create_dataset(name, description=description)
        have = {(e.metadata or {}).get("id") for e in client.list_examples(dataset_id=ds.id)}
        rows = [r for r in run.load(file) if r["id"] not in have]
        if rows:
            triples = [_example(name, r) for r in rows]
            client.create_examples(inputs=[t[0] for t in triples], outputs=[t[1] for t in triples],
                                   metadata=[t[2] for t in triples], dataset_id=ds.id)
        print(f"{name}: {len(have) + len(rows)} examples ({len(rows)} added)")


# ── router ───────────────────────────────────────────────────────────────────
async def route_target(inputs: dict) -> dict:
    from tulpar_ai.graph import chat as g

    st: dict = {"text": inputs["text"]}
    st.update(await g.precheck(st))
    out = await g.route(st)
    return {"intent": out["intent"], "red_flag": bool(out.get("red_flag")), "reason": out.get("reason", "")}


def intent_correct(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "intent_correct", "score": int(outputs["intent"] == reference_outputs["intent"])}


def escalation(outputs: dict, reference_outputs: dict) -> dict:
    """1 — the verdict about the trainer is right: a dangerous message escalated, a safe one not."""
    sent = outputs["intent"] == "escalate" or outputs["red_flag"]
    return {"key": "escalation_right", "score": int(sent == reference_outputs["red_flag"])}


# ── meal text ────────────────────────────────────────────────────────────────
async def meal_target(inputs: dict) -> dict:
    from tulpar_ai.graph import chat

    wanted = await chat.extract_meal(inputs["text"])
    items, unknown = await chat._resolve_items("eval", wanted)
    return {"card": [{k: it.get(k) for k in ("name", "grams", "grams_source", "measure")} for it in items],
            "unknown": [u["name"] for u in unknown]}


def meal_scores(outputs: dict, reference_outputs: dict) -> list[dict]:
    import meal_eval

    sc = meal_eval.score({"expected": reference_outputs["expected"]}, outputs["card"])
    n = len(sc["per_item"]) or 1
    return [{"key": "product_found", "score": sum(p["found"] is not None for p in sc["per_item"]) / n},
            {"key": "grams_ok", "score": sum(p["ok"] for p in sc["per_item"]) / n}]


# ── qa ───────────────────────────────────────────────────────────────────────
async def qa_target(inputs: dict) -> dict:
    from tulpar_ai.graph import chat as g

    st: dict = {"text": inputs["question"], "intent": "question"}
    while True:
        st.update(await g.retrieve_node(st))
        nxt = g.after_retrieve(st)
        if nxt != "rewrite":
            break
        st.update(await g.rewrite(st))
    hits = st.get("hits", [])
    if nxt == "answer":
        st.update(await g.answer(st))
    return {"answer": st.get("reply") or "", "answered": bool(st.get("reply")),
            "sources": [{"source": h["source"], "title": h["title"], "page": h.get("page"), "id": h.get("id")}
                        for h in hits[:10]]}


def qa_scores(outputs: dict, reference_outputs: dict) -> list[dict]:
    if not reference_outputs["answerable"]:
        return [{"key": "correct_refusal", "score": int(not outputs["answered"])}]
    rank = run._hit_rank(outputs["sources"], reference_outputs["expected_sources"])
    low = outputs["answer"].lower().replace("–", "-")
    keyfacts = outputs["answered"] and all(t.lower() in low for t in reference_outputs["must_include"] or [])
    return [{"key": "answered", "score": int(outputs["answered"])},
            {"key": "hit_at_4", "score": int(rank is not None and rank <= 4)},
            {"key": "keyfacts", "score": int(bool(keyfacts))}]


async def experiment(kind: str, prefix: str) -> None:
    from langsmith import aevaluate

    from tulpar_ai.prompts import active_version

    if kind == "router":
        for ds in ("tulpar-router", "tulpar-router-holdout"):
            await aevaluate(route_target, data=ds, evaluators=[intent_correct, escalation], max_concurrency=4,
                            experiment_prefix=f"{prefix}route-{active_version('route')}",
                            metadata={"route_prompt": active_version("route")})
    elif kind == "meal":
        from tulpar_ai.gateway import set_gateway
        from tulpar_ai.gateway.demo import DemoGateway

        set_gateway(DemoGateway(store=None))
        await aevaluate(meal_target, data="tulpar-meal-text", evaluators=[meal_scores], max_concurrency=4,
                        experiment_prefix=f"{prefix}meal_text-{active_version('meal_text')}",
                        metadata={"meal_text_prompt": active_version("meal_text")})
    else:
        from tulpar_ai.config import get_settings
        from tulpar_ai.rag.index import Index
        from tulpar_ai.rag.retrieve import set_index

        idx = Index(path=Path(get_settings().ai_data_dir) / "qdrant")
        await idx.build()
        set_index(idx)
        try:
            await aevaluate(qa_target, data="tulpar-qa", evaluators=[qa_scores], max_concurrency=4,
                            experiment_prefix=f"{prefix}answer-{active_version('answer')}",
                            metadata={"answer_prompt": active_version("answer"), "collection": idx.collection})
        finally:
            idx.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["datasets", "router", "meal", "qa"])
    ap.add_argument("--prefix", default="", help="experiment name prefix, e.g. «hedge-»")
    args = ap.parse_args()
    run.boot_settings(Namespace(trace=True))  # every example becomes an opened trace in LangSmith
    os.environ["LANGSMITH_TRACING"] = "true"
    from langsmith import Client

    if args.what == "datasets":
        sync_datasets(Client())
        return 0
    asyncio.run(experiment(args.what, args.prefix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
