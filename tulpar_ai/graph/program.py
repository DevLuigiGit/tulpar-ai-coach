"""Program-change graph: the agent proposes, the trainer decides. Lives for hours between steps.

  load_context ─► draft ─► validate ─┬─► draft            (validator errors, ≤3 drafts)
                                     └─► publish ─► review ⏸ interrupt ─┬─► apply ─► END
                                                                        ├─► draft  (trainer: «поправь», ≤2 rounds)
                                                                        └─► reject ─► END

`review` contains ONLY the interrupt: on resume LangGraph re-runs the interrupted node from its start,
so everything with side effects (LLM calls, writes, notifications) sits in other nodes. `apply` is
idempotent through the store: a replayed resume sees status=applied and does nothing.
"""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .. import notify
from ..config import get_settings
from ..gateway import get_gateway
from ..gateway.base import PlanOp
from ..llm import LLMError, json_call
from ..pii import mask
from ..prompts import prompt
from ..skill import instructions, validator
from ..store import get_store

MAX_DRAFTS = 3
MAX_EDIT_ROUNDS = 2


class ProgramState(TypedDict, total=False):
    proposal_id: str
    client_id: str
    trainer_id: str
    request: str
    source: str
    context: dict
    client: dict
    plan: dict
    candidates: list[dict]
    draft: dict
    violations: list[dict]
    draft_attempts: int
    edit_rounds: int
    feedback: str
    decision: dict
    status: str


def _catalog() -> dict[str, dict]:
    return {e.id: e.model_dump() for e in get_gateway().exercises()}


def pick_candidates(plan: dict, client: dict, catalog: dict[str, dict], per_group: int = 10) -> list[dict]:
    """Exercises the draft may use: allowed equipment, not contraindicated, same muscle groups as the plan."""
    v = validator()
    allowed = v.PLACE_EQUIPMENT.get(client.get("place") or "")
    hurt = v.injured_parts(client)
    groups = {e.get("muscle_group") for d in plan.get("days", []) for e in d.get("exercises", [])} | {"кардио", "кор", "пресс"}
    out: dict[str, list[dict]] = {}
    for e in sorted(catalog.values(), key=lambda e: (not e.get("is_compound"), e["name"])):
        g = e.get("muscle_group")
        if g not in groups:
            continue
        if allowed is not None and e.get("equipment") and e["equipment"] not in allowed:
            continue
        if hurt & v.body_parts(e.get("contraindications")):
            continue
        bucket = out.setdefault(g, [])
        if len(bucket) < per_group:
            bucket.append({"exercise_id": e["id"], "name": e["name"], "muscle_group": g, "equipment": e.get("equipment")})
    return [x for b in out.values() for x in b]


async def load_context(state: ProgramState) -> dict:
    ctx = await get_gateway().client_context(state["client_id"])
    if ctx.active_plan is None:
        await get_store().update_proposal(state["proposal_id"], status="failed", reply="у клиента нет активной программы")
        return {"status": "failed"}
    plan = ctx.active_plan.model_dump()
    client = {**ctx.profile, "trainer_note": ctx.trainer_note}
    await get_store().update_proposal(state["proposal_id"], status="drafting", before=plan)
    return {"context": ctx.for_llm(), "client": client, "plan": plan,
            "candidates": pick_candidates(plan, client, _catalog()), "draft_attempts": 0, "edit_rounds": 0}


def after_load(state: ProgramState) -> str:
    return END if state.get("status") == "failed" else "draft"


def _plan_for_prompt(plan: dict) -> str:
    lines = [f"План «{plan['title']}»"]
    for d in plan["days"]:
        lines.append(f"День day_index={d['day_index']} «{d['title']}»:")
        for e in d["exercises"]:
            lines.append(f"  - wex_id={e['id']} | {e['exercise_name']} | {e.get('muscle_group')} | "
                         f"{e.get('target_sets')}×{e.get('target_reps')}")
    return "\n".join(lines)


async def draft(state: ProgramState) -> dict:
    s = get_settings()
    feedback = []
    errors = [v for v in state.get("violations", []) if v["severity"] == "error"]
    if errors:
        feedback.append("Предыдущий черновик не прошёл проверку:\n" + "\n".join(f"- {v['code']}: {v['message']}" for v in errors))
    if state.get("feedback"):
        feedback.append(f"Комментарий тренера к прошлому черновику: {mask(state['feedback'])}")
    user = "\n\n".join([
        f"Запрос ({'клиент' if state.get('source') == 'client' else 'тренер'}): {mask(state['request'])}",
        "Контекст клиента (JSON): " + json.dumps(state["context"], ensure_ascii=False),
        _plan_for_prompt(state["plan"]),
        "Кандидаты (exercise_id | название | группа | инвентарь):\n" + "\n".join(
            f"- {c['exercise_id']} | {c['name']} | {c['muscle_group']} | {c['equipment']}" for c in state["candidates"]),
        *feedback,
    ])
    system = prompt("draft") + "\n\n# Skill: tulpar-program-builder\n" + instructions()
    try:
        data, _ = await json_call("text", system, user, temperature=s.draft_temperature, top_p=s.draft_top_p,
                                  max_tokens=s.draft_max_tokens)
    except LLMError as e:
        data = {"summary": "Модель недоступна, черновик не собран.", "rationale": str(e)[:200], "ops": []}
    ops = []
    for raw in data.get("ops", []) or []:
        try:
            ops.append(PlanOp(**raw).model_dump())
        except Exception:
            continue  # malformed op → the validator will see it missing and ask for a redraft
    d = {"summary": str(data.get("summary", "")), "rationale": str(data.get("rationale", "")), "ops": ops}
    return {"draft": d, "draft_attempts": state.get("draft_attempts", 0) + 1}


async def validate(state: ProgramState) -> dict:
    violations = validator().validate(state["plan"], state["draft"]["ops"], state["client"], _catalog())
    return {"violations": violations}


def after_validate(state: ProgramState) -> str:
    errors = [v for v in state["violations"] if v["severity"] == "error"]
    return "draft" if errors and state["draft_attempts"] < MAX_DRAFTS else "publish"


async def publish(state: ProgramState) -> dict:
    if not state["draft"]["ops"]:  # nothing to decide on (model down or request not doable) → not the trainer's queue
        await get_store().update_proposal(state["proposal_id"], status="failed", draft=state["draft"],
                                          violations=state["violations"],
                                          reply="черновик не собран: " + (state["draft"]["summary"] or "нет изменений"))
        return {"status": "failed"}
    p = await get_store().update_proposal(state["proposal_id"], status="pending", draft=state["draft"],
                                          violations=state["violations"])
    await notify.proposal_pending(p)
    return {"status": "pending"}


def after_publish(state: ProgramState) -> str:
    return END if state.get("status") == "failed" else "review"


def review(state: ProgramState) -> dict:
    decision = interrupt({"proposal_id": state["proposal_id"], "summary": state["draft"]["summary"]})
    action = (decision or {}).get("action")
    out: dict = {"decision": decision}
    if action == "edit":
        out.update(feedback=decision.get("comment") or "", edit_rounds=state.get("edit_rounds", 0) + 1, draft_attempts=0,
                   violations=[])
    return out


def after_review(state: ProgramState) -> str:
    action = (state.get("decision") or {}).get("action")
    if action == "accept":
        return "apply"
    if action == "edit" and state.get("edit_rounds", 0) <= MAX_EDIT_ROUNDS:
        return "draft"
    return "reject"


async def apply(state: ProgramState) -> dict:
    store = get_store()
    current = await store.get_proposal(state["proposal_id"])
    if current and current["status"] == "applied":  # replayed resume — already done
        return {"status": "applied"}
    ops = [PlanOp(**o) for o in state["draft"]["ops"]]
    try:
        before, after = await get_gateway().apply_ops(state["client_id"], state["trainer_id"], ops)
    except Exception as e:
        await store.update_proposal(state["proposal_id"], status="failed", decision=state.get("decision"),
                                    reply=f"не удалось применить: {e}"[:300])
        return {"status": "failed"}
    await store.update_proposal(state["proposal_id"], status="applied", before=before.model_dump(),
                                after=after.model_dump(), decision=state.get("decision"))
    await notify.to_client(state["client_id"], "Тренер обновил вашу программу: " + state["draft"]["summary"])
    return {"status": "applied"}


async def reject(state: ProgramState) -> dict:
    decision = state.get("decision") or {}
    await get_store().update_proposal(state["proposal_id"], status="rejected", decision=decision)
    if state.get("source") == "client":
        msg = "Тренер посмотрел ваш запрос и решил пока оставить программу без изменений."
        if decision.get("comment"):
            msg += f" Комментарий: {decision['comment']}"
        await notify.to_client(state["client_id"], msg)
    return {"status": "rejected"}


def build_program_graph():
    g = StateGraph(ProgramState)
    for name, fn in [("load_context", load_context), ("draft", draft), ("validate", validate), ("publish", publish),
                     ("review", review), ("apply", apply), ("reject", reject)]:
        g.add_node(name, fn)
    g.add_edge(START, "load_context")
    g.add_conditional_edges("load_context", after_load, ["draft", END])
    g.add_edge("draft", "validate")
    g.add_conditional_edges("validate", after_validate, ["draft", "publish"])
    g.add_conditional_edges("publish", after_publish, ["review", END])
    g.add_conditional_edges("review", after_review, ["apply", "draft", "reject"])
    g.add_edge("apply", END)
    g.add_edge("reject", END)
    return g
