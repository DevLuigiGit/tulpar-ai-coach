"""Use cases shared by the HTTP API, the Telegram bot and (through the API) the MCP server."""

from __future__ import annotations

from datetime import date

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from . import feedback, notify
from .gateway import get_gateway
from .gateway.base import MealItem, PlanOp, User
from .graph import runner
from .pii import mask
from .store import get_store

MEALS = {"breakfast", "lunch", "dinner", "snack"}


def _turn_inputs(inputs: dict) -> dict:
    """What the chat_turn trace shows: no raw bytes, no full user id, masked text."""
    user = inputs.get("user")
    return {"client": getattr(user, "id", "")[:8], "text": mask(inputs.get("text") or ""),
            "has_photo": bool(inputs.get("image")), "has_audio": bool(inputs.get("audio"))}


@traceable(run_type="chain", name="chat_turn", process_inputs=_turn_inputs)
async def chat_turn(user: User, text: str = "", image: bytes | None = None, audio: bytes | None = None,
                    audio_name: str = "voice.ogg") -> dict:
    """One client message → graph → stored reply. The reply carries `message_id` so the client can rate it;
    the LangSmith run id (when tracing is on) goes into the stored payload to attach 👍/👎 to the trace."""
    store = get_store()
    shown = text or ("[фото]" if image else "[голосовое]" if audio else "")
    await store.add_message(user.id, "user", shown, {"has_photo": bool(image), "has_audio": bool(audio)})
    res = await runner.run_chat_turn(user.id, text=text, image=image, audio=audio, audio_name=audio_name)
    reply = {
        "reply": res.get("reply") or "…",
        "kind": res.get("kind") or "info",
        "intent": res.get("intent"),
        "citations": res.get("citations") or [],
        "meal": res.get("meal"),
        "proposal_id": res.get("proposal_id"),
        "escalation_id": res.get("escalation_id"),
        "transcript": res.get("transcript"),
    }
    rt = get_current_run_tree()
    payload = {k: v for k, v in reply.items() if k != "reply"} | {"run_id": str(rt.id) if rt else None}
    reply["message_id"] = await store.add_message(user.id, "assistant", reply["reply"], payload)
    return reply


async def rate_message(user: User, message_id: int, rating: str, comment: str | None = None,
                       source: str = "web") -> dict:
    """👍/👎 on the client's own coach answer; also sent to LangSmith when the turn was traced."""
    if rating not in feedback.RATINGS:
        raise ValueError("rating must be up or down")
    store = get_store()
    m = await store.get_message(message_id)
    if m is None or m["client_id"] != user.id or m["role"] != "assistant":
        raise LookupError("message not found")
    payload = m["payload"] or {}
    if payload.get("kind") not in feedback.RATEABLE_KINDS:
        raise ValueError("this message cannot be rated")
    comment = (comment or "").strip()[: feedback.MAX_COMMENT] or None
    row, updated = await store.set_feedback(message_id=message_id, client_id=user.id, rating=rating, comment=comment,
                                            run_id=payload.get("run_id"), source=source)
    row["langsmith"] = await feedback.send_to_langsmith(row["run_id"], message_id, rating, comment, updated)
    return row


async def trainer_feedback(trainer: User, rating: str | None = None, limit: int = 50) -> dict:
    clients = {c.id: c.name for c in await get_gateway().list_clients(trainer.id)}
    ids = list(clients)
    items = await get_store().list_feedback(client_ids=ids, rating=rating, limit=limit)
    for it in items:
        it["client_name"] = clients.get(it["client_id"], "—")
        p = it.pop("payload") or {}
        it["kind"], it["intent"], it["citations"] = p.get("kind"), p.get("intent"), p.get("citations") or []
    return {"counts": await get_store().feedback_counts(ids), "items": items}


async def confirm_meal(user: User, card_id: str, grams: dict[int, float] | None = None, meal: str | None = None) -> dict:
    store, gw = get_store(), get_gateway()
    card = await store.get_meal_card(card_id)
    if card is None or card["client_id"] != user.id:
        raise LookupError("meal card not found")
    if card["status"] == "logged":
        return {"logged": 0, "already": True, "total_kcal": _total(card["items"])}
    items = []
    for i, it in enumerate(card["items"]):
        g = (grams or {}).get(i, it["grams"])
        if g and g > 0:
            items.append(MealItem(**{**{k: it[k] for k in MealItem.model_fields}, "grams": float(g)}))
    meal = meal if meal in MEALS else "snack"
    n = await gw.log_meal(user.id, items, meal, date.today().isoformat(), idem=f"card:{card_id}")
    await store.set_meal_card_status(card_id, "logged")
    total = round(sum(i.kcal * i.grams / 100 for i in items))
    await store.add_message(user.id, "assistant", f"Записал в дневник: {n} поз., ≈ {total} ккал.", {"kind": "meal_logged"})
    return {"logged": n, "already": False, "total_kcal": total}


def _total(items: list[dict]) -> int:
    return round(sum(i["kcal"] * i["grams"] / 100 for i in items))


def describe_ops(before: dict | None, ops: list[dict]) -> list[dict]:
    """Human-readable change list for the trainer: what was → what becomes."""
    cat = {e.id: e for e in get_gateway().exercises()}
    days = {d["day_index"]: d for d in (before or {}).get("days", [])}
    out = []
    for op in ops:
        day = days.get(op.get("day_index"), {})
        wex = next((e for e in day.get("exercises", []) if e["id"] == op.get("wex_id")), None)
        new = cat.get(op.get("exercise_id") or "")
        vol = lambda s, r: f"{s or '?'}×{r or '?'}"  # noqa: E731
        was = f"{wex['exercise_name']} {vol(wex.get('target_sets'), wex.get('target_reps'))}" if wex else None
        if op["op"] == "replace_exercise":
            becomes = f"{new.name if new else op.get('exercise_id')} {vol(op.get('sets') or (wex or {}).get('target_sets'), op.get('reps') or (wex or {}).get('target_reps'))}"
        elif op["op"] == "set_volume":
            becomes = f"{wex['exercise_name'] if wex else '?'} {vol(op.get('sets') or (wex or {}).get('target_sets'), op.get('reps') or (wex or {}).get('target_reps'))}"
        elif op["op"] == "remove_exercise":
            becomes = "убрать"
        else:
            becomes = f"{new.name if new else op.get('exercise_id')} {vol(op.get('sets') or 3, op.get('reps') or 10)}"
        out.append({"op": op["op"], "day": day.get("title", f"день {op.get('day_index')}"), "was": was,
                    "becomes": becomes, "reason": op.get("reason", "")})
    return out


async def proposal_view(pid: str) -> dict | None:
    p = await get_store().get_proposal(pid)
    if p is None:
        return None
    client = await get_gateway().get_user(p["client_id"])
    p["client_name"] = client.name if client else "—"
    if p["kind"] == "program" and p.get("draft"):
        p["changes"] = describe_ops(p.get("before"), p["draft"].get("ops", []))
    return p


async def queue(trainer: User, history: bool = False) -> list[dict]:
    statuses = ["applied", "rejected", "failed", "resolved"] if history else ["pending", "drafting", "open"]
    items = await get_store().list_proposals(trainer_id=trainer.id, statuses=statuses, limit=100)
    out = []
    for p in items:
        v = await proposal_view(p["id"])
        out.append(v)
    return out


async def request_change(trainer: User, client_id: str, request: str) -> dict:
    gw = get_gateway()
    t = await gw.trainer_of(client_id)
    if t is None or t.id != trainer.id:
        raise PermissionError("not your client")
    return await runner.new_program_proposal(client_id, trainer.id, request, source="trainer")


async def decide(trainer: User, pid: str, action: str, comment: str | None = None) -> dict:
    store = get_store()
    p = await store.get_proposal(pid)
    if p is None or p["kind"] != "program":
        raise LookupError("proposal not found")
    if p["trainer_id"] != trainer.id:
        raise PermissionError("not your proposal")
    if p["status"] != "pending":
        raise ValueError(f"proposal is {p['status']}, not pending")
    if action not in ("accept", "reject", "edit"):
        raise ValueError("action must be accept, reject or edit")
    if action == "edit":
        await store.update_proposal(pid, status="drafting", decision={"action": "edit", "comment": comment})
        runner.spawn(runner.resume_program(pid, "edit", comment))
    else:
        await runner.resume_program(pid, action, comment)
    return await proposal_view(pid)


async def resolve_escalation(trainer: User, eid: str, reply: str | None) -> dict:
    store = get_store()
    e = await store.get_proposal(eid)
    if e is None or e["kind"] != "escalation":
        raise LookupError("escalation not found")
    if e["trainer_id"] != trainer.id:
        raise PermissionError("not your client")
    await store.update_proposal(eid, status="resolved", reply=reply or None)
    if reply:
        text = f"Тренер ответил: {reply}"
        await store.add_message(e["client_id"], "assistant", text, {"kind": "trainer_reply"})
        await notify.to_client(e["client_id"], text)
    return await proposal_view(eid)


async def my_proposals(client: User) -> list[dict]:
    return [await proposal_view(p["id"]) for p in await get_store().list_proposals(client_id=client.id, limit=30)
            if p["kind"] == "program"]


def search_exercises(q: str | None, muscle_group: str | None, equipment: str | None, avoid_text: str | None,
                     limit: int = 30) -> list[dict]:
    from .skill import validator

    v = validator()
    hurt = v.body_parts(avoid_text) if avoid_text else set()
    ql = (q or "").lower()
    out = []
    for e in get_gateway().exercises():
        if ql and ql not in e.name.lower():
            continue
        if muscle_group and e.muscle_group != muscle_group:
            continue
        if equipment and e.equipment != equipment:
            continue
        if hurt & v.body_parts(e.contraindications):
            continue
        out.append(e.model_dump())
        if len(out) >= limit:
            break
    return out


async def apply_ops_direct(trainer: User, client_id: str, ops: list[dict]) -> dict:  # used by tests/tools only
    before, after = await get_gateway().apply_ops(client_id, trainer.id, [PlanOp(**o) for o in ops])
    return {"before": before.model_dump(), "after": after.model_dump()}
