"""Tools of the program-draft agent, on top of the gateway.

`find_exercises` is the ONE catalogue search: `/api/exercises` (and through it the MCP tool `search_exercises`
in a trainer's Claude Desktop) and the draft agent in the graph filter the catalogue the same way. The agent's
versions additionally apply the client's own restrictions (equipment of the place, body parts from the trainer's
note), so a model that forgets them still cannot pick a forbidden exercise.

Every agent tool is a LangSmith run of type `tool` named after the tool: the trace of a draft shows what the
agent looked up and what the validator told it.
"""

from __future__ import annotations

import re

from langsmith import traceable

from .gateway import get_gateway
from .gateway.base import Exercise, PlanOp
from .skill import validator

SEARCH_LIMIT = 15  # rows per agent search: enough to choose from, small enough for the next prompt


def _norm(text: str | None) -> str:
    return (text or "").lower().replace("ё", "е").strip()


def _stem(word: str) -> str:
    # «тяги» ~ «тяга», «гантелями» ~ «гантелей»: compare word starts, not whole words
    return word[: max(3, min(5, len(word) - 1))]


def name_matches(query: str | None, name: str) -> bool:
    """Substring (the old behaviour of /api/exercises), or every query word starts some word of the name."""
    q, n = _norm(query), _norm(name)
    if not q or q in n:
        return True
    stems = [_stem(w) for w in re.findall(r"[0-9a-zа-я]+", q) if len(w) >= 3]
    words = re.findall(r"[0-9a-zа-я]+", n)
    return bool(stems) and all(any(w.startswith(s) for w in words) for s in stems)


def _group(g: str | None) -> str | None:
    g = _norm(g) or None
    return validator().GROUP_ALIASES.get(g, g) if g else None


def find_exercises(query: str | None = None, muscle_group: str | None = None, equipment: str | None = None,
                   avoid_text: str | None = None, *, allowed_equipment: set[str] | None = None,
                   hurt: set[str] | None = None) -> list[Exercise]:
    """Catalogue rows in catalogue order. `avoid_text`/`hurt`: body parts whose contraindicated exercises are dropped;
    `allowed_equipment`: exercises that need anything else are dropped (unknown equipment passes, as in the validator)."""
    v = validator()
    parts = set(hurt or ()) | (v.body_parts(avoid_text) if avoid_text else set())
    group, equipment = _group(muscle_group), _norm(equipment) or None
    out = []
    for e in get_gateway().exercises():
        if query and not name_matches(query, e.name):
            continue
        if group and _group(e.muscle_group) != group:
            continue
        if equipment and e.equipment != equipment:
            continue
        if allowed_equipment is not None and e.equipment and e.equipment not in allowed_equipment:
            continue
        if parts & v.body_parts(e.contraindications):
            continue
        out.append(e)
    return out


def client_limits(client: dict) -> tuple[set[str] | None, set[str]]:
    """(equipment allowed at the client's place or None for any, injured body parts from the trainer's note)."""
    v = validator()
    return v.PLACE_EQUIPMENT.get(client.get("place") or ""), v.injured_parts(client)


# ── agent tools ──────────────────────────────────────────────────────────────
def _search_inputs(inputs: dict) -> dict:
    client = inputs.get("client") or {}
    allowed, hurt = client_limits(client)
    return {k: inputs.get(k) for k in ("query", "muscle_group", "equipment")} | {
        "place": client.get("place"), "avoid": sorted(hurt)}


@traceable(run_type="tool", name="search_exercises", process_inputs=_search_inputs)
def search_exercises(query: str = "", muscle_group: str = "", equipment: str = "", *, client: dict,
                     limit: int = SEARCH_LIMIT) -> list[dict]:
    allowed, hurt = client_limits(client)
    rows = find_exercises(query, muscle_group, equipment, allowed_equipment=allowed, hurt=hurt)
    rows.sort(key=lambda e: (not e.is_compound, e.name))
    return [{"exercise_id": e.id, "name": e.name, "muscle_group": e.muscle_group, "equipment": e.equipment,
             "compound": e.is_compound} for e in rows[:limit]]


def _context_inputs(inputs: dict) -> dict:
    return {"client": str(inputs.get("client_id") or "")[:8]}


@traceable(run_type="tool", name="get_client_context", process_inputs=_context_inputs)
async def get_client_context(client_id: str) -> dict:
    """What the model may see about the client: no name, no ids of people (ClientContext.for_llm)."""
    return (await get_gateway().client_context(client_id)).for_llm()


def normalize_ops(raw_ops) -> tuple[list[dict], list[str]]:
    """PlanOp-shaped dicts + a note per op that could not be read (the model is told, not silently ignored)."""
    ops, bad = [], []
    for i, raw in enumerate(raw_ops if isinstance(raw_ops, list) else []):
        try:
            ops.append(PlanOp(**raw).model_dump())
        except Exception as e:  # noqa: BLE001 — any shape error is reported back to the model
            bad.append(f"операция {i} не распознана: {type(e).__name__}")
    return ops, bad


def _check_inputs(inputs: dict) -> dict:
    return {"ops": inputs.get("ops"), "place": (inputs.get("client") or {}).get("place")}


@traceable(run_type="tool", name="check_plan", process_inputs=_check_inputs)
def check_plan(plan: dict, ops: list[dict], client: dict) -> dict:
    """skills/tulpar-program-builder/scripts/validate_plan.py on the draft — the same check as the graph's validate."""
    catalog = {e.id: e.model_dump() for e in get_gateway().exercises()}
    violations = validator().validate(plan, ops, client, catalog)
    short = lambda v: {"code": v["code"], "message": v["message"], "op": v.get("op_index")}  # noqa: E731
    errors = [short(v) for v in violations if v["severity"] == "error"]
    return {"ok": not errors, "errors": errors, "warnings": [short(v) for v in violations if v["severity"] != "error"]}
