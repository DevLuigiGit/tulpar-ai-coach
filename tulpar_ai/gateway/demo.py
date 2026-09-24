"""Demo mode: the Tulpar world is a set of fixtures exported from Tulpar's seeds.

Plans and the food diary are mutable and live in this service's SQLite, so accepting a trainer's
proposal really changes what the client sees next time. Each Telegram tester gets a personal copy of
the demo client (so two people testing the bot do not overwrite each other's plan).
"""

from __future__ import annotations

import copy
import json
import uuid
from datetime import date, timedelta

from ..config import ROOT
from ..skill import validator
from ..store import Store
from .base import (ClientContext, ClientSummary, Exercise, Food, MealItem, Plan, PlanOp, User, rank_foods)

FIX = ROOT / "fixtures"


class DemoGateway:
    mode = "demo"

    def __init__(self, store: Store):
        self.store = store
        self._exercises = [Exercise(**{k: e.get(k) for k in Exercise.model_fields}) for e in _load("exercises.json")]
        self._catalog = {e.id: e.model_dump() for e in self._exercises}
        self._foods = _load("foods.json")
        self._demo = _load("demo.json")

    async def start(self) -> None:
        if await self.store.demo_user_count() > 0:
            return
        tr = self._demo["trainer"]
        await self.store.upsert_demo_user({"id": tr["id"], "role": "trainer", "name": tr["name"],
                                           "telegram_id": tr["telegram_id"], "profile": {"gym": tr.get("gym")}})
        for c in self._demo["clients"]:
            await self._seed_client(c, trainer_id=tr["id"])

    async def close(self) -> None:
        return None

    async def _seed_client(self, c: dict, trainer_id: str) -> None:
        profile = {k: c.get(k) for k in ("sex", "age", "height_cm", "goal", "level", "place", "training_days",
                                          "activity", "weight_kg", "goal_weight_kg")}
        profile.update({"trainer_note": c.get("trainer_note"), "sessions": c.get("sessions", []),
                        "weights": c.get("weights", [])})
        await self.store.upsert_demo_user({"id": c["id"], "role": "client", "name": c["name"],
                                           "telegram_id": c.get("telegram_id"), "trainer_id": trainer_id, "profile": profile})
        await self.store.put_demo_plan(c["id"], c["active_plan"])
        await self.store.add_demo_diary(c["id"], c.get("diary", []), idem=f"seed:{c['id']}")

    # ── users ────────────────────────────────────────────────────────────────
    def _u(self, row: dict) -> User:
        return User(id=row["id"], role=row["role"], name=row["name"], telegram_id=row.get("telegram_id"))

    async def demo_user(self, role: str) -> User:
        tg = "demo-trainer" if role == "trainer" else "demo-student"
        return self._u(await self.store.demo_user_by_tg(tg))

    async def get_user(self, user_id: str) -> User | None:
        row = await self.store.demo_user(user_id)
        return self._u(row) if row else None

    async def telegram_user(self, telegram_id: str, name: str, as_trainer: bool) -> User:
        if as_trainer:
            return await self.demo_user("trainer")
        row = await self.store.demo_user_by_tg(telegram_id)
        if row:
            return self._u(row)
        template = copy.deepcopy(self._demo["clients"][0])
        new_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"tg-client:{telegram_id}"))
        template.update({"id": new_id, "name": name or "Клиент", "telegram_id": telegram_id})
        plan = template["active_plan"]
        plan["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"plan:{new_id}"))
        for d in plan["days"]:
            d["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"day:{new_id}:{d['day_index']}"))
            for i, e in enumerate(d["exercises"]):
                e["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"wex:{new_id}:{d['day_index']}:{i}"))
        await self._seed_client(template, trainer_id=self._demo["trainer"]["id"])
        return self._u(await self.store.demo_user(new_id))

    async def list_clients(self, trainer_id: str) -> list[ClientSummary]:
        out = []
        for row in await self.store.demo_clients(trainer_id):
            p = row["profile"]
            plan = await self.store.get_demo_plan(row["id"])
            out.append(ClientSummary(id=row["id"], name=row["name"], goal=p.get("goal"), level=p.get("level"),
                                     place=p.get("place"), injury=bool((p.get("trainer_note") or {}).get("injury")),
                                     active_plan_title=plan["title"] if plan else None))
        return out

    async def trainer_of(self, client_id: str) -> User | None:
        row = await self.store.demo_user(client_id)
        return await self.get_user(row["trainer_id"]) if row and row.get("trainer_id") else None

    # ── context & plan ───────────────────────────────────────────────────────
    async def client_context(self, client_id: str) -> ClientContext:
        row = await self.store.demo_user(client_id)
        if row is None or row["role"] != "client":
            raise LookupError("client not found")
        p = dict(row["profile"])
        note = p.pop("trainer_note", None)
        sessions = p.pop("sessions", [])
        weights = p.pop("weights", [])
        since = (date.fromisoformat(self._demo["today"]) - timedelta(days=14)).isoformat()
        diary = await self.store.demo_diary(client_id, since)
        plan = await self.active_plan(client_id)
        return ClientContext(client_id=client_id, name=row["name"], profile=p, trainer_note=note, active_plan=plan,
                             nutrition_14d=summarise_diary(diary), recent_sessions=sessions, weights=weights)

    async def active_plan(self, client_id: str) -> Plan | None:
        raw = await self.store.get_demo_plan(client_id)
        return Plan(**raw) if raw else None

    async def apply_ops(self, client_id: str, trainer_id: str, ops: list[PlanOp]) -> tuple[Plan, Plan]:
        before = await self.active_plan(client_id)
        if before is None:
            raise LookupError("client has no active plan")
        after_raw, errors = validator().apply_ops(before.model_dump(), [o.model_dump() for o in ops], self._catalog)
        if errors:
            raise ValueError("; ".join(e["message"] for e in errors))
        await self.store.put_demo_plan(client_id, after_raw)
        return before, Plan(**after_raw)

    async def restore_plan(self, client_id: str, trainer_id: str, before: Plan) -> None:
        await self.store.put_demo_plan(client_id, before.model_dump())

    def exercises(self) -> list[Exercise]:
        return self._exercises

    # ── food ─────────────────────────────────────────────────────────────────
    async def search_foods(self, client_id: str, query: str, limit: int = 5) -> list[Food]:
        return rank_foods(query, self._foods, limit)

    async def log_meal(self, client_id: str, items: list[MealItem], meal: str, on_date: str, idem: str) -> int:
        rows = [{**i.model_dump(), "meal": meal, "on_date": on_date} for i in items]
        return await self.store.add_demo_diary(client_id, rows, idem=idem)


def summarise_diary(rows: list[dict]) -> dict:
    """Per-day totals from per-100 g snapshots: value * grams / 100."""
    days: dict[str, dict] = {}
    for r in rows:
        d = days.setdefault(r["on_date"], {"kcal": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0})
        k = (r["grams"] or 0) / 100
        for f in d:
            d[f] += (r[f] or 0) * k
    n = len(days)
    if not n:
        return {"days_logged": 0}
    return {"days_logged": n, **{f"avg_{f}": round(sum(d[f] for d in days.values()) / n) for f in ("kcal", "protein", "fat", "carbs")}}


def _load(name: str):
    return json.loads((FIX / name).read_text(encoding="utf-8"))
