"""Demo mode: the Tulpar world is a set of fixtures exported from Tulpar's seeds.

Plans and the food diary are mutable and live in this service's SQLite, so accepting a trainer's
proposal really changes what the client sees next time. The two demo clients (Айдар, Дана) are shown as they are.
Every other client — a Telegram user or a web «Новый клиент» guest — is a person of their own: they start with an
empty profile and a starter plan, and fill in a short questionnaire on first launch (`needs_onboarding`). Their
answers become the profile the coach counts norms and checks restrictions with.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from datetime import date, timedelta

from ..config import ROOT
from ..skill import validator
from ..store import Store
from .base import (ClientContext, ClientSummary, Exercise, Food, MealItem, Plan, PlanOp, User, rank_foods)

FIX = ROOT / "fixtures"
PROFILE_FIELDS = ("sex", "age", "height_cm", "weight_kg", "goal", "level", "place", "activity")


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
        new_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"tg-client:{telegram_id}"))
        return await self._new_client(new_id, name or "Клиент", telegram_id)

    async def guest_user(self, key: str) -> User:
        """A web visitor who chose «Новый клиент»: the browser keeps a random key, the same key is the same person."""
        tg = "guest:" + hashlib.sha256(key.encode()).hexdigest()[:24]
        row = await self.store.demo_user_by_tg(tg)
        if row:
            return self._u(row)
        return await self._new_client(str(uuid.uuid5(uuid.NAMESPACE_URL, f"guest-client:{tg}")), "Гость", tg)

    def _template(self, place: str) -> dict:
        return next((c for c in self._demo["clients"] if c["active_plan"].get("meta", {}).get("place") == place),
                    self._demo["clients"][0])

    def _plan_copy(self, place: str, owner_id: str) -> dict:
        """The demo plan for this place, with ids of its own: accepting a change for one client touches no other."""
        plan = copy.deepcopy(self._template(place)["active_plan"])
        plan["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"plan:{owner_id}:{place}"))
        for d in plan["days"]:
            d["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"day:{owner_id}:{place}:{d['day_index']}"))
            for i, e in enumerate(d["exercises"]):
                e["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"wex:{owner_id}:{place}:{d['day_index']}:{i}"))
        return plan

    async def _new_client(self, client_id: str, name: str, telegram_id: str) -> User:
        """A new person: no profile yet (the questionnaire fills it), no history, the gym starter plan."""
        profile = {k: None for k in PROFILE_FIELDS} | {"onboarded": False, "trainer_note": None, "sessions": [],
                                                        "weights": [], "restrictions": []}
        await self.store.upsert_demo_user({"id": client_id, "role": "client", "name": name, "telegram_id": telegram_id,
                                           "trainer_id": self._demo["trainer"]["id"], "profile": profile})
        await self.store.put_demo_plan(client_id, self._plan_copy("gym", client_id))
        return self._u(await self.store.demo_user(client_id))

    # ── first launch: the questionnaire ──────────────────────────────────────
    @staticmethod
    def _is_demo(row: dict) -> bool:
        return str(row.get("telegram_id") or "").startswith("demo-")

    async def needs_onboarding(self, user: User) -> bool:
        if user.role != "client":
            return False
        row = await self.store.demo_user(user.id)
        return bool(row) and not self._is_demo(row) and not row["profile"].get("onboarded")

    async def client_profile(self, client_id: str) -> dict:
        row = await self.store.demo_user(client_id)
        if row is None:
            raise LookupError("client not found")
        p = row["profile"]
        return {"name": row["name"], **{k: p.get(k) for k in PROFILE_FIELDS},
                "limitations": p.get("limitations") or "", "onboarded": bool(p.get("onboarded")) or self._is_demo(row)}

    async def update_profile(self, client_id: str, name: str, fields: dict) -> None:
        """Save the questionnaire. The first time, whatever a copy of the demo client brought along (Telegram users
        before onboarding existed got Айдар's knee note, weights, sessions and diary) is cleared: it is not theirs."""
        row = await self.store.demo_user(client_id)
        if row is None or row["role"] != "client":
            raise LookupError("client not found")
        p = dict(row["profile"])
        old_place = p.get("place") or "gym"  # a new client starts with the gym plan
        if not p.get("onboarded") and not self._is_demo(row):
            p.update({"trainer_note": None, "sessions": [], "weights": [], "goal_weight_kg": None, "training_days": None})
            await self.store.delete_seed_diary(client_id)
        limitations = (fields.pop("limitations", "") or "").strip()
        p.update({k: fields[k] for k in PROFILE_FIELDS if k in fields})
        p.update({"limitations": limitations, "restrictions": [limitations] if limitations else [], "onboarded": True})
        if fields.get("weight_kg"):  # the questionnaire weight is a measurement too: the trainer sees it in the card
            p["weights"] = [*(p.get("weights") or []), {"measured_at": date.today().isoformat(),
                                                        "weight_kg": float(fields["weight_kg"])}][-30:]
        await self.store.upsert_demo_user({**row, "name": name or row["name"], "profile": p})
        # Another place means another starter plan. Compared with the previous answer, not with the plan: an accepted
        # change drops the plan's meta, and the trainer's changes must survive a save of the same place.
        if fields.get("place") and fields["place"] != old_place:
            await self.store.put_demo_plan(client_id, self._plan_copy(fields["place"], client_id))

    async def list_clients(self, trainer_id: str) -> list[ClientSummary]:
        out = []
        for row in await self.store.demo_clients(trainer_id):
            p = row["profile"]
            plan = await self.store.get_demo_plan(row["id"])
            out.append(ClientSummary(id=row["id"], name=row["name"], goal=p.get("goal"), level=p.get("level"),
                                     place=p.get("place"),
                                     injury=bool((p.get("trainer_note") or {}).get("injury") or p.get("limitations")),
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
