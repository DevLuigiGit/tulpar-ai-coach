"""Real mode: a LOCAL Tulpar stand behind the same Gateway interface.

Reads: Tulpar HTTP API where it serves the trainer's view (plans), read-only SQL where the API has no
trainer-scoped endpoint (client goals, 14-day macros, weights). Writes: only through Tulpar's own API
(plan edits, diary), with short-lived tokens minted from that stand's JWT secret — the same way Tulpar's
QA tooling does. Every SQL connection is opened READ ONLY.

config.get_settings() refuses production-looking URLs, so this cannot be pointed at the live product
by accident.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import asyncpg
import httpx
import jwt

from ..config import get_settings
from .base import (ClientContext, ClientSummary, Exercise, Food, MealItem, Plan, PlanDay, PlanExercise, PlanOp, User,
                   food_score, rank_foods)
from .demo import summarise_diary


class TulparGateway:
    mode = "tulpar"

    def __init__(self):
        self.s = get_settings()
        self.pool: asyncpg.Pool | None = None
        self.http = httpx.AsyncClient(base_url=self.s.tulpar_api_url.rstrip("/"), timeout=30)
        self._exercises: list[Exercise] = []

    async def start(self) -> None:
        dsn = self.s.tulpar_database_url.replace("postgresql+asyncpg://", "postgresql://")
        self.pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3,
                                              server_settings={"default_transaction_read_only": "on"})
        rows = await self.pool.fetch(
            "SELECT id::text, name, muscle_group, equipment, is_compound, measurement_type, contraindications "
            "FROM exercises WHERE is_preset ORDER BY name")
        self._exercises = [Exercise(**dict(r)) for r in rows]

    async def close(self) -> None:
        await self.http.aclose()
        if self.pool:
            await self.pool.close()

    # ── auth towards Tulpar ──────────────────────────────────────────────────
    def _token(self, user_id: str) -> str:
        now = int(time.time())
        return jwt.encode({"sub": user_id, "iat": now, "exp": now + 300}, self.s.tulpar_jwt_secret, algorithm="HS256")

    async def _req(self, method: str, path: str, user_id: str, **kw) -> httpx.Response:
        headers = kw.pop("headers", {}) | {"Authorization": f"Bearer {self._token(user_id)}"}
        r = await self.http.request(method, path, headers=headers, **kw)
        r.raise_for_status()
        return r

    # ── users ────────────────────────────────────────────────────────────────
    async def _user_row(self, where: str, arg) -> User | None:
        r = await self.pool.fetchrow(
            f"SELECT u.id::text AS id, u.name, u.telegram_id, (u.trainer_optin OR EXISTS("
            f"SELECT 1 FROM trainer_students t WHERE t.trainer_user_id=u.id)) AS is_trainer FROM users u WHERE {where}", arg)
        if r is None:
            return None
        return User(id=r["id"], role="trainer" if r["is_trainer"] else "client", name=r["name"] or "—",
                    telegram_id=r["telegram_id"])

    async def demo_user(self, role: str) -> User:
        u = await self._user_row("u.telegram_id=$1", "demo-trainer" if role == "trainer" else "demo-student")
        if u is None:
            raise LookupError("run Tulpar's seed_demo.py on the local stand first")
        return u

    async def get_user(self, user_id: str) -> User | None:
        return await self._user_row("u.id=$1::uuid", user_id)

    async def telegram_user(self, telegram_id: str, name: str, as_trainer: bool) -> User:
        u = await self._user_row("u.telegram_id=$1", telegram_id)
        if u is None:
            raise LookupError("this Telegram account is not a Tulpar user on the local stand")
        return u

    async def list_clients(self, trainer_id: str) -> list[ClientSummary]:
        rows = await self.pool.fetch(
            "SELECT u.id::text AS id, u.name, u.goal, u.level, u.place, COALESCE(n.injury,false) AS injury, "
            "(SELECT p.title FROM workout_plans p WHERE p.owner_user_id=u.id AND p.is_active LIMIT 1) AS plan_title "
            "FROM trainer_students t JOIN users u ON u.id=t.student_user_id "
            "LEFT JOIN trainer_notes n ON n.student_user_id=u.id AND n.trainer_user_id=t.trainer_user_id "
            "WHERE t.trainer_user_id=$1::uuid ORDER BY u.name", trainer_id)
        return [ClientSummary(id=r["id"], name=r["name"] or "—", goal=r["goal"], level=r["level"], place=r["place"],
                              injury=r["injury"], active_plan_title=r["plan_title"]) for r in rows]

    async def trainer_of(self, client_id: str) -> User | None:
        r = await self.pool.fetchrow("SELECT trainer_user_id::text AS id FROM trainer_students WHERE student_user_id=$1::uuid "
                                     "ORDER BY created_at LIMIT 1", client_id)
        return await self.get_user(r["id"]) if r else None

    # ── context & plan ───────────────────────────────────────────────────────
    async def client_context(self, client_id: str) -> ClientContext:
        u = await self.pool.fetchrow(
            "SELECT id::text, name, sex, birthdate, height_cm, goal, level, place, training_days, activity, goal_weight_kg "
            "FROM users WHERE id=$1::uuid", client_id)
        if u is None:
            raise LookupError("client not found")
        trainer = await self.trainer_of(client_id)
        note = None
        if trainer:
            n = await self.pool.fetchrow("SELECT body, injury FROM trainer_notes WHERE student_user_id=$1::uuid AND "
                                         "trainer_user_id=$2::uuid", client_id, trainer.id)
            note = {"injury": n["injury"], "body": n["body"]} if n else None
        since = date.today() - timedelta(days=14)
        diary = [dict(r) for r in await self.pool.fetch(
            "SELECT on_date::text AS on_date, grams, kcal, protein, fat, carbs FROM diary_entries "
            "WHERE user_id=$1::uuid AND on_date>=$2 AND eaten", client_id, since)]
        weights = [{"measured_at": r["measured_at"].date().isoformat(), "weight_kg": float(r["weight_kg"])}
                   for r in await self.pool.fetch("SELECT measured_at, weight_kg FROM weight_logs WHERE user_id=$1::uuid "
                                                  "ORDER BY measured_at DESC LIMIT 10", client_id)][::-1]
        sessions = [{"completed_at": r["completed_at"].date().isoformat(), "day_title": r["title"]} for r in await self.pool.fetch(
            "SELECT s.completed_at, d.title FROM workout_sessions s LEFT JOIN workout_days d ON d.id=s.workout_day_id "
            "WHERE s.user_id=$1::uuid AND s.completed_at IS NOT NULL AND s.counted ORDER BY s.completed_at DESC LIMIT 5",
            client_id)][::-1]
        age = None
        if u["birthdate"]:
            age = (date.today() - u["birthdate"]).days // 365
        profile = {"sex": u["sex"], "age": age, "height_cm": u["height_cm"], "goal": u["goal"], "level": u["level"],
                   "place": u["place"], "training_days": list(u["training_days"] or []), "activity": u["activity"],
                   "weight_kg": weights[-1]["weight_kg"] if weights else None,
                   "goal_weight_kg": float(u["goal_weight_kg"]) if u["goal_weight_kg"] else None}
        return ClientContext(client_id=client_id, name=u["name"] or "—", profile=profile, trainer_note=note,
                             active_plan=await self.active_plan(client_id), nutrition_14d=summarise_diary(diary),
                             recent_sessions=sessions, weights=weights)

    async def active_plan(self, client_id: str) -> Plan | None:
        r = await self.pool.fetchrow("SELECT id::text, trainer_user_id::text AS tid FROM workout_plans WHERE owner_user_id=$1::uuid "
                                     "AND is_active LIMIT 1", client_id)
        if r is None:
            return None
        viewer = r["tid"] or client_id
        d = (await self._req("GET", f"/plans/{r['id']}", viewer)).json()
        days = []
        for day in d.get("days", []):
            days.append(PlanDay(id=str(day["id"]), day_index=day.get("day_index") or 0, title=day.get("title") or "",
                                weekday=day.get("weekday"), exercises=[
                    PlanExercise(id=str(e["id"]), exercise_id=str(e.get("exercise_id")) if e.get("exercise_id") else None,
                                 exercise_name=e.get("exercise_name") or "", muscle_group=e.get("muscle_group"),
                                 equipment=e.get("equipment"), target_sets=e.get("target_sets"),
                                 target_reps=e.get("target_reps")) for e in day.get("exercises", [])]))
        return Plan(id=str(d["id"]), title=d.get("title") or "", days=days)

    async def apply_ops(self, client_id: str, trainer_id: str, ops: list[PlanOp]) -> tuple[Plan, Plan]:
        before = await self.active_plan(client_id)
        if before is None:
            raise LookupError("client has no active plan")
        day_id = {d.day_index: d.id for d in before.days}
        try:
            for op in ops:
                did = day_id[op.day_index]
                if op.op == "set_volume":
                    body = {k: v for k, v in {"target_sets": op.sets, "target_reps": op.reps}.items() if v}
                    await self._req("PATCH", f"/days/{did}/exercises/{op.wex_id}", trainer_id, json=body)
                elif op.op == "remove_exercise":
                    await self._req("DELETE", f"/days/{did}/exercises/{op.wex_id}", trainer_id)
                elif op.op in ("replace_exercise", "add_exercise"):
                    old = next((e for d in before.days for e in d.exercises if e.id == op.wex_id), None)
                    body = {"exercise_id": op.exercise_id,
                            "target_sets": op.sets or (old.target_sets if old else 3),
                            "target_reps": op.reps or (old.target_reps if old else 10)}
                    await self._req("POST", f"/days/{did}/exercises", trainer_id, json=body)
                    if op.op == "replace_exercise":
                        await self._req("DELETE", f"/days/{did}/exercises/{op.wex_id}", trainer_id)
        except Exception:
            await self.restore_plan(client_id, trainer_id, before)
            raise
        return before, await self.active_plan(client_id)

    async def restore_plan(self, client_id: str, trainer_id: str, before: Plan) -> None:
        """Best-effort: bring each day back to the snapshot (remove extras, re-add missing, reset volume)."""
        now = await self.active_plan(client_id)
        if now is None:
            return
        for bday in before.days:
            nday = next((d for d in now.days if d.day_index == bday.day_index), None)
            if nday is None:
                continue
            want = {e.exercise_id: e for e in bday.exercises}
            for e in nday.exercises:
                if e.exercise_id not in want:
                    await self._req("DELETE", f"/days/{nday.id}/exercises/{e.id}", trainer_id)
                else:
                    w = want[e.exercise_id]
                    if (e.target_sets, e.target_reps) != (w.target_sets, w.target_reps):
                        await self._req("PATCH", f"/days/{nday.id}/exercises/{e.id}", trainer_id,
                                        json={"target_sets": w.target_sets, "target_reps": w.target_reps})
            have = {e.exercise_id for e in nday.exercises}
            for w in bday.exercises:
                if w.exercise_id not in have:
                    await self._req("POST", f"/days/{nday.id}/exercises", trainer_id,
                                    json={"exercise_id": w.exercise_id, "target_sets": w.target_sets, "target_reps": w.target_reps})

    def exercises(self) -> list[Exercise]:
        return self._exercises

    # ── food ─────────────────────────────────────────────────────────────────
    async def search_foods(self, client_id: str, query: str, limit: int = 5) -> list[Food]:
        seen: dict[str, dict] = {}
        words = sorted({w for w in query.split() if len(w) > 2}, key=len, reverse=True)
        for q in [query, *words[:2]]:
            r = await self._req("GET", "/foods", client_id, params={"q": q, "limit": 20})
            for f in r.json():
                seen.setdefault(str(f["id"]), {"id": str(f["id"]), "name": f["name"], "kcal": f.get("kcal") or 0,
                                               "protein": f.get("protein") or 0, "fat": f.get("fat") or 0,
                                               "carbs": f.get("carbs") or 0})
            if any(food_score(query, f["name"]) >= 0.5 for f in seen.values()):
                break
        return rank_foods(query, list(seen.values()), limit)

    async def log_meal(self, client_id: str, items: list[MealItem], meal: str, on_date: str, idem: str) -> int:
        n = 0
        for i, it in enumerate(items):
            body = {"on_date": on_date, "meal": meal, "name": it.name, "grams": it.grams, "kcal": it.kcal,
                    "protein": it.protein, "fat": it.fat, "carbs": it.carbs}
            if it.food_id:
                body["food_id"] = it.food_id
            await self._req("POST", "/me/diary", client_id, json=body, headers={"Idempotency-Key": f"{idem}:{i}"})
            n += 1
        return n
