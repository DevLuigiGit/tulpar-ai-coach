"""The ONE boundary between the agent and Tulpar.

Everything the agent needs from the product goes through `Gateway`. Two implementations:
  DemoGateway   — fixtures + own SQLite; runs without Tulpar (Madina, mentors, CI, Railway demo)
  TulparGateway — HTTP API + read-only SQL of a LOCAL Tulpar stand
The graph, the API, the bot and the MCP server never know which one is behind.
"""

from __future__ import annotations

import re
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class User(BaseModel):
    id: str
    role: Literal["client", "trainer"]
    name: str
    telegram_id: str | None = None


class ClientSummary(BaseModel):
    id: str
    name: str
    goal: str | None = None
    level: str | None = None
    place: str | None = None
    injury: bool = False
    active_plan_title: str | None = None


class PlanExercise(BaseModel):
    id: str
    exercise_id: str | None = None
    exercise_name: str
    muscle_group: str | None = None
    equipment: str | None = None
    target_sets: int | None = None
    target_reps: int | None = None


class PlanDay(BaseModel):
    id: str
    day_index: int
    title: str
    weekday: int | None = None
    exercises: list[PlanExercise] = Field(default_factory=list)


class Plan(BaseModel):
    id: str
    title: str
    days: list[PlanDay] = Field(default_factory=list)


class PlanOp(BaseModel):
    op: Literal["replace_exercise", "set_volume", "remove_exercise", "add_exercise"]
    day_index: int
    wex_id: str | None = None
    exercise_id: str | None = None
    sets: int | None = None
    reps: int | None = None
    reason: str = ""


class ClientContext(BaseModel):
    client_id: str
    name: str
    profile: dict = Field(default_factory=dict)
    trainer_note: dict | None = None
    active_plan: Plan | None = None
    nutrition_14d: dict = Field(default_factory=dict)
    recent_sessions: list[dict] = Field(default_factory=list)
    weights: list[dict] = Field(default_factory=list)

    def for_llm(self) -> dict:
        """What the model sees: no name, no ids of people, no raw diary."""
        return {
            "profile": self.profile,
            "trainer_note": self.trainer_note,
            "nutrition_14d": self.nutrition_14d,
            "recent_sessions": [
                {"completed_at": s.get("completed_at"), "day_title": s.get("day_title"), "avg_rpe": s.get("avg_rpe")}
                for s in self.recent_sessions[-5:]
            ],
            "weights": self.weights[-5:],
        }


class Exercise(BaseModel):
    id: str
    name: str
    muscle_group: str | None = None
    equipment: str | None = None
    is_compound: bool = False
    measurement_type: str | None = None
    contraindications: str | None = None


class Food(BaseModel):
    id: str
    name: str
    kcal: float
    protein: float
    fat: float
    carbs: float
    score: float = 0.0


class MealItem(BaseModel):
    food_id: str | None = None
    name: str
    grams: float
    kcal: float  # per 100 g — Tulpar's diary invariant
    protein: float
    fat: float
    carbs: float


class Gateway(Protocol):
    mode: str

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def demo_user(self, role: str) -> User: ...
    async def get_user(self, user_id: str) -> User | None: ...
    async def telegram_user(self, telegram_id: str, name: str, as_trainer: bool) -> User: ...
    async def list_clients(self, trainer_id: str) -> list[ClientSummary]: ...
    async def trainer_of(self, client_id: str) -> User | None: ...
    async def client_context(self, client_id: str) -> ClientContext: ...
    async def active_plan(self, client_id: str) -> Plan | None: ...
    async def apply_ops(self, client_id: str, trainer_id: str, ops: list[PlanOp]) -> tuple[Plan, Plan]: ...
    async def restore_plan(self, client_id: str, trainer_id: str, before: Plan) -> None: ...
    def exercises(self) -> list[Exercise]: ...
    async def search_foods(self, client_id: str, query: str, limit: int = 5) -> list[Food]: ...
    async def log_meal(self, client_id: str, items: list[MealItem], meal: str, on_date: str, idem: str) -> int: ...


# ── food name matching (same idea as Tulpar's photo pipeline: token Jaccard, threshold 0.34) ──
_STOP = {"с", "и", "в", "на", "из", "по", "для", "без", "со", "под", "от"}
MATCH_THRESHOLD = 0.34


def tokens(text: str) -> set[str]:
    t = re.sub(r"[^0-9a-zа-я]+", " ", (text or "").lower().replace("ё", "е"))
    return {w[:6] for w in t.split() if w and w not in _STOP}


def food_score(query: str, name: str) -> float:
    a, b = tokens(query), tokens(name)
    if not a or not b:
        return 0.0
    score = len(a & b) / len(a | b)
    if (name or "").lower().startswith((query or "").lower().strip()):
        score += 0.15
    return min(score, 1.0)


def rank_foods(query: str, foods: list[dict], limit: int = 5) -> list[Food]:
    scored = [(food_score(query, f["name"]), f) for f in foods]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: (-x[0], len(x[1]["name"])))
    return [Food(**{**f, "score": round(s, 3)}) for s, f in scored[:limit]]
