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
    async def guest_user(self, key: str) -> User: ...
    async def needs_onboarding(self, user: User) -> bool: ...
    async def client_profile(self, client_id: str) -> dict: ...
    async def update_profile(self, client_id: str, name: str, fields: dict) -> None: ...
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


# What a Russian noun or adjective changes in its cases: «плов» ~ «плова», but «хлеб» ≁ «хлебцы», «батон» ≁ «батончик».
_ENDINGS = {"", "а", "я", "у", "ю", "е", "и", "ы", "о", "ь", "й", "ой", "ей", "ом", "ем", "ам", "ям", "ах", "ях", "ов", "ев",
            "ий", "ый", "ая", "яя", "ое", "ее", "ые", "ие", "ую", "юю", "их", "ых", "им", "ым", "ого", "его", "ому", "ему",
            "ами", "ями", "ыми", "ими"}


def _same(x: str, y: str) -> bool:
    """Russian cases differ in endings: «плова» ~ «плов», «гречки» ~ «гречка». Common prefix of ≥4 letters, and what
    is left of both words after it must be a case ending, not a new word («хлеб» → «хлебцы» is another product)."""
    if x == y:
        return True
    short, long_ = sorted((x, y), key=len)
    if len(short) < 4:
        return False
    stem = short[: max(4, len(short) - 1)]
    return long_.startswith(stem) and long_[len(stem):] in _ENDINGS and short[len(stem):] in _ENDINGS


ADDITION_PENALTY = 0.05


def name_parts(text: str) -> tuple[set[str], set[str], set[str]]:
    """(base, added, without) stems. «Чай с лимоном и сахаром» adds лимон and сахар; «Гусь без кожи сырой»
    leaves out кожа. A preposition takes one word (more only through «и»), so «сырой» stays in the base."""
    base: set[str] = set()
    added: set[str] = set()
    without: set[str] = set()
    part, left = base, 0
    for w in re.findall(r"[0-9a-zа-я]+|[()]", (text or "").lower().replace("ё", "е")):
        if w in "()":
            part, left = base, 0
        elif w in ("с", "со", "без"):
            part, left = (without if w == "без" else added), 1
        elif w == "и" and part is not base:
            left = 1
        elif w not in _STOP:
            (part if left else base).add(w[:6])
            left = max(left - 1, 0)
    return base, added, without


def food_score(query: str, name: str) -> float:
    """Token Jaccard. A name's «без X» counts only when the query also says «без X»: otherwise plain «чай» loses
    to «чай с мёдом» on the extra words of «чай чёрный без сахара». Each addition the user did not ask for
    (the «с X» of the name) costs a little: it changes the calories."""
    qb, qa, qw = name_parts(query)
    nb, na, nw = name_parts(name)
    if not any(w[0].isdigit() for w in qb | qa):  # «Молоко 2.5%» is plain «молоко» to someone who named no fat %
        nb = {w for w in nb if not w[0].isdigit()}
    if qb and not any(_same(x, y) for x in qb for y in nb):
        return 0.0  # the dish must match, not its addition: «пельмени со сметаной» is not «Сметана 15%»
    a = qb | qa | {"-" + x for x in qw}
    b = nb | na | {"-" + y for y in nw if any(_same(x, y) for x in qw)}
    if not a or not b:
        return 0.0
    matched = sum(1 for x in a if any(_same(x, y) for y in b))
    score = matched / (len(a) + len(b) - matched)
    if (name or "").lower().startswith((query or "").lower().strip()):
        score += 0.15
    score -= ADDITION_PENALTY * sum(1 for y in na if not any(_same(x, y) for x in qb | qa))
    return max(0.0, min(score, 1.0))


# Everyday names the catalog spells differently: «чёрный хлеб» is rye bread, not «Хлебцы» or «Чернослив».
_ALIASES = [(re.compile(r"\bч[её]рн\w*\s+хлеб\w*|\bхлеб\w*\s+ч[её]рн\w*", re.I), "хлеб ржаной")]


def rank_foods(query: str, foods: list[dict], limit: int = 5) -> list[Food]:
    for pattern, canonical in _ALIASES:
        query = pattern.sub(canonical, query or "")
    scored = [(food_score(query, f["name"]), f) for f in foods]
    scored = [x for x in scored if x[0] > 0]
    # ties keep the catalog order: Tulpar lists the everyday variant first («Яйцо куриное» before «Яйцо утиное»,
    # «Молоко 2.5%» before «Молоко козье»), which the old shortest-name rule turned upside down
    scored.sort(key=lambda x: -x[0])
    return [Food(**{**f, "score": round(s, 3)}) for s, f in scored[:limit]]
