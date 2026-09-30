"""Tests run with NO network and NO keys: a deterministic fake LLM, the local hashing embedder."""

from __future__ import annotations

import json
import re

import pytest
import pytest_asyncio

UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


class FakeLLM:
    """Answers by role and prompt content. `draft_plan` lets a test script the sequence of drafts."""

    def __init__(self):
        self.calls: list[dict] = []
        self.draft_plan: list[str] = ["valid"]
        self.drafts = 0  # drafts started, in either DRAFT_MODE
        self.agent_kind = "valid"

    def __call__(self, *, role, system, user, images, json_mode):
        self.calls.append({"role": role, "system": system[:80], "user": user[:200]})
        if "маршрутизатор" in system:
            u = user.lower()
            if "колено болит" in u or "стероид" in u:
                return json.dumps({"intent": "escalate", "red_flag": True, "reason": "pain"})
            if "съел" in u or " г " in f"{u} ":
                return json.dumps({"intent": "meal_text", "red_flag": False, "reason": "food"})
            if "замени" in u or "программ" in u:
                return json.dumps({"intent": "program_request", "red_flag": False, "reason": "plan"})
            return json.dumps({"intent": "question", "red_flag": False, "reason": "q"})
        if "поисковый запрос" in system:
            return json.dumps({"query": user + " physical activity minutes per week"})
        if '"action": "check_plan"' in system:  # draft agent (DRAFT_MODE=agent): search → check → final
            return self.agent_step(user)
        if "tulpar-program-builder" in system:
            self.drafts += 1
            kind = self.draft_plan.pop(0) if self.draft_plan else "valid"
            wex, group = re.search(rf"wex_id=({UUID}) \| [^|]+ \| ([^|]+) \|", user).groups()
            cand = re.findall(rf"^- ({UUID}) \| [^|]+ \| {re.escape(group.strip())} \|", user, re.M)
            ex = "00000000-0000-0000-0000-000000000000" if kind == "invalid" else cand[0]
            return json.dumps({"summary": self.summary(kind, user), "rationale": "тест",
                               "ops": [{"op": "replace_exercise", "day_index": 0, "wex_id": wex, "exercise_id": ex,
                                        "sets": 3, "reps": 10, "reason": "тест"}]})
        if "AI-коуч фитнес-клуба Tulpar" in system:
            return json.dumps({"answer": "По рекомендациям ВОЗ взрослым нужно 150–300 минут умеренной активности в неделю [1].",
                               "citations": [1], "sufficient": True})
        if "выпиши продукты" in system:
            return json.dumps({"items": [{"name": "плов", "grams": 200}, {"name": "чай", "grams": None}], "meal": "lunch"})
        if "по фотографии" in system:
            return json.dumps({"items": [{"name": "плов", "alternatives": ["плов с говядиной"], "confidence": "high"}], "note": ""})
        return json.dumps({"score": 5, "reason": "fake"})

    def agent_step(self, user: str) -> str:
        """One step of the draft agent. `draft_plan` kinds apply per draft: "invalid" keeps submitting an unknown
        exercise until the forced final, so the graph's validator sends the draft back."""
        if "Шаги:\n(пока нет)" in user:
            self.drafts += 1
            self.agent_kind = self.draft_plan.pop(0) if self.draft_plan else "valid"
        kind = self.agent_kind
        wex, group = re.search(rf"wex_id=({UUID}) \| [^|]+ \| ([^|]+) \|", user).groups()
        group = group.strip()
        steps = user.split("Шаги:", 1)[1]
        found = re.findall(rf"^({UUID}) \| [^|]+ \| {re.escape(group)} \|", steps, re.M)
        ex = "00000000-0000-0000-0000-000000000000" if kind == "invalid" else (found[0] if found else None)
        if ex is None:
            return json.dumps({"action": "search_exercises", "args": {"query": "", "muscle_group": group}})
        ops = [{"op": "replace_exercise", "day_index": 0, "wex_id": wex, "exercise_id": ex, "sets": 3, "reps": 10,
                "reason": "тест"}]
        if kind != "invalid" and "check_plan" not in steps:
            return json.dumps({"action": "check_plan", "args": {"ops": ops}})
        return json.dumps({"action": "final", "summary": self.summary(kind, user), "rationale": "тест", "ops": ops})

    @staticmethod
    def summary(kind: str, user: str) -> str:
        """«overclaim»: the summary promises to replace the first two exercises, the ops replace only the first."""
        if kind == "overclaim":
            first, second = re.findall(rf"wex_id={UUID} \| ([^|]+) \|", user)[:2]
            return f"Заменить {first.strip()} и {second.strip()} на щадящие варианты ({kind})"
        return f"Заменить первое упражнение ({kind})"


@pytest.fixture
def fake_llm():
    from tulpar_ai import llm

    fake = FakeLLM()
    llm.set_fake(fake)
    yield fake
    llm.set_fake(None)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TULPAR_MODE", "demo")
    monkeypatch.setenv("RAG_EMBEDDER", "local")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")  # rate-limit tests switch it on explicitly
    for k in ("OLLAMA_API_KEY", "GROQ_API_KEY", "JINA_API_KEY", "TELEGRAM_BOT_TOKEN", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.setenv(k, "")
    from tulpar_ai.config import get_settings

    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


async def boot(data_dir):
    """What the API lifespan does, minus HTTP and the bot."""
    from tulpar_ai.gateway import build_gateway, set_gateway
    from tulpar_ai.graph import runner
    from tulpar_ai.store import Store, set_store

    store = await Store(data_dir / "app.sqlite").open()
    set_store(store)
    gw = build_gateway(store)
    await gw.start()
    set_gateway(gw)
    await runner.open_graphs(data_dir / "graph.sqlite")
    return store, gw


async def shutdown(store):
    from tulpar_ai.gateway import set_gateway
    from tulpar_ai.graph import runner
    from tulpar_ai.store import set_store

    await runner.close_graphs()
    await store.close()
    set_store(None)
    set_gateway(None)


@pytest_asyncio.fixture
async def app_state(env, fake_llm):
    store, gw = await boot(env)
    yield store, gw
    await shutdown(store)
