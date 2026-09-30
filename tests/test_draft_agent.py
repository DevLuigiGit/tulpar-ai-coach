"""The draft agent (DRAFT_MODE=agent): tool loop, step limit, protocol errors, traced tool runs, shared search."""

from __future__ import annotations

import asyncio
import json
import re
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from langsmith import Client
from langsmith.run_helpers import tracing_context

from conftest import boot, shutdown


@pytest.fixture
def agent_env(env, monkeypatch):
    from tulpar_ai.config import get_settings

    monkeypatch.setenv("DRAFT_MODE", "agent")
    get_settings.cache_clear()
    yield env
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def world(agent_env, fake_llm):
    store, gw = await boot(agent_env)
    yield store, gw
    await shutdown(store)


async def _state(store, gw, request: str = "Замени первое упражнение") -> dict:
    """A program-graph state right after load_context (demo client with the knee note), to drive the agent directly."""
    from tulpar_ai.graph import program as pg

    client, trainer = await gw.demo_user("client"), await gw.demo_user("trainer")
    p = await store.create_proposal(kind="program", client_id=client.id, trainer_id=trainer.id, source="trainer",
                                    request=request, status="drafting")
    st = {"proposal_id": p["id"], "client_id": client.id, "trainer_id": trainer.id, "request": request,
          "source": "trainer"}
    st.update(await pg.load_context(st))
    return st


class Scripted:
    """Replies of the draft agent in order; `seen` keeps every prompt the agent sent."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, *, role, system, user, images, json_mode):
        assert '"action": "check_plan"' in system, "the agent must run on draft.v3"
        self.seen.append(user)
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return r(user) if callable(r) else r


def _first_wex(st) -> tuple[str, str]:
    e = st["plan"]["days"][0]["exercises"][0]
    return e["id"], e["muscle_group"]


def _ids_in(text: str) -> list[str]:
    return re.findall(r"^([0-9a-f-]{36}) \|", text.split("Шаги:", 1)[1], re.M)


async def wait_status(store, pid, status, timeout=60.0):
    for _ in range(int(timeout * 20)):
        p = await store.get_proposal(pid)
        if p["status"] == status:
            return p
        await asyncio.sleep(0.05)
    raise AssertionError(f"proposal stayed {p['status']!r}, expected {status!r}")


# ── the loop inside the graph ────────────────────────────────────────────────
async def test_agent_searches_checks_then_submits(world, fake_llm):
    """Scripted fake: search → check_plan → final. The graph keeps its validate node and the trainer's review."""
    from tulpar_ai.graph import runner

    store, gw = world
    client, trainer = await gw.demo_user("client"), await gw.demo_user("trainer")
    p = await runner.new_program_proposal(client.id, trainer.id, "Замени первое упражнение", "trainer")
    p = await wait_status(store, p["id"], "pending")
    assert p["draft"]["ops"] and not [v for v in p["violations"] if v["severity"] == "error"]
    # «pending» is written by `publish` a moment before the review pause is checkpointed: wait for the pause
    cfg = {"configurable": {"thread_id": f"prop:{p['id']}"}}
    for _ in range(200):
        snap = await runner.program_graph().aget_state(cfg)
        if snap.next == ("review",):
            break
        await asyncio.sleep(0.05)
    assert snap.next == ("review",)
    log = snap.values["agent_log"]
    assert [r["action"] for r in log] == ["search_exercises", "check_plan", "final"]
    assert log[1]["ok"] and log[2]["ok"] and log[2]["auto_check"]
    assert len([c for c in fake_llm.calls if c["system"].startswith("Ты помогаешь тренеру")]) == 3
    await runner.resume_program(p["id"], "accept")
    assert (await store.get_proposal(p["id"]))["status"] == "applied"


async def test_invalid_agent_draft_still_goes_through_the_validator_loop(world, fake_llm):
    """An agent that never fixes its draft uses up its steps; the graph's validator then asks for a redraft."""
    from tulpar_ai.graph import runner

    store, gw = world
    fake_llm.draft_plan = ["invalid", "valid"]
    client, trainer = await gw.demo_user("client"), await gw.demo_user("trainer")
    p = await runner.new_program_proposal(client.id, trainer.id, "Замени первое упражнение", "trainer")
    p = await wait_status(store, p["id"], "pending")
    assert "(valid)" in p["draft"]["summary"] and fake_llm.drafts == 2


# ── step limit ───────────────────────────────────────────────────────────────
async def test_step_limit_forces_a_final_call(world):
    from tulpar_ai import llm
    from tulpar_ai.graph import draft_agent

    store, gw = world
    st = await _state(store, gw)
    fake = Scripted([json.dumps({"action": "search_exercises", "args": {"muscle_group": "ноги"}})])
    llm.set_fake(fake)
    draft, log = await draft_agent.run(st)
    assert len(fake.seen) == 6  # 5 tool steps + 1 call that may only return final
    assert "последний шаг" in fake.seen[-1] and "последний шаг" not in fake.seen[-2]
    assert [r["action"] for r in log] == ["search_exercises"] * 5 + ["no_final"]
    assert draft["ops"] == [] and "шаги" in draft["summary"]
    assert draft_agent.stats(log) | {"tools": None} == {"llm_steps": 6, "tool_calls": 5, "tools": None,
                                                          "rejected_finals": 0, "protocol_errors": 0}


async def test_step_limit_is_a_setting_and_falls_back_to_checked_ops(world, monkeypatch):
    from tulpar_ai import llm
    from tulpar_ai.config import get_settings
    from tulpar_ai.graph import draft_agent

    monkeypatch.setattr(get_settings(), "draft_agent_max_steps", 2)
    store, gw = world
    st = await _state(store, gw)
    wex, group = _first_wex(st)

    def check(user):
        ids = _ids_in(user)
        op = {"op": "replace_exercise", "day_index": 0, "wex_id": wex, "exercise_id": ids[0], "reason": "тест"}
        return json.dumps({"action": "check_plan", "args": {"ops": [op]}})

    fake = Scripted([json.dumps({"action": "search_exercises", "args": {"muscle_group": group}}), check,
                     json.dumps({"action": "search_exercises", "args": {"query": "мост"}})])
    llm.set_fake(fake)
    draft, log = await draft_agent.run(st)
    assert len(fake.seen) == 3
    assert [r["action"] for r in log] == ["search_exercises", "check_plan", "no_final"]
    assert len(draft["ops"]) == 1 and draft["ops"][0]["wex_id"] == wex  # the ops the validator passed


# ── malformed replies ────────────────────────────────────────────────────────
async def test_malformed_actions_are_reported_back_to_the_model(world):
    from tulpar_ai import llm
    from tulpar_ai.graph import draft_agent

    store, gw = world
    st = await _state(store, gw)
    wex, group = _first_wex(st)

    def good_final(user):
        ids = _ids_in(user)
        return json.dumps({"action": "final", "summary": "Замена", "rationale": "тест", "ops": [
            {"op": "replace_exercise", "day_index": 0, "wex_id": wex, "exercise_id": ids[0], "reason": "тест"}]})

    fake = Scripted([
        json.dumps({"thought": "подумаю"}),  # no action
        json.dumps({"action": "drop_plan", "args": {}}),  # unknown tool
        json.dumps([1, 2]),  # JSON, but not an object
        json.dumps({"action": "final", "summary": "x", "ops": [{"op": "explode", "day_index": 0}]}),  # unreadable op
        json.dumps({"action": "search_exercises", "muscle_group": group}),  # args at the top level are accepted
        good_final,
    ])
    llm.set_fake(fake)
    draft, log = await draft_agent.run(st)
    assert [r["action"] for r in log] == ["invalid", "invalid", "invalid", "final", "search_exercises", "final"]
    assert 'нет поля "action"' in fake.seen[1] and "неизвестное действие 'drop_plan'" in fake.seen[2]
    assert "одним JSON-объектом" in fake.seen[3] and "не распознана" in fake.seen[4]
    assert log[4]["args"]["muscle_group"] == group and log[4]["found"] > 0
    assert draft["summary"] == "Замена" and len(draft["ops"]) == 1
    s = draft_agent.stats(log)
    assert s["protocol_errors"] == 3 and s["rejected_finals"] == 1 and s["llm_steps"] == 6


def test_parse_reply_batch_and_final():
    from tulpar_ai.graph.draft_agent import parse_reply

    calls, final, problem = parse_reply({"actions": [{"action": "get_client_context"}] + [
        {"action": "search_exercises", "args": {"query": q}} for q in ("мост", "тяга", "жим", "присед")]})
    assert [c[0] for c in calls] == ["get_client_context", "search_exercises", "search_exercises"] and not problem
    assert parse_reply({"summary": "s", "ops": []})[1] == {"summary": "s", "ops": []}  # no action + ops = final
    assert parse_reply({"actions": [{"action": "search_exercises"}, {"action": "final", "ops": []}]})[1] == {"ops": []}
    assert parse_reply({"actions": []})[2] == "пустой список actions"


# ── traces ───────────────────────────────────────────────────────────────────
async def test_each_tool_call_is_a_tool_run_under_the_draft(agent_env, fake_llm):
    from tulpar_ai import llm
    from tulpar_ai.graph import runner

    client_ls = MagicMock(spec=Client)
    store, gw = await boot(agent_env)
    try:
        st_client, trainer = await gw.demo_user("client"), await gw.demo_user("trainer")
        steps = iter(["context", "search", "check", "final"])

        def scripted(*, role, system, user, images, json_mode):
            if '"action": "check_plan"' not in system:
                return fake_llm(role=role, system=system, user=user, images=images, json_mode=json_mode)
            step = next(steps)
            if step == "context":
                return json.dumps({"action": "get_client_context", "args": {}})
            wex, group = re.search(r"wex_id=(\S+) \| [^|]+ \| ([^|]+) \|", user).groups()
            if step == "search":
                return json.dumps({"action": "search_exercises", "args": {"muscle_group": group.strip()}})
            op = {"op": "replace_exercise", "day_index": 0, "wex_id": wex, "exercise_id": _ids_in(user)[0], "reason": "т"}
            if step == "check":
                return json.dumps({"action": "check_plan", "args": {"ops": [op]}})
            return json.dumps({"action": "final", "summary": "Замена", "rationale": "т", "ops": [op]})

        llm.set_fake(scripted)
        with tracing_context(enabled=True, client=client_ls, project_name="test"):
            p = await runner.new_program_proposal(st_client.id, trainer.id, "Замени первое упражнение", "trainer")
            await wait_status(store, p["id"], "pending")
    finally:
        await shutdown(store)
    runs = {}
    for call in client_ls.create_run.call_args_list:
        r = call.kwargs
        runs[str(r["id"])] = r

    def ancestors(r):
        out = []
        while r.get("parent_run_id") and str(r["parent_run_id"]) in runs:
            r = runs[str(r["parent_run_id"])]
            out.append(r["name"])
        return out

    tools = [r for r in runs.values() if r.get("run_type") == "tool"]
    assert [r["name"] for r in sorted(tools, key=lambda r: r["start_time"])] == [
        "get_client_context", "search_exercises", "check_plan", "check_plan"]  # the last one checks the final
    for r in tools:
        assert "draft_agent" in ancestors(r) and "program_change" in ancestors(r), r["name"]
    ctx = next(r for r in tools if r["name"] == "get_client_context")
    assert ctx["inputs"] == {"client": st_client.id[:8]}  # no full client id in the trace
    search = next(r for r in tools if r["name"] == "search_exercises")
    assert search["inputs"]["place"] == "gym" and search["inputs"]["avoid"] == ["колени"]
    # the node's own run carries the counts, so a LangSmith filter finds drafts by agent behaviour
    [node] = [c.kwargs for c in client_ls.update_run.call_args_list if c.kwargs.get("name") == "draft_agent"]
    meta = node["extra"]["metadata"]
    assert meta["draft_mode"] == "agent" and meta["llm_steps"] == 4 and meta["tool_calls"] == 4
    assert meta["tools"] == {"check_plan": 2, "get_client_context": 1, "search_exercises": 1}


# ── one catalogue search for the agent and the MCP tool ──────────────────────
async def test_shared_search_filters(world):
    from tulpar_ai import service, tools
    from tulpar_ai.skill import validator

    store, gw = world
    v = validator()
    # /api/exercises keeps its old substring behaviour and gains word-start matching
    assert {e["name"] for e in service.search_exercises("присед", None, None, None)} >= {"Приседания со штангой"}
    assert any(e["name"] == "Жим гантелей лёжа" for e in service.search_exercises("гантели жим", None, None, None))
    assert all(e["muscle_group"] in ("пресс", "кор") for e in service.search_exercises(None, "кор", None, None))
    knee = service.search_exercises(None, "ноги", None, "колено")
    assert knee and not any("колени" in v.body_parts(e["contraindications"]) for e in knee)

    # the agent's search applies the client's own limits: Dana trains at home, Aidar has a knee note
    home = {"place": "home", "trainer_note": {"body": "Тренируется дома"}}
    rows = tools.search_exercises(muscle_group="грудь", client=home)
    assert rows and all(r["equipment"] in (None, "bodyweight", "band", "dumbbell", "kettlebell") for r in rows)
    assert len(rows) <= tools.SEARCH_LIMIT and rows[0]["compound"] >= rows[-1]["compound"]
    knee_client = {"place": "gym", "trainer_note": {"injury": True, "body": "Жалуется на левое колено"}}
    cat = {e.id: e for e in gw.exercises()}
    rows = tools.search_exercises(muscle_group="ноги", client=knee_client, limit=100)
    assert rows and not any("колени" in v.body_parts(cat[r["exercise_id"]].contraindications) for r in rows)


def test_name_matching():
    from tulpar_ai.tools import name_matches

    assert name_matches("жим", "Отжимания")  # substring, as /api/exercises always did
    assert name_matches("тяги гантели", "Тяга гантели в наклоне")
    assert not name_matches("тяга штанги", "Тяга гантели в наклоне")
    assert name_matches("", "Любое")
