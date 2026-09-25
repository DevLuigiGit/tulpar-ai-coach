"""tools/demo_scenario.py end to end over HTTP with the fake LLM; pure helpers of tools/mcp_demo.py."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from conftest import FakeLLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import demo_scenario  # noqa: E402
import mcp_demo  # noqa: E402


class DemoFake(FakeLLM):
    """The shared fake routes only «замени/программ» to program_request; the demo asks «Хочу добавить кардио»."""

    def __call__(self, *, role, system, user, images, json_mode):
        if "маршрутизатор" in system and "кардио" in user.lower():
            return json.dumps({"intent": "program_request", "red_flag": False, "reason": "plan"})
        return super().__call__(role=role, system=system, user=user, images=images, json_mode=json_mode)


def test_demo_scenario_runs_all_steps(env):
    from tulpar_ai import llm
    from tulpar_ai.api.app import app

    llm.set_fake(DemoFake())
    lines: list[str] = []
    try:
        with TestClient(app) as c:
            sc = demo_scenario.Scenario(c, out=lines.append, draft_timeout=15, poll_s=0.1)
            sc.run()
    finally:
        llm.set_fake(None)
    text = "\n".join(lines)
    assert sc.n == 9 and not sc.warnings, text
    assert "kind=answer" in text and "kind=meal_card" in text and "kind=proposal" in text
    assert "статус: applied" in text
    assert any(line.strip().startswith("+ ") for line in lines), text  # the client's plan really changed
    assert sc.sent == [demo_scenario.QUESTION, demo_scenario.MEAL, demo_scenario.PROGRAM] and sc.proposal_id


def test_plan_diff():
    before = {"days": [{"title": "Ноги", "exercises": [{"exercise_name": "Присед", "target_sets": 4, "target_reps": 8}]}]}
    after = {"days": [{"title": "Ноги", "exercises": [{"exercise_name": "Ягодичный мост", "target_sets": 4, "target_reps": 10}]}]}
    assert demo_scenario.plan_diff(before, after) == (["Ноги: Присед 4×8"], ["Ноги: Ягодичный мост 4×10"])
    assert demo_scenario.plan_diff(before, before) == ([], [])


def test_match_runs_keeps_only_this_scenario():
    t = datetime.now(timezone.utc)
    run = lambda name, inputs, meta=None: SimpleNamespace(name=name, inputs=inputs, start_time=t,  # noqa: E731
                                                          extra={"metadata": meta or {}})
    mine = run("coach_turn", {"text": "Съел 200 г плова и чай", "client_id": "c1"})
    other_client = run("coach_turn", {"text": "Съел 200 г плова и чай", "client_id": "c2"})
    resume = run("program_change", {}, {"proposal": "abcdef12"})
    foreign = run("program_change", {}, {"proposal": "99999999"})
    got = demo_scenario.match_runs([mine, other_client, resume, foreign], ["Съел 200 г плова и чай"], "c1",
                                   "abcdef12-0000")
    assert got == [mine, resume]


def test_mcp_demo_unpack_and_summaries():
    text = lambda s: SimpleNamespace(text=s)  # noqa: E731
    clients = [{"id": "9b7e3eaa-1", "name": "Айдар Ким", "goal": "cut", "place": "gym", "injury": True}]
    wrapped = SimpleNamespace(is_error=False, structured_content={"result": clients}, content=[])
    assert mcp_demo.unpack(wrapped) == clients
    plain = SimpleNamespace(is_error=False, structured_content=None, content=[text(json.dumps({"status": "pending"}))])
    assert mcp_demo.unpack(plain) == {"status": "pending"}
    assert "ограничение по здоровью: да" in mcp_demo.summarize("list_clients", clients)[0]
    lines = mcp_demo.summarize("propose_program_change", {
        "proposal_id": "abcdef12-x", "status": "pending", "summary": "облегчить ноги",
        "changes": [{"day": "Ноги", "was": "Присед 4×8", "becomes": "Ягодичный мост 4×10", "reason": "колено"}],
        "warnings": ["в плане остаётся «Жим ногами»"]})
    assert lines[0].startswith("статус: pending") and lines[2] == "Ноги: Присед 4×8 → Ягодичный мост 4×10 (колено)"
    err = SimpleNamespace(is_error=True, structured_content=None, content=[text("404: client not found")])
    try:
        mcp_demo.unpack(err)
    except RuntimeError as e:
        assert "404" in str(e)
    else:
        raise AssertionError("an MCP tool error must raise")
