"""POST /api/chat/stream: the graph's stages as server-sent events, then the same reply as POST /api/chat."""

import json

from fastapi.testclient import TestClient


def _events(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def test_question_streams_stages_then_the_reply(env, fake_llm):
    from tulpar_ai.api.app import app

    with TestClient(app) as c:
        token = c.post("/api/auth/demo-login", json={"role": "client"}).json()["token"]
        h = {"Authorization": f"Bearer {token}"}
        r = c.post("/api/chat/stream", data={"text": "Сколько минут активности в неделю рекомендует ВОЗ?"}, headers=h)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = _events(r.text)
        stages = [e["stage"] for e in events if e["type"] == "stage"]
        assert stages[:3] == ["route", "search", "answer"]
        done = events[-1]
        assert done["type"] == "done" and done["reply"]["kind"] == "answer" and done["reply"]["citations"]
        history = c.get("/api/chat/history", headers=h).json()
        assert history[-1]["text"] == done["reply"]["reply"]  # stored like any other turn


def test_meal_reports_its_own_stage(env, fake_llm):
    from tulpar_ai.api.app import app

    with TestClient(app) as c:
        token = c.post("/api/auth/demo-login", json={"role": "client"}).json()["token"]
        r = c.post("/api/chat/stream", data={"text": "Съел 200 г плова и чай"},
                   headers={"Authorization": f"Bearer {token}"})
        events = _events(r.text)
        assert [e["stage"] for e in events if e["type"] == "stage"] == ["route", "meal"]
        assert events[-1]["reply"]["kind"] == "meal_card"


def test_empty_message_is_refused_before_streaming(env, fake_llm):
    from tulpar_ai.api.app import app

    with TestClient(app) as c:
        token = c.post("/api/auth/demo-login", json={"role": "client"}).json()["token"]
        r = c.post("/api/chat/stream", data={"text": "  "}, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 422
