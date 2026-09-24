"""tulpar-mcp — MCP server that gives a trainer's AI assistant (Claude Desktop / Claude Code) the coach's tools.

Why MCP and not «just the REST API»: the same capabilities the web app and the bot use become typed,
self-describing tools that any MCP host discovers and calls on its own; the trainer scope and the
masking of client data are enforced in one place (the service), and no host needs custom glue code.

Transport: stdio (Claude Desktop starts it as a subprocess). It is a thin client of the running
service (SERVICE_URL), authenticated with INTERNAL_SECRET and acting as TULPAR_TRAINER_ID.
Logs go to stderr only: anything printed to stdout would corrupt the protocol.

    python -m tulpar_ai.mcp_server
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx
from mcp.server.mcpserver import MCPServer

from ..config import get_settings

server = MCPServer(
    name="tulpar-mcp",
    instructions=(
        "Инструменты AI-коуча фитнес-платформы Tulpar в режиме тренера. Сначала list_clients, затем "
        "get_client_context по id клиента. Изменения программы только предлагай через propose_program_change: "
        "применяет их тренер в очереди. Для составления изменений используй Skill tulpar-program-builder."
    ),
)


def _log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


async def _trainer_id(c: httpx.AsyncClient) -> str:
    tid = os.environ.get("TULPAR_TRAINER_ID")
    if tid:
        return tid
    s = get_settings()
    r = await c.post("/api/auth/demo-login", json={"role": "trainer"})  # demo mode: the demo trainer
    r.raise_for_status()
    return r.json()["user"]["id"]


async def _call(method: str, path: str, **kw):
    s = get_settings()
    async with httpx.AsyncClient(base_url=s.service_url, timeout=120) as c:
        headers = {"X-Internal-Secret": s.internal_secret, "X-Act-As": await _trainer_id(c)}
        r = await c.request(method, path, headers=headers, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
        return r.json()


@server.tool()
async def list_clients() -> list[dict]:
    """Клиенты тренера: id, имя, цель, уровень, место тренировок, есть ли ограничение по здоровью, активная программа."""
    return await _call("GET", "/api/trainer/clients")


@server.tool()
async def get_client_context(client_id: str) -> dict:
    """Контекст клиента для решений по программе: профиль (цель, уровень, место, дни), заметка тренера об
    ограничениях, активный план с id дней и упражнений, средние КБЖУ за 14 дней, последние тренировки и вес."""
    return await _call("GET", f"/api/trainer/clients/{client_id}/context")


@server.tool()
async def search_exercises(query: str = "", muscle_group: str = "", equipment: str = "",
                           avoid_problems_with: str = "") -> list[dict]:
    """Поиск в каталоге упражнений Tulpar. muscle_group: ноги, спина, грудь, плечи, руки, пресс, кор, кардио,
    растяжка. equipment: bodyweight, dumbbell, barbell, machine, cable, band, kettlebell.
    avoid_problems_with — текст ограничения («колено», «поясница»): упражнения с такими противопоказаниями
    исключаются."""
    params = {k: v for k, v in {"q": query, "muscle_group": muscle_group, "equipment": equipment,
                                "avoid": avoid_problems_with}.items() if v}
    return await _call("GET", "/api/exercises", params=params)


@server.tool()
async def propose_program_change(client_id: str, request: str, wait_seconds: int = 90) -> dict:
    """Попросить AI-коуча подготовить черновик изменения программы клиента. Черновик собирается по Skill
    tulpar-program-builder, проходит валидатор и попадает в очередь тренера; программа НЕ меняется, пока тренер
    не примет черновик. Возвращает статус и список изменений «было → станет»."""
    p = await _call("POST", "/api/trainer/proposals", json={"client_id": client_id, "request": request})
    for _ in range(max(1, wait_seconds // 3)):
        if p["status"] != "drafting":
            break
        await asyncio.sleep(3)
        p = await _call("GET", f"/api/proposals/{p['id']}")
    return {"proposal_id": p["id"], "status": p["status"], "summary": (p.get("draft") or {}).get("summary"),
            "changes": p.get("changes"), "warnings": [v["message"] for v in (p.get("violations") or [])],
            "next_step": "Тренер принимает или отклоняет черновик в очереди AI Coach."}


@server.tool()
async def list_pending_proposals() -> list[dict]:
    """Очередь тренера: черновики изменений программ, ждущие решения, и эскалации (сообщения клиентов о боли,
    лекарствах и т. п., на которые бот не отвечает сам)."""
    items = await _call("GET", "/api/queue")
    return [{"id": p["id"], "kind": p["kind"], "status": p["status"], "client": p.get("client_name"),
             "request": p.get("request"), "summary": (p.get("draft") or {}).get("summary"),
             "changes": p.get("changes")} for p in items]


def main() -> None:
    _log(f"tulpar-mcp → {get_settings().service_url}")
    server.run("stdio")


if __name__ == "__main__":
    main()
