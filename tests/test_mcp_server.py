"""tulpar-mcp speaks real MCP over stdio and its tools drive the service API."""

import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"list_clients", "get_client_context", "search_exercises", "propose_program_change", "list_pending_proposals"}


async def test_stdio_handshake_and_tools():
    env = {**os.environ, "LANGSMITH_TRACING": "false", "SERVICE_URL": "http://127.0.0.1:9"}
    params = StdioServerParameters(command=sys.executable, args=["-m", "tulpar_ai.mcp_server"], env=env, cwd=str(ROOT))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            assert init.server_info.name == "tulpar-mcp"
            tools = {t.name: t for t in (await s.list_tools()).tools}
            assert set(tools) == EXPECTED
            assert "client_id" in json.dumps(tools["propose_program_change"].input_schema)


async def test_propose_waits_for_the_draft(monkeypatch):
    from tulpar_ai.mcp_server import server

    calls = []
    states = iter(["drafting", "pending"])

    async def fake_call(method, path, **kw):
        calls.append((method, path))
        status = next(states, "pending")
        return {"id": "p1", "status": status, "draft": {"summary": "замена приседа"},
                "changes": [{"day": "Ноги", "was": "Присед 4×8", "becomes": "Ягодичный мост 4×10", "reason": "колено"}],
                "violations": []}

    monkeypatch.setattr(server, "_call", fake_call)
    monkeypatch.setattr(server.asyncio, "sleep", lambda s: _noop())
    out = await server.propose_program_change("c1", "облегчи из-за колена", wait_seconds=6)
    assert out["status"] == "pending" and out["changes"][0]["becomes"].startswith("Ягодичный")
    assert calls[0] == ("POST", "/api/trainer/proposals") and calls[-1] == ("GET", "/api/proposals/p1")


async def _noop():
    return None
