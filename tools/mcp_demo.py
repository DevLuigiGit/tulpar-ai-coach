"""MCP demo without an MCP host: the official `mcp` Python client starts tulpar-mcp over stdio, exactly as
Claude Desktop / Claude Code would, lists its tools and calls them as a trainer's assistant would.

    .venv/bin/python tools/mcp_demo.py [--base http://localhost:8089] [--client Айдар]

The service must be running at --base. INTERNAL_SECRET is taken by the server from env / .env and is never
printed. Note: propose_program_change creates a real draft in the trainer's queue (it is not applied).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
REQUEST = "Облегчи план из-за боли в колене: убери ударную и глубокую осевую нагрузку на колено"


def unpack(result: Any) -> Any:
    """Tool result → Python value: structured content when the server sends it, else the JSON text block."""
    if getattr(result, "is_error", False):
        raise RuntimeError(" ".join(getattr(c, "text", "") for c in result.content)[:300])
    sc = getattr(result, "structured_content", None)
    if isinstance(sc, dict) and set(sc) == {"result"}:
        return sc["result"]
    if sc is not None:
        return sc
    texts = [c.text for c in result.content if getattr(c, "text", None)]
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            return texts[0]
    return [json.loads(t) for t in texts]


def fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())


def summarize(name: str, value: Any) -> list[str]:
    """A few human lines per tool, so the transcript stays readable."""
    if name == "list_clients":
        return [f"{c.get('name')} ({str(c.get('id'))[:8]}): цель {c.get('goal')}, место {c.get('place')}, "
                f"ограничение по здоровью: {'да' if c.get('injury') or c.get('has_injury') else 'нет'}"
                for c in value]
    if name == "get_client_context":
        plan = value.get("active_plan") or {}
        note = value.get("trainer_note") or {}
        days = [f"{d.get('title')} ({len(d.get('exercises', []))} упр.)" for d in plan.get("days", [])]
        return [f"заметка тренера: {note.get('body') if isinstance(note, dict) else note}",
                f"план «{plan.get('title')}»: " + "; ".join(days)]
    if name == "search_exercises":
        return [f"найдено {len(value)}: " + ", ".join(e.get("name", "?") for e in value[:8])
                + (" …" if len(value) > 8 else "")]
    if name == "propose_program_change":
        lines = [f"статус: {value.get('status')}, черновик {str(value.get('proposal_id'))[:8]}",
                 f"summary: {value.get('summary')}"]
        lines += [f"{c['day']}: {c.get('was') or '—'} → {c['becomes']} ({c.get('reason') or ''})".rstrip(" ()")
                  for c in value.get("changes") or []]
        lines += [f"предупреждение валидатора: {w}" for w in value.get("warnings") or []]
        return lines
    if name == "list_pending_proposals":
        return [f"в очереди {len(value)}: " + "; ".join(f"{p['kind']} {p['status']} — {p.get('client')}" for p in value)]
    return [json.dumps(value, ensure_ascii=False)[:300]]


async def demo(base: str, client_name: str, out: Callable[[str], None] = print) -> None:
    env = {**os.environ, "SERVICE_URL": base, "TELEGRAM_BOT_TOKEN": "", "PYTHONPATH": str(ROOT)}
    params = StdioServerParameters(command=sys.executable, args=["-m", "tulpar_ai.mcp_server"], env=env, cwd=str(ROOT))
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        init = await s.initialize()
        tools = (await s.list_tools()).tools
        out(f"сервер: {init.server_info.name}, инструментов: {len(tools)}")
        for t in tools:
            out(f"  - {t.name}: {(t.description or '').splitlines()[0][:110]}")

        async def call(name: str, **args) -> Any:
            t0 = time.perf_counter()
            value = unpack(await s.call_tool(name, args, read_timeout_seconds=240))
            out(f"\n→ {name}({fmt_args(args)})  [{time.perf_counter() - t0:.1f} с]")
            for line in summarize(name, value):
                out(f"  {line}")
            return value

        clients = await call("list_clients")
        client = next((c for c in clients if client_name.lower() in str(c.get("name", "")).lower()), None)
        if client is None:
            raise RuntimeError(f"клиент «{client_name}» не найден")
        await call("get_client_context", client_id=client["id"])
        await call("search_exercises", muscle_group="ноги", avoid_problems_with="колено")
        await call("propose_program_change", client_id=client["id"], request=REQUEST, wait_seconds=180)
        await call("list_pending_proposals")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://localhost:8089")
    ap.add_argument("--client", default="Айдар")
    args = ap.parse_args()
    print(f"MCP-демо: tulpar-mcp (stdio) → {args.base}")
    try:
        asyncio.run(demo(args.base, args.client))
    except Exception as e:  # noqa: BLE001
        print(f"ОШИБКА: {type(e).__name__}: {str(e)[:300]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
