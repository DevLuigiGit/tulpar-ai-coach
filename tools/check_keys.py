"""Check every external dependency with one cheap call. Prints OK/FAIL per service, never the keys.

    .venv/bin/python tools/check_keys.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from tulpar_ai import llm  # noqa: E402
from tulpar_ai.config import get_settings  # noqa: E402


async def check(name, coro):
    try:
        detail = await coro
        print(f"OK    {name}  {detail or ''}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name}  {type(e).__name__}: {str(e)[:160]}")
        return False


async def ollama_models():
    s = get_settings()
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{s.ollama_url}/api/tags", headers={"Authorization": f"Bearer {s.ollama_api_key}"})
    r.raise_for_status()
    names = sorted(m["name"] for m in r.json().get("models", []))
    return f"{len(names)} моделей: " + ", ".join(names[:12])


async def groq_models():
    s = get_settings()
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{s.groq_url}/models", headers={"Authorization": f"Bearer {s.groq_api_key}"})
    r.raise_for_status()
    names = sorted(m["id"] for m in r.json().get("data", []))
    return f"{len(names)} моделей: " + ", ".join(names[:15])


async def role(r):
    res = await llm.chat(r, "Верни JSON {\"ok\": true}", "ping", json_mode=True, temperature=0, max_tokens=20)
    return f"{res.provider}:{res.model} {res.latency_ms} мс" + (" (резервный провайдер)" if res.fallback_used else "")


async def jina():
    from tulpar_ai.rag.embed import JinaEmbedder, JinaReranker

    v = await JinaEmbedder().embed(["жим лёжа"], task="retrieval.query")
    rr = await JinaReranker().rerank("жим", ["жим лёжа", "бег"], 1)
    return f"dim={len(v[0])}, rerank ok ({rr[0][1]:.2f})"


async def langsmith():
    key = os.environ.get("LANGSMITH_API_KEY")
    if not key:
        raise RuntimeError("LANGSMITH_API_KEY не задан")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get("https://api.smith.langchain.com/api/v1/sessions?limit=1", headers={"x-api-key": key})
    r.raise_for_status()
    return f"проект {os.environ.get('LANGSMITH_PROJECT', 'default')}, tracing={os.environ.get('LANGSMITH_TRACING')}"


async def telegram():
    s = get_settings()
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"https://api.telegram.org/bot{s.telegram_bot_token}/getMe")
    r.raise_for_status()
    me = r.json()["result"]
    return f"@{me['username']}; тренеры: {', '.join(sorted(s.trainer_tg_ids)) or 'не заданы'}"


async def main():
    s = get_settings()
    checks = []
    if s.ollama_api_key:
        checks.append(("Ollama Cloud: список моделей", ollama_models()))
    if s.groq_api_key:
        checks.append(("Groq: список моделей", groq_models()))
    for r in ("route", "text", "vision", "judge"):
        checks.append((f"LLM роль {r}", role(r)))
    checks.append(("Jina embeddings + rerank", jina()))
    checks.append(("LangSmith", langsmith()))
    if s.telegram_bot_token:
        checks.append(("Telegram bot", telegram()))
    ok = [await check(n, c) for n, c in checks]
    print(f"\n{sum(ok)}/{len(ok)} проверок прошли. Режим: {s.tulpar_mode}.")


if __name__ == "__main__":
    asyncio.run(main())
