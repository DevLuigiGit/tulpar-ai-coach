"""One entry point for every LLM call: provider fallback chain, JSON mode, tracing.

Why a thin wrapper instead of a framework client:
  * Ollama Cloud and Groq take JSON mode differently (`format: "json"` vs `response_format`);
  * the fallback must be visible in traces without leaking model names to users;
  * tests replace the whole thing with a deterministic fake (`set_fake`).

Every call is a LangSmith run of type `llm` with provider, model, token usage and a fallback flag.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from .config import get_settings


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    data: dict | list | None = None
    provider: str = ""
    model: str = ""
    fallback_used: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    errors: list[str] = field(default_factory=list)
    hedged: bool = False  # the next provider was asked too because this role's first one was slow (LLM_HEDGE)


FakeFn = Callable[..., Awaitable[str] | str]
_fake: FakeFn | None = None
_recorder: list | None = None


class record:
    """`with llm.record() as calls:` collects every LLMResult made inside (evals: tokens, latency, fallbacks)."""

    def __enter__(self) -> list:
        global _recorder
        self._prev, _recorder = _recorder, []
        return _recorder

    def __exit__(self, *exc) -> None:
        global _recorder
        _recorder = self._prev


def set_fake(fn: FakeFn | None) -> None:
    """Tests: route every call to `fn(role=, system=, user=, images=, json_mode=)` → str."""
    global _fake
    _fake = fn


def parse_json(text: str) -> Any:
    """First JSON value in the reply. Models add ```fences, prose before, or a second object after — ignore both."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                return dec.raw_decode(t, i)[0]
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("no JSON value found", t, 0)


async def _ollama(model: str, system: str, user: str, *, images, json_mode, temperature, top_p, max_tokens) -> tuple[str, int, int]:
    s = get_settings()
    msg: dict[str, Any] = {"role": "user", "content": user}
    if images:
        msg["images"] = images
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, msg],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature},
    }
    if top_p is not None:
        body["options"]["top_p"] = top_p
    if max_tokens:
        body["options"]["num_predict"] = max_tokens
    if json_mode:
        body["format"] = "json"
    async with httpx.AsyncClient(timeout=s.llm_timeout_s) as c:
        r = await c.post(f"{s.ollama_url}/api/chat", json=body, headers={"Authorization": f"Bearer {s.ollama_api_key}"})
    if r.status_code >= 400:
        raise LLMError(f"ollama {r.status_code}: {r.text[:200]}")
    d = r.json()
    return d.get("message", {}).get("content", ""), int(d.get("prompt_eval_count") or 0), int(d.get("eval_count") or 0)


async def _groq(model: str, system: str, user: str, *, images, json_mode, temperature, top_p, max_tokens) -> tuple[str, int, int]:
    s = get_settings()
    content: Any = user
    if images:
        content = [{"type": "text", "text": user}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in images
        ]
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "temperature": temperature,
    }
    if top_p is not None:
        body["top_p"] = top_p
    if model.startswith("openai/gpt-oss"):
        # reasoning model: reasoning tokens count against the completion budget — keep it short and leave room
        body["reasoning_effort"] = "low"
        max_tokens = max(max_tokens or 0, 1024)
    if max_tokens:
        body["max_tokens"] = max_tokens
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    async with httpx.AsyncClient(timeout=s.llm_timeout_s) as c:
        r = await c.post(f"{s.groq_url}/chat/completions", json=body, headers={"Authorization": f"Bearer {s.groq_api_key}"})
    if r.status_code >= 400:
        raise LLMError(f"groq {r.status_code}: {r.text[:200]}")
    d = r.json()
    u = d.get("usage") or {}
    return d["choices"][0]["message"]["content"] or "", int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


_PROVIDERS = {"ollama": _ollama, "groq": _groq}


def trace_llm_inputs(inputs: dict) -> dict:
    """Photos go to the trace as size only: the base64 JPEG would be stored once per model tried in the chain,
    and uploads are deliberately not kept anywhere else."""
    images = inputs.get("images")
    if not images:
        return inputs
    return {**inputs, "images": [{"type": "image/jpeg", "bytes": _b64_size(b)} for b in images]}


def _b64_size(b64: str) -> int:
    return len(b64) * 3 // 4 - (len(b64) - len(b64.rstrip("=")))


@traceable(run_type="llm", name="llm", process_inputs=trace_llm_inputs)
async def _traced_call(provider: str, model: str, system: str, user: str, *, images, json_mode, temperature, top_p, max_tokens, fallback_used: bool):
    rt = get_current_run_tree()
    if rt is not None:
        rt.metadata.update({"ls_provider": provider, "ls_model_name": model, "fallback_used": fallback_used,
                            "ls_temperature": temperature, "ls_max_tokens": max_tokens})
    text, pt, ct = await _PROVIDERS[provider](
        model, system, user, images=images, json_mode=json_mode, temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )
    return {"text": text, "usage_metadata": {"input_tokens": pt, "output_tokens": ct, "total_tokens": pt + ct}}


def hedge_after(role: str) -> float:
    """Seconds after which a slow first provider gets company from the next one: LLM_HEDGE, e.g. «route:5,text:8».
    0 — no hedging for the role (vision, judge, guard: the guard has its own timeout and fails open)."""
    for part in (get_settings().llm_hedge or "").split(","):
        name, _, sec = part.partition(":")
        if name.strip() == role:
            try:
                return max(0.0, float(sec))
            except ValueError:
                return 0.0
    return 0.0


async def _attempt(provider: str, model: str, system: str, user: str, *, images, json_mode, temperature, top_p,
                   max_tokens, fallback_used: bool, errors: list[str]) -> LLMResult:
    t0 = time.perf_counter()
    out = await _traced_call(provider, model, system, user, images=images, json_mode=json_mode, temperature=temperature,
                             top_p=top_p, max_tokens=max_tokens, fallback_used=fallback_used)
    res = LLMResult(text=out["text"], provider=provider, model=model, fallback_used=fallback_used,
                    input_tokens=out["usage_metadata"]["input_tokens"],
                    output_tokens=out["usage_metadata"]["output_tokens"],
                    latency_ms=int((time.perf_counter() - t0) * 1000), errors=list(errors))
    if json_mode:
        res.data = parse_json(res.text)  # invalid JSON is a failure of this provider, like a 5xx
    return res


async def chat(
    role: str,
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    temperature: float = 0.2,
    top_p: float | None = None,
    max_tokens: int | None = None,
    images: list[str] | None = None,
    models: list[tuple[str, str]] | None = None,
) -> LLMResult:
    """Walk the provider chain for `role` (route|text|vision|judge) until one answers."""
    if _fake is not None:
        out = _fake(role=role, system=system, user=user, images=images, json_mode=json_mode)
        text = await out if hasattr(out, "__await__") else out
        res = LLMResult(text=text, provider="fake", model="fake")
        if json_mode:
            res.data = parse_json(text)
        return res

    s = get_settings()
    chain = models or s.chain(role)
    skipped: list[tuple[int, str]] = []  # providers without a key, by position in the chain
    errors: list[str] = []  # real failures so far: network, 4xx/5xx, invalid JSON
    ready: list[tuple[int, str, str]] = []  # (position in the chain, provider, model); a later position is a fallback
    for pos, (p, m) in enumerate(chain):
        if p in _PROVIDERS and s.provider_ready(p):
            ready.append((pos, p, m))
        else:
            skipped.append((pos, f"{p}: no key"))

    def before(pos: int) -> list[str]:
        return [msg for at, msg in skipped if at < pos] + errors
    hedge = hedge_after(role)
    kw = dict(images=images, json_mode=json_mode, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    i = 0
    while i < len(ready):
        # A failure moves on to the next provider at once, as before. A provider that is merely slow (Ollama Cloud
        # answered in 15–26 s now and then) gets the next one asked in parallel after `hedge` seconds; the first
        # answer wins and the other request is cancelled. Only the tail is affected: p95 of an answer is ~7 s.
        who: dict[asyncio.Future, str] = {}
        pos, provider, model = ready[i]
        first = asyncio.ensure_future(_attempt(provider, model, system, user, fallback_used=pos > 0 or bool(errors),
                                               errors=before(pos), **kw))
        who[first] = f"{provider}:{model}"
        pending: set[asyncio.Future] = {first}
        hedged = False
        try:
            if hedge and i + 1 < len(ready):
                done, _ = await asyncio.wait(pending, timeout=hedge)
                if not done:
                    bpos, bp, bm = ready[i + 1]
                    second = asyncio.ensure_future(_attempt(
                        bp, bm, system, user, fallback_used=True,
                        errors=before(bpos) + [f"{provider}:{model}: no answer after {hedge:g} s"], **kw))
                    who[second] = f"{bp}:{bm}"
                    pending.add(second)
                    hedged = True
            winner: LLMResult | None = None
            while pending and winner is None:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t.exception() is None and winner is None:
                        winner = t.result()
                    elif t.exception() is not None:  # network, 4xx/5xx, invalid JSON
                        e = t.exception()
                        errors.append(f"{who[t]}: {type(e).__name__}: {str(e)[:160]}")
        finally:
            for t in pending:  # the slower answer, or everything if the caller gave up (guard's wait_for)
                t.cancel()
        if winner is not None:
            winner.hedged = hedged
            if _recorder is not None:
                _recorder.append({"role": role, "provider": winner.provider, "model": winner.model,
                                  "in": winner.input_tokens, "out": winner.output_tokens, "ms": winner.latency_ms,
                                  "fallback": winner.fallback_used, "hedged": hedged})
            return winner
        i += 2 if hedged else 1
    raise LLMError("all providers failed: " + " | ".join([msg for _, msg in skipped] + errors))


async def json_call(role: str, system: str, user: str, **kw) -> tuple[dict, LLMResult]:
    res = await chat(role, system, user, json_mode=True, **kw)
    if not isinstance(res.data, dict):
        raise LLMError(f"expected a JSON object, got {type(res.data).__name__}")
    return res.data, res
