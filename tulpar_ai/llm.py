"""One entry point for every LLM call: provider fallback chain, JSON mode, tracing.

Why a thin wrapper instead of a framework client:
  * Ollama Cloud and Groq take JSON mode differently (`format: "json"` vs `response_format`);
  * the fallback must be visible in traces without leaking model names to users;
  * tests replace the whole thing with a deterministic fake (`set_fake`).

Every call is a LangSmith run of type `llm` with provider, model, token usage and a fallback flag.
"""

from __future__ import annotations

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
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", t, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(1))


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


@traceable(run_type="llm", name="llm")
async def _traced_call(provider: str, model: str, system: str, user: str, *, images, json_mode, temperature, top_p, max_tokens, fallback_used: bool):
    rt = get_current_run_tree()
    if rt is not None:
        rt.metadata.update({"ls_provider": provider, "ls_model_name": model, "fallback_used": fallback_used,
                            "ls_temperature": temperature, "ls_max_tokens": max_tokens})
    text, pt, ct = await _PROVIDERS[provider](
        model, system, user, images=images, json_mode=json_mode, temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )
    return {"text": text, "usage_metadata": {"input_tokens": pt, "output_tokens": ct, "total_tokens": pt + ct}}


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
    errors: list[str] = []
    for i, (provider, model) in enumerate(chain):
        if provider not in _PROVIDERS or not s.provider_ready(provider):
            errors.append(f"{provider}: no key")
            continue
        t0 = time.perf_counter()
        try:
            out = await _traced_call(provider, model, system, user, images=images, json_mode=json_mode,
                                     temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                     fallback_used=bool(errors))
            res = LLMResult(text=out["text"], provider=provider, model=model, fallback_used=i > 0 or bool(errors),
                            input_tokens=out["usage_metadata"]["input_tokens"],
                            output_tokens=out["usage_metadata"]["output_tokens"],
                            latency_ms=int((time.perf_counter() - t0) * 1000), errors=errors)
            if json_mode:
                res.data = parse_json(res.text)
            if _recorder is not None:
                _recorder.append({"role": role, "provider": provider, "model": model, "in": res.input_tokens,
                                  "out": res.output_tokens, "ms": res.latency_ms, "fallback": res.fallback_used})
            return res
        except Exception as e:  # network, 4xx/5xx, invalid JSON → next provider
            errors.append(f"{provider}:{model}: {type(e).__name__}: {str(e)[:160]}")
    raise LLMError("all providers failed: " + " | ".join(errors))


async def json_call(role: str, system: str, user: str, **kw) -> tuple[dict, LLMResult]:
    res = await chat(role, system, user, json_mode=True, **kw)
    if not isinstance(res.data, dict):
        raise LLMError(f"expected a JSON object, got {type(res.data).__name__}")
    return res.data, res
