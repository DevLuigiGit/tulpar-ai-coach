"""One entry point for every LLM call: provider fallback chain, JSON mode, tracing.

Why a thin wrapper instead of a framework client:
  * Ollama Cloud and Groq take JSON mode differently (`format: "json"` vs `response_format`);
  * the fallback must be visible in traces without leaking model names to users;
  * tests replace the whole thing with a deterministic fake (`set_fake`).

Every call is a LangSmith run of type `llm` with provider, model, token usage and a fallback flag.
`chat_stream` is the same chain with the reply delivered piece by piece (web chat, POST /api/chat/stream): it falls back
to the next provider only before the first piece arrives; after that a failure is final.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable

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
    first_delta_ms: int | None = None  # streaming only: time to the first piece of text


FakeFn = Callable[..., Awaitable[str] | str]
FakeStreamFn = Callable[..., Iterable[str] | AsyncIterator[str]]
DeltaFn = Callable[[str], Awaitable[None] | None]
_fake: FakeFn | None = None
_fake_stream: FakeStreamFn | None = None
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


def set_fake_stream(fn: FakeStreamFn | None) -> None:
    """Tests: `chat_stream` takes its pieces from `fn(role=, system=, user=, images=, json_mode=)` → iterable of str.
    Without it, a `set_fake` reply is cut into small pieces, so every fake LLM also streams."""
    global _fake_stream
    _fake_stream = fn


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


def _ollama_body(model: str, system: str, user: str, *, images, json_mode, temperature, top_p, max_tokens,
                 stream: bool = False) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "user", "content": user}
    if images:
        msg["images"] = images
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, msg],
        "stream": stream,
        "think": False,
        "options": {"temperature": temperature},
    }
    if top_p is not None:
        body["options"]["top_p"] = top_p
    if max_tokens:
        body["options"]["num_predict"] = max_tokens
    if json_mode:
        body["format"] = "json"
    return body


async def _ollama(model: str, system: str, user: str, *, images, json_mode, temperature, top_p, max_tokens) -> tuple[str, int, int]:
    s = get_settings()
    body = _ollama_body(model, system, user, images=images, json_mode=json_mode, temperature=temperature, top_p=top_p,
                        max_tokens=max_tokens)
    async with httpx.AsyncClient(timeout=s.llm_timeout_s) as c:
        r = await c.post(f"{s.ollama_url}/api/chat", json=body, headers={"Authorization": f"Bearer {s.ollama_api_key}"})
    if r.status_code >= 400:
        raise LLMError(f"ollama {r.status_code}: {r.text[:200]}")
    d = r.json()
    return d.get("message", {}).get("content", ""), int(d.get("prompt_eval_count") or 0), int(d.get("eval_count") or 0)


async def _ollama_stream(model: str, system: str, user: str, *, images, json_mode, temperature, top_p, max_tokens,
                         on_delta: DeltaFn) -> tuple[str, int, int]:
    """NDJSON: one {"message": {"content": piece}} per line; the last line has done=true and the token counts."""
    s = get_settings()
    body = _ollama_body(model, system, user, images=images, json_mode=json_mode, temperature=temperature, top_p=top_p,
                        max_tokens=max_tokens, stream=True)
    parts: list[str] = []
    pt = ct = 0
    async with httpx.AsyncClient(timeout=s.llm_timeout_s) as c:
        async with c.stream("POST", f"{s.ollama_url}/api/chat", json=body,
                            headers={"Authorization": f"Bearer {s.ollama_api_key}"}) as r:
            if r.status_code >= 400:
                raise LLMError(f"ollama {r.status_code}: {(await r.aread()).decode(errors='replace')[:200]}")
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("error"):
                    raise LLMError(f"ollama stream: {str(d['error'])[:200]}")
                piece = (d.get("message") or {}).get("content") or ""
                if piece:
                    parts.append(piece)
                    await _call(on_delta, piece)
                if d.get("done"):
                    pt, ct = int(d.get("prompt_eval_count") or 0), int(d.get("eval_count") or 0)
    return "".join(parts), pt, ct


def _groq_body(model: str, system: str, user: str, *, images, json_mode, temperature, top_p, max_tokens,
               stream: bool = False) -> dict[str, Any]:
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
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


async def _groq(model: str, system: str, user: str, *, images, json_mode, temperature, top_p, max_tokens) -> tuple[str, int, int]:
    s = get_settings()
    body = _groq_body(model, system, user, images=images, json_mode=json_mode, temperature=temperature, top_p=top_p,
                      max_tokens=max_tokens)
    async with httpx.AsyncClient(timeout=s.llm_timeout_s) as c:
        r = await c.post(f"{s.groq_url}/chat/completions", json=body, headers={"Authorization": f"Bearer {s.groq_api_key}"})
    if r.status_code >= 400:
        raise LLMError(f"groq {r.status_code}: {r.text[:200]}")
    d = r.json()
    u = d.get("usage") or {}
    return d["choices"][0]["message"]["content"] or "", int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


async def _groq_stream(model: str, system: str, user: str, *, images, json_mode, temperature, top_p, max_tokens,
                       on_delta: DeltaFn) -> tuple[str, int, int]:
    """OpenAI-style SSE: `data: {chunk}` lines until `data: [DONE]`; reasoning pieces are not part of the reply."""
    s = get_settings()
    body = _groq_body(model, system, user, images=images, json_mode=json_mode, temperature=temperature, top_p=top_p,
                      max_tokens=max_tokens, stream=True)
    parts: list[str] = []
    usage: dict = {}
    async with httpx.AsyncClient(timeout=s.llm_timeout_s) as c:
        async with c.stream("POST", f"{s.groq_url}/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {s.groq_api_key}"}) as r:
            if r.status_code >= 400:
                raise LLMError(f"groq {r.status_code}: {(await r.aread()).decode(errors='replace')[:200]}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                d = json.loads(data)
                if d.get("error"):
                    raise LLMError(f"groq stream: {str(d['error'])[:200]}")
                usage = d.get("usage") or (d.get("x_groq") or {}).get("usage") or usage
                for ch in d.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content") or ""
                    if piece:
                        parts.append(piece)
                        await _call(on_delta, piece)
    return "".join(parts), int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


async def _call(fn: DeltaFn, piece: str) -> None:
    out = fn(piece)
    if hasattr(out, "__await__"):
        await out


_PROVIDERS = {"ollama": _ollama, "groq": _groq}
_STREAMERS = {"ollama": _ollama_stream, "groq": _groq_stream}


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


def _trace_stream_inputs(inputs: dict) -> dict:
    return trace_llm_inputs({k: v for k, v in inputs.items() if k != "on_delta"})


@traceable(run_type="llm", name="llm", process_inputs=_trace_stream_inputs)
async def _traced_stream(provider: str, model: str, system: str, user: str, *, images, json_mode, temperature, top_p,
                         max_tokens, fallback_used: bool, on_delta: DeltaFn):
    """The same `llm` run as `_traced_call`, plus `streamed` and the time to the first piece (also a `new_token`
    event, which LangSmith shows as time to first token)."""
    rt = get_current_run_tree()
    if rt is not None:
        rt.metadata.update({"ls_provider": provider, "ls_model_name": model, "fallback_used": fallback_used,
                            "ls_temperature": temperature, "ls_max_tokens": max_tokens, "streamed": True})
    t0, first = time.perf_counter(), []

    async def relay(piece: str) -> None:
        if not first:
            first.append(int((time.perf_counter() - t0) * 1000))
            if rt is not None:
                rt.metadata["first_delta_ms"] = first[0]
                try:
                    rt.add_event({"name": "new_token", "time": datetime.now(timezone.utc).isoformat()})
                except Exception:  # a tracing detail must never break a reply
                    pass
        await _call(on_delta, piece)

    text, pt, ct = await _STREAMERS[provider](
        model, system, user, images=images, json_mode=json_mode, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, on_delta=relay)
    return {"text": text, "first_delta_ms": first[0] if first else None,
            "usage_metadata": {"input_tokens": pt, "output_tokens": ct, "total_tokens": pt + ct}}


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


class StreamBroken(LLMError):
    """The stream failed after the reply had started: no fallback, the caller ends the turn cleanly."""


async def chat_stream(
    role: str,
    system: str,
    user: str,
    *,
    on_delta: DeltaFn,
    json_mode: bool = False,
    temperature: float = 0.2,
    top_p: float | None = None,
    max_tokens: int | None = None,
    images: list[str] | None = None,
    models: list[tuple[str, str]] | None = None,
) -> LLMResult:
    """`chat` with the reply passed to `on_delta` piece by piece. The chain falls back to the next provider only while
    nothing has been delivered; once a piece is out, any error (network, provider, invalid JSON) raises StreamBroken."""
    if _fake_stream is not None or _fake is not None:
        return await _fake_chat_stream(role, system, user, on_delta=on_delta, images=images, json_mode=json_mode)

    s = get_settings()
    chain = models or s.chain(role)
    errors: list[str] = []
    for i, (provider, model) in enumerate(chain):
        if provider not in _STREAMERS or not s.provider_ready(provider):
            errors.append(f"{provider}: no key")
            continue
        started: list[bool] = []

        async def relay(piece: str, started=started) -> None:
            started.append(True)
            await _call(on_delta, piece)

        t0 = time.perf_counter()
        try:
            out = await _traced_stream(provider, model, system, user, images=images, json_mode=json_mode,
                                       temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                       fallback_used=bool(errors), on_delta=relay)
            res = LLMResult(text=out["text"], provider=provider, model=model, fallback_used=i > 0 or bool(errors),
                            input_tokens=out["usage_metadata"]["input_tokens"],
                            output_tokens=out["usage_metadata"]["output_tokens"],
                            latency_ms=int((time.perf_counter() - t0) * 1000), errors=errors,
                            first_delta_ms=out["first_delta_ms"])
            if json_mode:
                res.data = parse_json(res.text)
            if _recorder is not None:
                _recorder.append({"role": role, "provider": provider, "model": model, "in": res.input_tokens,
                                  "out": res.output_tokens, "ms": res.latency_ms, "fallback": res.fallback_used,
                                  "stream": True, "first_delta_ms": res.first_delta_ms})
            return res
        except Exception as e:
            errors.append(f"{provider}:{model}: {type(e).__name__}: {str(e)[:160]}")
            if started:
                raise StreamBroken("stream failed after the first piece: " + errors[-1]) from e
    raise LLMError("all providers failed: " + " | ".join(errors))


async def _fake_chat_stream(role: str, system: str, user: str, *, on_delta: DeltaFn, images, json_mode) -> LLMResult:
    kw = dict(role=role, system=system, user=user, images=images, json_mode=json_mode)
    if _fake_stream is not None:
        pieces = _fake_stream(**kw)
    else:
        out = _fake(**kw)
        text = await out if hasattr(out, "__await__") else out
        pieces = [text[i:i + 7] for i in range(0, len(text), 7)]
    parts: list[str] = []
    try:
        if hasattr(pieces, "__aiter__"):
            async for p in pieces:
                parts.append(p)
                await _call(on_delta, p)
        else:
            for p in pieces:
                parts.append(p)
                await _call(on_delta, p)
    except Exception as e:
        if parts:
            raise StreamBroken(f"fake stream broke: {e}") from e
        raise LLMError(f"fake stream failed: {e}") from e
    res = LLMResult(text="".join(parts), provider="fake", model="fake", first_delta_ms=0 if parts else None)
    if json_mode:
        try:
            res.data = parse_json(res.text)
        except json.JSONDecodeError as e:
            raise StreamBroken(f"invalid JSON after streaming: {e}") from e
    return res


async def json_call(role: str, system: str, user: str, **kw) -> tuple[dict, LLMResult]:
    res = await chat(role, system, user, json_mode=True, **kw)
    if not isinstance(res.data, dict):
        raise LLMError(f"expected a JSON object, got {type(res.data).__name__}")
    return res.data, res
