"""Embeddings and reranking.

Primary: Jina (multilingual — questions are Russian, the WHO PDF is English; one vector space for both).
Fallback: a local hashing embedder + lexical reranker, so RAG still works with no keys at all (CI, a
mentor's laptop). The fallback is also a baseline arm in the RAG A/B.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from typing import Protocol

import httpx
from langsmith import traceable

from ..config import get_settings
from .emb_cache import cache_key, get_embedding_cache

JINA = "https://api.jina.ai/v1"
USAGE = {"requests": 0, "tokens": 0}  # Jina calls made by this process (evals report the quota a run spent)


async def _post(c: httpx.AsyncClient, path: str, body: dict, attempts: int = 5) -> dict:
    """Jina free tier answers 429 under bursts: back off (honouring Retry-After) instead of failing the turn."""
    s = get_settings()
    delay = 1.0
    for i in range(attempts):
        r = await c.post(f"{JINA}{path}", headers={"Authorization": f"Bearer {s.jina_api_key}"}, json=body)
        if r.status_code not in (429, 500, 502, 503, 504) or i == attempts - 1:
            r.raise_for_status()
            data = r.json()
            USAGE["requests"] += 1
            USAGE["tokens"] += int((data.get("usage") or {}).get("total_tokens") or 0)
            return data
        wait = float(r.headers.get("retry-after") or delay)
        await asyncio.sleep(min(wait, 20.0))
        delay *= 2
    raise RuntimeError("unreachable")


class Embedder(Protocol):
    id: str
    dim: int

    async def embed(self, texts: list[str], task: str) -> list[list[float]]: ...


class Reranker(Protocol):
    id: str

    async def rerank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]: ...


def trace_embed_inputs(inputs: dict) -> dict:
    """Trace the batch size and a preview, not the whole corpus: the index build embeds every chunk at once."""
    texts = inputs.get("texts") or []
    return {"task": inputs.get("task"), "count": len(texts), "texts": [t[:300] for t in texts[:3]]}


def trace_embed_outputs(vectors: list[list[float]] | None) -> dict:
    """Vectors are unreadable in a trace and heavy: 1579 chunks were a 20 MB run output per index build."""
    return {"vectors": len(vectors or []), "dim": len(vectors[0]) if vectors else 0}


def trace_late_inputs(inputs: dict) -> dict:
    groups = inputs.get("groups") or []
    return {"task": inputs.get("task"), "groups": len(groups), "chunks": sum(len(g) for g in groups),
            "first": [t[:200] for t in (groups[0] if groups else [])[:2]]}


def trace_late_outputs(groups: list[list[list[float]]] | None) -> dict:
    return {"groups": len(groups or []), "vectors": sum(len(g) for g in groups or [])}


class JinaEmbedder:
    id = "jina3"
    dim = 1024
    matryoshka = True  # jina-embeddings-v3 is trained so that a prefix of the vector is a valid smaller embedding
    late_chunking = True

    @traceable(run_type="embedding", name="jina_embed", process_inputs=trace_embed_inputs,
               process_outputs=trace_embed_outputs)
    async def embed(self, texts: list[str], task: str) -> list[list[float]]:
        s = get_settings()
        cache = get_embedding_cache(task)
        keys = [cache_key(s.embed_model, task, t) for t in texts] if cache is not None else []
        out: list[list[float] | None] = [cache.get(k) for k in keys] if cache is not None else [None] * len(texts)
        todo = [i for i, v in enumerate(out) if v is None]
        if todo:
            async with httpx.AsyncClient(timeout=60) as c:
                for b in range(0, len(todo), 64):
                    batch = todo[b:b + 64]
                    data = await _post(c, "/embeddings", {"model": s.embed_model, "task": task,
                                                          "input": [texts[i] for i in batch]})
                    for i, d in zip(batch, sorted(data["data"], key=lambda d: d["index"])):
                        out[i] = d["embedding"]
                        if cache is not None:
                            cache.put(keys[i], d["embedding"])
        return out  # type: ignore[return-value]

    @traceable(run_type="embedding", name="jina_embed_late", process_inputs=trace_late_inputs,
               process_outputs=trace_late_outputs)
    async def embed_late(self, groups: list[list[str]], task: str) -> list[list[list[float]]]:
        """Late chunking: the chunks of one group are read as ONE document, then pooled back per chunk.

        Every chunk vector then carries its neighbours' context («they» → the adults of the previous chunk). Jina
        concatenates the whole `input` of a request, so each group is its own request and must fit the model's
        8192-token context; the caller sizes groups by characters (RAG_LATE_MAX_CHARS) and a group the API still
        rejects as too long is split in half and retried.
        """
        s = get_settings()
        cache = get_embedding_cache(task)
        out: list[list[list[float]]] = [[] for _ in groups]
        # A lone chunk (an exercise card) has no neighbours: its late vector is its plain vector, so all of them go
        # together in ordinary batches of 64 instead of one request each.
        singles = [i for i, g in enumerate(groups) if len(g) == 1]
        for i, v in zip(singles, await self.embed([groups[i][0] for i in singles], task) if singles else []):
            out[i] = [v]
        async with httpx.AsyncClient(timeout=120) as c:
            for i, group in enumerate(groups):
                if len(group) > 1:
                    out[i] = await self._late_group(c, group, task, cache, s.embed_model)
        return out

    async def _late_group(self, c: httpx.AsyncClient, group: list[str], task: str, cache, model: str) -> list[list[float]]:
        if len(group) == 1:  # a lone chunk has no neighbours: late chunking equals the plain embedding
            return await self.embed(group, task)
        digest = hashlib.sha256(json.dumps(group, ensure_ascii=False).encode()).hexdigest()
        keys = [cache_key(model, task, t, f"late:{digest}:{i}") for i, t in enumerate(group)] if cache is not None else []
        cached = [cache.get(k) for k in keys] if cache is not None else []
        if cache is not None and all(v is not None for v in cached):
            return cached  # type: ignore[return-value]
        try:
            data = await _post(c, "/embeddings", {"model": model, "task": task, "late_chunking": True, "input": group})
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in (400, 413, 422) or len(group) < 2:
                raise
            half = len(group) // 2  # longer than the context: two smaller documents
            return (await self._late_group(c, group[:half], task, cache, model)
                    + await self._late_group(c, group[half:], task, cache, model))
        vecs = [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
        for k, v in zip(keys, vecs):
            cache.put(k, v)
        return vecs


class JinaReranker:
    id = "jina-rerank"

    @traceable(run_type="tool", name="jina_rerank")
    async def rerank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        s = get_settings()
        async with httpx.AsyncClient(timeout=60) as c:
            data = await _post(c, "/rerank", {"model": s.rerank_model, "query": query, "documents": docs, "top_n": top_n})
        return [(x["index"], float(x["relevance_score"])) for x in data["results"]]


_WORD = re.compile(r"[0-9a-zа-яё]+")


def _norm(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


class LocalHashEmbedder:
    """Character 3–5-grams hashed into a fixed vector. Crude, deterministic, zero dependencies."""

    id = "local"
    dim = 768

    async def embed(self, texts: list[str], task: str) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in _WORD.findall(_norm(text)):
            w = f" {w} "
            for n in (3, 4, 5):
                for i in range(len(w) - n + 1):
                    h = int(hashlib.md5(w[i:i + n].encode()).hexdigest()[:8], 16)
                    v[h % self.dim] += 1.0 if (h >> 20) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


class LexicalReranker:
    """Share of the query's word stems present in the document — the no-key baseline reranker."""

    id = "lexical"

    async def rerank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        q = {w[:5] for w in _WORD.findall(_norm(query)) if len(w) > 2}
        scores = []
        for i, d in enumerate(docs):
            dw = {w[:5] for w in _WORD.findall(_norm(d))}
            scores.append((i, len(q & dw) / (len(q) or 1)))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_n]


def get_embedder(kind: str | None = None) -> Embedder:
    s = get_settings()
    kind = kind or s.rag_embedder
    if kind == "jina" or (kind == "auto" and s.jina_api_key):
        return JinaEmbedder()
    return LocalHashEmbedder()


def get_reranker(embedder: Embedder) -> Reranker:
    return JinaReranker() if isinstance(embedder, JinaEmbedder) else LexicalReranker()
