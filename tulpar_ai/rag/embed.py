"""Embeddings and reranking.

Primary: Jina (multilingual — questions are Russian, the WHO PDF is English; one vector space for both).
Fallback: a local hashing embedder + lexical reranker, so RAG still works with no keys at all (CI, a
mentor's laptop). The fallback is also a baseline arm in the RAG A/B.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Protocol

import httpx
from langsmith import traceable

from ..config import get_settings

JINA = "https://api.jina.ai/v1"


async def _post(c: httpx.AsyncClient, path: str, body: dict, attempts: int = 5) -> dict:
    """Jina free tier answers 429 under bursts: back off (honouring Retry-After) instead of failing the turn."""
    s = get_settings()
    delay = 1.0
    for i in range(attempts):
        r = await c.post(f"{JINA}{path}", headers={"Authorization": f"Bearer {s.jina_api_key}"}, json=body)
        if r.status_code not in (429, 500, 502, 503, 504) or i == attempts - 1:
            r.raise_for_status()
            return r.json()
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


class JinaEmbedder:
    id = "jina3"
    dim = 1024

    @traceable(run_type="embedding", name="jina_embed", process_inputs=trace_embed_inputs,
               process_outputs=trace_embed_outputs)
    async def embed(self, texts: list[str], task: str) -> list[list[float]]:
        s = get_settings()
        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=60) as c:
            for i in range(0, len(texts), 64):
                data = await _post(c, "/embeddings", {"model": s.embed_model, "task": task, "input": texts[i:i + 64]})
                out.extend(d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"]))
        return out


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
