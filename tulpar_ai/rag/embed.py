"""Embeddings and reranking.

Primary: Jina (multilingual — questions are Russian, the WHO PDF is English; one vector space for both).
Fallback: a local hashing embedder + lexical reranker, so RAG still works with no keys at all (CI, a
mentor's laptop). The fallback is also a baseline arm in the RAG A/B.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx
from langsmith import traceable

from ..config import get_settings

JINA = "https://api.jina.ai/v1"


class Embedder(Protocol):
    id: str
    dim: int

    async def embed(self, texts: list[str], task: str) -> list[list[float]]: ...


class Reranker(Protocol):
    id: str

    async def rerank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]: ...


class JinaEmbedder:
    id = "jina3"
    dim = 1024

    @traceable(run_type="embedding", name="jina_embed")
    async def embed(self, texts: list[str], task: str) -> list[list[float]]:
        s = get_settings()
        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=60) as c:
            for i in range(0, len(texts), 64):
                r = await c.post(f"{JINA}/embeddings", headers={"Authorization": f"Bearer {s.jina_api_key}"},
                                 json={"model": s.embed_model, "task": task, "input": texts[i:i + 64]})
                r.raise_for_status()
                out.extend(d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"]))
        return out


class JinaReranker:
    id = "jina-rerank"

    @traceable(run_type="tool", name="jina_rerank")
    async def rerank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        s = get_settings()
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{JINA}/rerank", headers={"Authorization": f"Bearer {s.jina_api_key}"},
                             json={"model": s.rerank_model, "query": query, "documents": docs, "top_n": top_n})
        r.raise_for_status()
        return [(x["index"], float(x["relevance_score"])) for x in r.json()["results"]]


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
