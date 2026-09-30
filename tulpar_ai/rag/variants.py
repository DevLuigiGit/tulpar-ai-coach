"""Retrieval variants behind settings, and one entry point to build a retriever for any of them.

    RAG_HYBRID=on             dense + BM25 sparse vectors in one Qdrant collection, Query API: prefetch dense top-k and
                              sparse top-k → RRF fusion (rag/sparse.py)
    RAG_LATE_CHUNKING=on      Jina late chunking: WHO chunks of a page window / nutrition chunks of the file are
                              embedded as one document, exercise cards stay single (rag/embed.py, rag/index.py)
    RAG_EMBED_DIM=512|256     Matryoshka: Jina vectors (passages and queries) cut to the first N components and
                              L2-renormalised; nothing is re-embedded for a smaller dimension
    RAG_CONTEXT_HEADERS=on    «ВОЗ 2020 · раздел · стр. N» in front of the embedded text only (rag/context.py)
    RAG_MULTI_QUERY=en        a Russian question is also searched in English; the two lists are fused by RRF

Defaults reproduce the setup measured before these variants: same collection name, same vectors, same search.
Everything that changes what is stored is part of the collection name (`suffix`), so one Qdrant never mixes schemas;
the query-side variant (multi-query) changes the answer-cache fingerprint (`cache_tag`) instead.

`build_retriever(settings)` is what evals use to compare variants:

    r = build_retriever(settings)          # or build_retriever(overrides={"rag_hybrid": True})
    await r.build()                        # index (reuses the collection and the on-disk embedding cache)
    hits = await r.retrieve(query, k=4)    # [{"id": chunk id, "payload": {...}, "score": ..., ...}]
    r.close()
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings

VALID_DIMS = (1024, 512, 256)


@dataclass(frozen=True)
class RetrievalVariant:
    hybrid: bool = False
    bm25: str = "auto"
    late_chunking: bool = False
    late_max_chars: int = 12000
    late_window_pages: int = 3
    embed_dim: int = 1024
    context_headers: bool = False
    multi_query: str = "off"
    rrf_k: int = 60

    @classmethod
    def from_settings(cls, s: Settings | None = None) -> "RetrievalVariant":
        s = s or get_settings()
        dim = int(s.rag_embed_dim)
        if dim not in VALID_DIMS:
            raise ValueError(f"RAG_EMBED_DIM must be one of {VALID_DIMS}, got {dim}")
        mq = str(s.rag_multi_query).lower()
        if mq not in ("off", "en"):
            raise ValueError(f"RAG_MULTI_QUERY must be off or en, got {s.rag_multi_query!r}")
        return cls(hybrid=bool(s.rag_hybrid), bm25=str(s.rag_bm25).lower(), late_chunking=bool(s.rag_late_chunking),
                   late_max_chars=int(s.rag_late_max_chars), late_window_pages=int(s.rag_late_window_pages),
                   embed_dim=dim, context_headers=bool(s.rag_context_headers), multi_query=mq, rrf_k=int(s.rag_rrf_k))

    @property
    def is_default(self) -> bool:
        return self.cache_tag() == ""

    def cache_tag(self) -> str:
        """Short description of every non-default knob ("" for the default setup); goes into fingerprints and labels."""
        parts = []
        if self.hybrid:
            parts.append(f"hybrid-{self.bm25}")
        if self.late_chunking:
            parts.append(f"late-w{self.late_window_pages}-{self.late_max_chars}")
        if self.embed_dim != 1024:
            parts.append(f"d{self.embed_dim}")
        if self.context_headers:
            parts.append("ctx")
        if self.multi_query != "off":
            parts.append(f"mq-{self.multi_query}")
        if self.rrf_k != 60 and (self.hybrid or self.multi_query != "off"):
            parts.append(f"rrf{self.rrf_k}")
        return "+".join(parts)


DEFAULT = RetrievalVariant()


def variant_of(index) -> RetrievalVariant:
    """The variant an index was built with; the default for index-like stubs that do not say."""
    return getattr(index, "variant", None) or DEFAULT


def truncate(vector: list[float], dim: int) -> list[float]:
    """Matryoshka: the first `dim` components, L2-renormalised (a no-op when the vector is not longer than dim)."""
    if dim >= len(vector):
        return list(vector)
    head = vector[:dim]
    norm = math.sqrt(sum(x * x for x in head)) or 1.0
    return [x / norm for x in head]


def rrf_fuse(lists: list[list[dict]], k: int = 60, limit: int | None = None, key: str = "id") -> list[dict]:
    """Reciprocal rank fusion of ranked hit lists: sum of 1/(k + rank), rank from 1.

    A hit keeps the payload of its first appearance; `score` becomes the best dense cosine it had in any list (the
    «is this enough?» signal of the chat graph stays a cosine, not a tiny RRF number), `fusion_score` is the RRF sum.
    """
    fused: dict[Any, dict] = {}
    for hits in lists:
        for rank, h in enumerate(hits, 1):
            ident = h.get(key) or (h.get("source"), h.get("title"), h.get("page"), h.get("text"))
            if ident not in fused:
                fused[ident] = {**h, "fusion_score": 0.0, "score": h.get("score", 0.0)}
            f = fused[ident]
            f["fusion_score"] += 1.0 / (k + rank)
            f["score"] = max(f["score"], h.get("score", 0.0))
    out = sorted(fused.values(), key=lambda h: -h["fusion_score"])
    return out[:limit] if limit else out


# ── one retriever for evals ──────────────────────────────────────────────────
class Retriever:
    """An Index for one variant plus the production retrieve() over it (translation, fusion, rerank as configured)."""

    def __init__(self, index, variant: RetrievalVariant, top_k: int, rerank: bool | None):
        self.index = index
        self.variant = variant
        self.top_k = top_k
        self.rerank = rerank

    @property
    def collection(self) -> str:
        return self.index.collection

    async def build(self, force: bool = False) -> int:
        return await self.index.build(force=force)

    async def retrieve(self, query: str, k: int = 4) -> list[dict]:
        """Top-k chunks: [{"id", "payload", "score", "rerank_score", "fusion_score"?, "queries"?}], best first."""
        from .retrieve import retrieve

        hits = await retrieve(query, top_k=max(self.top_k, k), top_n=k, rerank=self.rerank, index=self.index)
        out = []
        for h in hits:
            payload = {key: v for key, v in h.items() if key not in _SCORE_KEYS}
            out.append({"id": h.get("id"), "payload": payload, **{key: h[key] for key in _SCORE_KEYS if key in h}})
        return out

    async def retrieve_ids(self, query: str, k: int = 4) -> list[str]:
        return [h["id"] for h in await self.retrieve(query, k)]

    def close(self) -> None:
        self.index.close()


_SCORE_KEYS = ("score", "rerank_score", "fusion_score", "queries")


def build_retriever(settings: Settings | None = None, *, overrides: dict | None = None, embedder=None,
                    path: Path | None = None, pdf_chunk: int = 400, rerank: bool | None = None) -> Retriever:
    """A Retriever for the variant in `settings` (default: current settings), optionally with field overrides.

    `overrides` takes settings names (rag_hybrid, rag_embed_dim, …) and is applied to a copy: the process-wide
    settings are left alone, so an eval can build several variants side by side.
    """
    from .embed import get_embedder
    from .index import Index

    s = settings or get_settings()
    if overrides:
        s = s.model_copy(update=overrides)
    variant = RetrievalVariant.from_settings(s)
    index = Index(embedder=embedder or get_embedder(s.rag_embedder), path=path or Path(s.ai_data_dir) / "qdrant",
                  pdf_chunk=pdf_chunk, variant=variant)
    return Retriever(index, variant, top_k=s.rag_top_k, rerank=rerank)


def describe(variant: RetrievalVariant) -> dict:
    return {**asdict(variant), "tag": variant.cache_tag() or "default"}
