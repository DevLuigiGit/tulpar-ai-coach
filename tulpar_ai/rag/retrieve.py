"""Dense top-k from Qdrant → rerank → top-n. The rerank score is the «is this enough?» signal."""

from __future__ import annotations

from langsmith import traceable

from ..config import get_settings
from .embed import get_reranker
from .index import Index

_index: Index | None = None


def set_index(index: Index | None) -> None:
    global _index
    _index = index


def get_index() -> Index:
    global _index
    if _index is None:
        _index = Index()
    return _index


@traceable(run_type="retriever", name="retrieve")
async def retrieve(query: str, *, top_k: int | None = None, top_n: int | None = None, rerank: bool | None = None,
                   index: Index | None = None) -> list[dict]:
    s = get_settings()
    idx = index or get_index()
    top_k, top_n = top_k or s.rag_top_k, top_n or s.rag_top_n
    rerank = s.rag_rerank if rerank is None else rerank
    hits = await idx.search(query, top_k)
    if not hits:
        return []
    if rerank:
        order = await get_reranker(idx.embedder).rerank(query, [h["text"] for h in hits], top_n)
        out = [{**hits[i], "rerank_score": sc} for i, sc in order]
    else:
        out = [{**h, "rerank_score": h["score"]} for h in hits[:top_n]]
    return out
