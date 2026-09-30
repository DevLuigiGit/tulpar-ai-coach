"""Dense top-k from Qdrant → rerank → top-n. The rerank score is the «is this enough?» signal.

The index decides how one query is searched (dense, or dense + BM25 fused by RRF — rag/index.py). With
RAG_MULTI_QUERY=en a Russian question is searched a second time in English and the two ranked lists are fused by
RRF (rag/variants.py) before the optional rerank; `score` stays the best dense cosine of a hit.
"""

from __future__ import annotations

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from ..config import get_settings
from .embed import JinaEmbedder, get_reranker
from .index import Index
from .translate import to_english
from .variants import rrf_fuse, variant_of

_index: Index | None = None


def set_index(index: Index | None) -> None:
    global _index
    _index = index


def get_index() -> Index:
    global _index
    if _index is None:
        _index = Index()
    return _index


def effective_rerank(idx: Index | None = None) -> bool:
    """A/B: the reranker helps the weak local embedder and adds nothing but latency on top of Jina."""
    mode = str(get_settings().rag_rerank).lower()
    if mode in ("on", "true", "1"):
        return True
    if mode in ("off", "false", "0"):
        return False
    return not isinstance((idx or get_index()).embedder, JinaEmbedder)


async def candidates(query: str, top_k: int, idx: Index) -> list[dict]:
    """Top-k candidates for one question: one search, or the RRF fusion of the question and its English version."""
    variant = variant_of(idx)
    if variant.multi_query != "en":
        return await idx.search(query, top_k)
    english = await to_english(query)
    rt = get_current_run_tree()
    if rt is not None:
        rt.metadata["query_en"] = english or None
    if not english or english.strip().lower() == query.strip().lower():
        return await idx.search(query, top_k)
    lists = [await idx.search(query, top_k), await idx.search(english, top_k)]
    fused = rrf_fuse(lists, k=variant.rrf_k, limit=top_k)
    for h in fused:
        h["queries"] = [n for n, hits in (("ru", lists[0]), ("en", lists[1])) if any(x.get("id") == h.get("id") for x in hits)]
    return fused


@traceable(run_type="retriever", name="retrieve")
async def retrieve(query: str, *, top_k: int | None = None, top_n: int | None = None, rerank: bool | None = None,
                   index: Index | None = None) -> list[dict]:
    s = get_settings()
    idx = index or get_index()
    top_k, top_n = top_k or s.rag_top_k, top_n or s.rag_top_n
    rerank = effective_rerank(idx) if rerank is None else rerank
    rt = get_current_run_tree()
    if rt is not None and not variant_of(idx).is_default:
        rt.metadata["rag_variant"] = variant_of(idx).cache_tag()
    hits = await candidates(query, top_k, idx)
    if not hits:
        return []
    if rerank:
        try:
            order = await get_reranker(idx.embedder).rerank(query, [h["text"] for h in hits], top_n)
            out = [{**hits[i], "rerank_score": sc} for i, sc in order]
        except Exception:  # reranker down or rate-limited → dense order, still an answer
            rt = get_current_run_tree()
            if rt is not None:
                rt.metadata["rerank_fallback"] = True
            out = [{**h, "rerank_score": h["score"]} for h in hits[:top_n]]
    else:
        out = [{**h, "rerank_score": h["score"]} for h in hits[:top_n]]
    return out
