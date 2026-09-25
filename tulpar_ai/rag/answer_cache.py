"""Semantic answer cache for the question branch.

Answers are not personalized: the answer node sees only the masked question and the retrieved sources, so the
answer to one client's question is safe to give another client who asks the same thing. Only intent=question
reaches the cache — red flags, meals, program requests and injections are routed away before it — and only a
final answer with citations is stored.

A hit needs two things: the cosine clears the threshold, and the lexical guard (rag/cache_guard.py) finds no slot on
which the two questions disagree. Embeddings put "Я женщина…" and "Я мужчина…" at 0.99, so the threshold alone cannot
keep a man from getting the women's answer.

The collection name carries a fingerprint of everything that shapes an answer: the answer prompt (version and
text) and its sampling settings, the text model chain, the RAG index collection, retrieval settings and the corpus
files. Changing any of them starts from an empty cache instead of serving answers produced by the old setup.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from functools import lru_cache
from pathlib import Path

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from qdrant_client import models

from ..config import get_settings
from ..prompts import active_version, prompt
from .cache_guard import conflicts
from .embed import JinaEmbedder
from .index import CORPUS, Index
from .qdrant import get_client
from .retrieve import get_index

NS = uuid.UUID("5b8f2d0e-6c3a-4e1f-9a7d-1c2b3e4f5a60")
CANDIDATES = 5  # the nearest entry may be a guarded near-miss while the next one is the real repeat


@lru_cache
def corpus_digest(root: Path = CORPUS) -> str:
    """Content hash of the corpus: a re-indexed corpus keeps the collection name, so its name alone is not enough."""
    h = hashlib.sha1()
    for p in sorted(root.rglob("*")) if root.is_dir() else []:
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def fingerprint(index: Index) -> str:
    s, version = get_settings(), active_version("answer")
    parts = [
        f"answer={version}:{hashlib.sha1(prompt('answer', version).encode()).hexdigest()[:10]}",
        f"sampling={s.answer_temperature}/{s.answer_top_p}/{s.answer_max_tokens}",
        f"text={s.text_models}",
        f"index={index.collection}",
        f"retrieval={s.rag_top_k}/{s.rag_top_n}/{s.rag_rerank}/{s.rag_min_score}",
        f"corpus={corpus_digest()}",
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def pick(question: str, candidates: list[dict], min_score: float) -> tuple[dict | None, list[dict]]:
    """First candidate (nearest first) above the threshold whose question agrees with this one on every guarded slot.

    Also returns the candidates the guard turned down, for the trace. Shared with evals/cache_eval.py so the sweep
    measures exactly what production serves.
    """
    rejected = []
    for c in candidates:
        if c["score"] < min_score:
            break
        slots = conflicts(question, c["question"])
        if not slots:
            return c, rejected
        rejected.append({"score": round(c["score"], 4), "slots": slots})
    return None, rejected


def _key(question: str) -> str:
    return str(uuid.uuid5(NS, " ".join(question.lower().split())))


def _only_question(inputs: dict) -> dict:
    return {"question": inputs.get("question")}


class AnswerCache:
    def __init__(self, index: Index):
        self.index = index
        self.client = get_client(index.path)
        self._created: set[str] = set()

    @property
    def collection(self) -> str:
        return f"answer_cache_{self.index.embedder.id}_{fingerprint(self.index)}"

    @property
    def min_score(self) -> float:
        s = get_settings()
        return s.answer_cache_min_score if isinstance(self.index.embedder, JinaEmbedder) else s.answer_cache_min_score_local

    @traceable(run_type="tool", name="answer_cache", process_inputs=_only_question)
    async def lookup(self, question: str) -> dict | None:
        """Best fresh entry for a masked question that clears the threshold and the guard, else None."""
        vec = await self.index.embed_query(question)
        name = self.collection
        found = self._nearest(name, vec, time.time(), CANDIDATES)
        best, rejected = pick(question, found, self.min_score)
        rt = get_current_run_tree()
        if rt is not None:
            rt.metadata.update({"hit": best is not None, "score": round(found[0]["score"], 4) if found else None,
                                "min_score": self.min_score, "collection": name, "guard_rejected": rejected})
        return best

    def _nearest(self, name: str, vector: list[float], now: float, limit: int = 1) -> list[dict]:
        if not self.client.collection_exists(name):
            return []
        fresh = models.Filter(must=[models.FieldCondition(
            key="created_at", range=models.Range(gte=now - get_settings().answer_cache_ttl_h * 3600))])
        res = self.client.query_points(name, query=vector, query_filter=fresh, limit=limit, with_payload=True)
        return [{**p.payload, "score": float(p.score)} for p in res.points]

    async def put(self, question: str, reply: str, citations: list[dict], created_at: float | None = None) -> bool:
        """Store a final answer. Anything without citations is not a grounded answer and is never cached."""
        if not question or not reply or not citations:
            return False
        vec = await self.index.embed_query(question)
        name, now = self.collection, created_at or time.time()
        self._ensure(name, len(vec))
        self.client.upsert(name, points=[models.PointStruct(id=_key(question), vector=vec, payload={
            "question": question, "reply": reply, "citations": citations, "created_at": now})])
        expired = models.Filter(must=[models.FieldCondition(
            key="created_at", range=models.Range(lt=time.time() - get_settings().answer_cache_ttl_h * 3600))])
        self.client.delete(name, points_selector=models.FilterSelector(filter=expired))
        return True

    def count(self) -> int:
        name = self.collection
        return self.client.count(name).count if self.client.collection_exists(name) else 0

    def _ensure(self, name: str, dim: int) -> None:
        if name in self._created:
            return
        if not self.client.collection_exists(name):
            self.client.create_collection(name, vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE))
            if get_settings().qdrant_url:  # payload indexes only matter on a server; embedded mode warns
                self.client.create_payload_index(name, "created_at", models.PayloadSchemaType.FLOAT)
        self._created.add(name)


_cache: AnswerCache | None = None


def get_answer_cache() -> AnswerCache | None:
    """None when switched off. Follows the current RAG index, which the API lifespan, tests and evals swap."""
    global _cache
    if not get_settings().answer_cache:
        return None
    idx = get_index()
    if _cache is None or _cache.index is not idx:
        _cache = AnswerCache(idx)
    return _cache


def set_answer_cache(cache: AnswerCache | None) -> None:
    global _cache
    _cache = cache
