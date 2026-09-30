"""Disk cache of embeddings for retrieval experiments: each text is sent to the API once per (model, task, options).

The Jina key is shared and its quota is small, while retrieval variants re-embed the same 1 476 chunks and the
same questions again and again. Every vector is stored as one float32 .npy file:

    <cache dir>/<model>__<task>__<options hash>/<sha256[:2]>/<sha256 of the text>.npy

Writes are atomic (temp file + rename), so several processes, and several git worktrees pointing EMB_CACHE_DIR at
one folder, can share the cache. Vectors are stored L2-normalised: cosine and dot product rank the same way, and a
Matryoshka variant is the first `dim` numbers of the cached 1024-d vector, renormalised — never a new API call.

    python evals/emb_cache.py stats
    python evals/emb_cache.py seed --from ../../data/final_nutrition/qdrant --collection coach_jina3_400_p2_b064ec4f
    python evals/emb_cache.py check --n 3      # re-embed 3 cached passages and compare (3 texts of quota)

`seed` copies passage vectors out of an existing embedded-Qdrant index (the payload keeps the exact text that was
embedded), so the corpus does not have to be embedded again at all.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BATCH = 64  # texts per API call; the cache is written after every batch, so an interrupted run keeps its progress
RETRY_WAITS = (15.0, 45.0, 90.0)  # after JinaEmbedder's own short 429 backoff gives up


def default_dir() -> Path:
    """EMB_CACHE_DIR if set (share one cache between worktrees), else AI_DATA_DIR/emb_cache."""
    env = os.environ.get("EMB_CACHE_DIR")
    if env:
        return Path(env)
    from tulpar_ai.config import get_settings

    return Path(get_settings().ai_data_dir) / "emb_cache"


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def namespace(model: str, task: str, options: dict | None = None) -> str:
    opts = json.dumps(options or {}, sort_keys=True, ensure_ascii=False)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{model}__{task}")
    return f"{slug}__{hashlib.sha1(opts.encode()).hexdigest()[:8]}"


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def truncate(m: np.ndarray, dim: int | None) -> np.ndarray:
    """Matryoshka: keep the first `dim` coordinates of each row and renormalise (rows stay unit length)."""
    m = np.asarray(m, dtype=np.float32)
    if dim is None or dim >= m.shape[-1]:
        return m
    t = m[..., :dim]
    n = np.linalg.norm(t, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return t / n


class EmbeddingCache:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else default_dir()
        self.hits = 0
        self.misses = 0

    def path(self, ns: str, text: str) -> Path:
        k = text_key(text)
        return self.root / ns / k[:2] / f"{k}.npy"

    def get(self, ns: str, text: str) -> np.ndarray | None:
        p = self.path(ns, text)
        if not p.exists():
            self.misses += 1
            return None
        self.hits += 1
        return np.load(p)

    def put(self, ns: str, text: str, vec) -> None:
        p = self.path(ns, text)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, unit(vec))
        os.replace(tmp, p)

    def has(self, ns: str, text: str) -> bool:
        return self.path(ns, text).exists()

    def count(self, ns: str | None = None) -> dict[str, int]:
        if not self.root.exists():
            return {}
        spaces = [self.root / ns] if ns else [d for d in sorted(self.root.iterdir()) if d.is_dir()]
        return {d.name: sum(1 for _ in d.glob("*/*.npy")) for d in spaces if d.exists()}


def _is_rate_limit(e: Exception) -> bool:
    resp = getattr(e, "response", None)
    return getattr(resp, "status_code", None) in (429, 500, 502, 503, 504)


class CachedEmbedder:
    """The Embedder protocol (id, dim, embed(texts, task)) in front of a real embedder.

    Only texts missing from the cache reach `inner`, in batches of 64, written to disk after each batch.
    `dim` < the model's dimension gives a Matryoshka variant from the same cached vectors (its own `id`, so a Qdrant
    collection of 256-d vectors never collides with the 1024-d one). `api_texts` counts what really went to the API.
    """

    def __init__(self, inner, cache: EmbeddingCache | None = None, *, model: str | None = None,
                 options: dict | None = None, dim: int | None = None):
        self.inner = inner
        self.cache = cache or EmbeddingCache()
        self.model = model or _model_name(inner)
        self.options = options or {}
        full = int(inner.dim)
        self.dim = full if not dim or dim >= full else int(dim)
        self.id = inner.id if self.dim == full else f"{inner.id}d{self.dim}"
        self.api_texts = 0
        self.api_calls = 0

    async def embed(self, texts: list[str], task: str) -> list[list[float]]:
        return truncate(await self.matrix(texts, task), self.dim).tolist()

    async def matrix(self, texts: list[str], task: str) -> np.ndarray:
        """Full-dimension unit vectors as one float32 matrix, in the order of `texts`."""
        ns = namespace(self.model, task, self.options)
        found: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for t in dict.fromkeys(texts):
            v = self.cache.get(ns, t)
            if v is None:
                missing.append(t)
            else:
                found[t] = v
        for i in range(0, len(missing), BATCH):
            batch = missing[i:i + BATCH]
            vecs = await self._call(batch, task)
            for t, v in zip(batch, vecs):
                self.cache.put(ns, t, v)
                found[t] = unit(v)
        if not texts:
            return np.zeros((0, int(self.inner.dim)), dtype=np.float32)
        return np.stack([found[t] for t in texts]).astype(np.float32)

    async def _call(self, batch: list[str], task: str) -> list[list[float]]:
        for attempt in range(len(RETRY_WAITS) + 1):
            try:
                self.api_calls += 1
                out = await self.inner.embed(batch, task=task)
                self.api_texts += len(batch)
                return out
            except Exception as e:  # noqa: BLE001 — only rate limits and 5xx are retried, the rest propagate
                if attempt == len(RETRY_WAITS) or not _is_rate_limit(e):
                    raise
                await asyncio.sleep(RETRY_WAITS[attempt])
        raise RuntimeError("unreachable")


def _model_name(inner) -> str:
    if getattr(inner, "id", "") == "jina3":
        from tulpar_ai.config import get_settings

        return get_settings().embed_model
    return str(getattr(inner, "id", type(inner).__name__))


def seed_from_qdrant(storage: Path, collection: str, cache: EmbeddingCache, model: str,
                     task: str = "retrieval.passage") -> dict:
    """Copy (payload text → vector) pairs of an embedded-Qdrant collection into the cache.

    The storage folder is copied to a temp dir first: embedded Qdrant locks its folder, and the source may belong
    to a running service. Returns counts; texts already cached are left alone.
    """
    from qdrant_client import QdrantClient

    ns = namespace(model, task)
    added = skipped = 0
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "qdrant"
        shutil.copytree(storage, copy, ignore=shutil.ignore_patterns(".lock"))
        client = QdrantClient(path=str(copy))
        try:
            offset = None
            while True:
                points, offset = client.scroll(collection, limit=256, offset=offset, with_payload=["text"],
                                               with_vectors=True)
                for p in points:
                    text = (p.payload or {}).get("text")
                    if not text:
                        continue
                    if cache.has(ns, text):
                        skipped += 1
                        continue
                    cache.put(ns, text, p.vector)
                    added += 1
                if offset is None:
                    break
        finally:
            client.close()
    return {"namespace": ns, "added": added, "already_cached": skipped}


async def check_against_api(cache: EmbeddingCache, texts: list[str], model: str, task: str = "retrieval.passage") -> list[float]:
    """Cosine between cached vectors and fresh API vectors for a few texts (costs len(texts) of quota)."""
    from tulpar_ai.rag.embed import JinaEmbedder

    ns = namespace(model, task)
    fresh = await JinaEmbedder().embed(texts, task=task)
    return [round(float(np.dot(cache.get(ns, t), unit(v))), 6) for t, v in zip(texts, fresh)]


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    sd = sub.add_parser("seed")
    sd.add_argument("--from", dest="src", required=True, help="embedded Qdrant folder (AI_DATA_DIR/qdrant)")
    sd.add_argument("--collection", required=True)
    sd.add_argument("--task", default="retrieval.passage")
    ck = sub.add_parser("check")
    ck.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    from tulpar_ai.config import get_settings

    os.environ["LANGSMITH_TRACING"] = "false"
    model = get_settings().embed_model
    cache = EmbeddingCache()
    if args.cmd == "stats":
        print(json.dumps({"dir": str(cache.root), "vectors": cache.count()}, ensure_ascii=False, indent=1))
    elif args.cmd == "seed":
        print(json.dumps(seed_from_qdrant(Path(args.src), args.collection, cache, model, args.task), ensure_ascii=False))
    else:
        from tulpar_ai.rag.index import load_chunks

        ns = namespace(model, "retrieval.passage")
        texts = [c.text for c in load_chunks(pdf_chunk=400) if cache.has(ns, c.text)]
        pick = [texts[i] for i in np.random.default_rng(0).choice(len(texts), size=min(args.n, len(texts)), replace=False)]
        print(json.dumps({"cosine_cached_vs_api": asyncio.run(check_against_api(cache, pick, model))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
