"""Jina vectors on disk, so a rebuilt index or another retrieval variant does not spend the API quota again.

One `.npy` (float32, the full 1024-d vector) per key; the key hashes everything the vector depends on: the model,
the task (retrieval.passage / retrieval.query), the exact text and any option that changes the output — for late
chunking the whole group the chunk was embedded with and its position in it. Matryoshka dimensions are never cached
separately: they are cut from the full vector (rag/variants.py).

RAG_EMBED_CACHE: off (default) | passages (index builds) | all (also query vectors: evals, where the same golden
questions are embedded once per arm). Query vectors stay out of the disk cache in production by default: they are
derived from client questions.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np

from ..config import get_settings

PASSAGE_TASKS = {"retrieval.passage"}


def cache_key(model: str, task: str, text: str, options: str = "") -> str:
    return hashlib.sha256(json.dumps([model, task, options, text], ensure_ascii=False).encode()).hexdigest()


class EmbeddingCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.npy"

    def get(self, key: str) -> list[float] | None:
        p = self._path(key)
        try:
            vec = np.load(p, allow_pickle=False)
        except (OSError, ValueError):  # missing, or a torn write from a killed process: embed again
            self.misses += 1
            return None
        self.hits += 1
        return vec.astype(np.float64).tolist()

    def put(self, key: str, vector: list[float]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.stem}.{uuid.uuid4().hex}.tmp.npy")
        np.save(tmp, np.asarray(vector, dtype=np.float32), allow_pickle=False)
        os.replace(tmp, p)  # atomic: parallel eval arms never read half a file

    def size(self) -> int:
        return sum(1 for _ in self.root.rglob("*.npy")) if self.root.is_dir() else 0


_caches: dict[str, EmbeddingCache] = {}


def get_embedding_cache(task: str) -> EmbeddingCache | None:
    """The cache for this task under the current settings, or None when it must not be used."""
    s = get_settings()
    mode = str(s.rag_embed_cache).lower()
    if mode in ("off", "false", "0", ""):
        return None
    if mode != "all" and task not in PASSAGE_TASKS:
        return None
    root = str(Path(s.ai_data_dir) / "emb_cache")
    if root not in _caches:
        _caches[root] = EmbeddingCache(Path(root))
    return _caches[root]
