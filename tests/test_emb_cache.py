"""evals/emb_cache.py: every text reaches the embedding API once; Matryoshka dims come from the cache."""

import httpx
import numpy as np
import pytest

from evals import emb_cache as ec


class CountingEmbedder:
    id = "fake"
    dim = 4

    def __init__(self, fail_with: list[int] | None = None):
        self.batches: list[list[str]] = []
        self.fail_with = list(fail_with or [])

    async def embed(self, texts, task):
        if self.fail_with:
            code = self.fail_with.pop(0)
            req = httpx.Request("POST", "https://example.invalid/v1/embeddings")
            raise httpx.HTTPStatusError(str(code), request=req, response=httpx.Response(code, request=req))
        self.batches.append(list(texts))
        # deterministic, not unit length: the cache normalises
        return [[float(len(t)), 1.0 if task == "retrieval.query" else 2.0, 3.0, 4.0] for t in texts]


async def test_second_pass_costs_nothing(tmp_path):
    inner = CountingEmbedder()
    emb = ec.CachedEmbedder(inner, ec.EmbeddingCache(tmp_path), model="fake-model")
    texts = [f"text {i}" for i in range(70)] + ["text 0"]  # a duplicate is embedded once
    first = await emb.embed(texts, "retrieval.passage")
    assert [len(b) for b in inner.batches] == [64, 6] and emb.api_texts == 70
    again = await emb.embed(texts, "retrieval.passage")
    assert len(inner.batches) == 2 and np.allclose(first, again)
    assert all(abs(np.linalg.norm(v) - 1) < 1e-6 for v in first)
    # the query task is another namespace: embedded separately
    await emb.embed(["text 0"], "retrieval.query")
    assert inner.batches[-1] == ["text 0"] and emb.api_texts == 71


async def test_matryoshka_is_truncated_cache(tmp_path):
    cache = ec.EmbeddingCache(tmp_path)
    full = ec.CachedEmbedder(CountingEmbedder(), cache, model="fake-model")
    [v] = await full.embed(["abc"], "retrieval.passage")
    inner = CountingEmbedder()
    small = ec.CachedEmbedder(inner, cache, model="fake-model", dim=2)
    [s] = await small.embed(["abc"], "retrieval.passage")
    assert inner.batches == [] and small.id == "faked2" and small.dim == 2 and full.id == "fake"
    want = np.array(v[:2]) / np.linalg.norm(v[:2])
    assert np.allclose(s, want) and abs(np.linalg.norm(s) - 1) < 1e-6


async def test_rate_limit_is_retried_other_errors_are_not(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "RETRY_WAITS", (0.0, 0.0, 0.0))
    inner = CountingEmbedder(fail_with=[429, 503])
    emb = ec.CachedEmbedder(inner, ec.EmbeddingCache(tmp_path), model="m")
    assert len(await emb.embed(["a"], "retrieval.query")) == 1 and emb.api_calls == 3
    bad = ec.CachedEmbedder(CountingEmbedder(fail_with=[401]), ec.EmbeddingCache(tmp_path / "x"), model="m")
    with pytest.raises(httpx.HTTPStatusError):
        await bad.embed(["a"], "retrieval.query")


def test_namespace_and_roundtrip(tmp_path):
    assert ec.namespace("jina-embeddings-v3", "retrieval.passage") != ec.namespace("jina-embeddings-v3", "retrieval.query")
    assert ec.namespace("m", "t", {"late_chunking": True}) != ec.namespace("m", "t")
    cache = ec.EmbeddingCache(tmp_path)
    ns = ec.namespace("m", "t")
    assert cache.get(ns, "текст") is None
    cache.put(ns, "текст", [3.0, 4.0])
    assert np.allclose(cache.get(ns, "текст"), [0.6, 0.8]) and cache.count() == {ns: 1}
    assert not list(tmp_path.rglob("*.tmp"))


def test_truncate_keeps_full_dim_and_zero_rows():
    m = np.array([[3.0, 4.0, 12.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    assert ec.truncate(m, None) is not None and ec.truncate(m, 3).shape == (2, 3)
    t = ec.truncate(m, 2)
    assert np.allclose(t[0], [0.6, 0.8]) and np.allclose(t[1], [0.0, 0.0])


def test_seed_from_embedded_qdrant(tmp_path):
    from qdrant_client import QdrantClient, models

    src = tmp_path / "qdrant"
    client = QdrantClient(path=str(src))
    client.create_collection("c", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
    client.upsert("c", points=[models.PointStruct(id=i, vector=v, payload={"text": t})
                               for i, (t, v) in enumerate([("один", [1.0, 0.0, 0.0]), ("два", [0.0, 3.0, 4.0])])])
    client.close()
    cache = ec.EmbeddingCache(tmp_path / "cache")
    res = ec.seed_from_qdrant(src, "c", cache, "m")
    assert res["added"] == 2 and res["already_cached"] == 0
    ns = ec.namespace("m", "retrieval.passage")
    assert np.allclose(cache.get(ns, "два"), [0.0, 0.6, 0.8])
    assert ec.seed_from_qdrant(src, "c", cache, "m")["already_cached"] == 2
