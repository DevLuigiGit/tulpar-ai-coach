"""evals/embedding_map.py: no network, no Jina — a fake embedder, a tiny embedded Qdrant, hand-made vectors."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("evals_embedding_map", ROOT / "evals" / "embedding_map.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evals_embedding_map"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeEmbed:
    def __init__(self, dim=4):
        self.calls: list[tuple[list[str], str]] = []
        self.dim = dim

    async def __call__(self, texts, task):
        self.calls.append((list(texts), task))
        return [[float(len(t) + i) for i in range(self.dim)] for t in texts]


async def test_cache_sends_only_misses_once(tmp_path):
    em = _module()
    cache, embed = em.VectorCache(tmp_path / "emb_cache"), FakeEmbed()
    vecs, sent = await em.embed_with_cache(["а", "бб", "а"], cache=cache, embed=embed)
    assert vecs.shape == (3, 4) and sent == 2
    assert embed.calls == [(["а", "бб"], "retrieval.query")]
    np.testing.assert_array_equal(vecs[0], vecs[2])

    again, sent = await em.embed_with_cache(["бб", "а"], cache=cache, embed=embed)
    assert sent == 0 and len(embed.calls) == 1  # a rerun costs no API calls
    np.testing.assert_array_equal(again[0], vecs[1])

    await em.embed_with_cache(["а"], cache=cache, embed=embed, task="retrieval.passage")
    await em.embed_with_cache(["а"], cache=cache, embed=embed, model="other-model")
    assert [c[0] for c in embed.calls[1:]] == [["а"], ["а"]]  # task and model are part of the key


async def test_uncached_question_without_key_stops(tmp_path):
    em = _module()
    with pytest.raises(SystemExit, match="not cached"):
        await em.embed_with_cache(["вопрос"], cache=em.VectorCache(tmp_path), embed=None)


def test_copy_index_skips_the_lock(tmp_path):
    em = _module()
    src = tmp_path / "src"
    (src / "collection" / "c").mkdir(parents=True)
    (src / "meta.json").write_text("{}")
    (src / ".lock").write_text("owned by a running app")
    (src / "collection" / "c" / "storage.sqlite").write_text("x")
    dst = tmp_path / "copy" / "qdrant"
    em.copy_index(src, dst)
    em.copy_index(src, dst)  # a second copy replaces the first
    assert (dst / "meta.json").exists() and (dst / "collection" / "c" / "storage.sqlite").exists()
    assert not (dst / ".lock").exists() and (src / ".lock").exists()
    with pytest.raises(SystemExit):
        em.copy_index(tmp_path / "nothing", tmp_path / "x")


CHUNKS = [  # id, source, title, page, vector
    ("who:400:12:0", "who2020", "WHO", 12, [1.0, 0.1, 0.0, 0.0]),
    ("who:400:30:0", "who2020", "WHO", 30, [0.9, 0.5, 0.0, 0.1]),
    ("who:400:50:0", "who2020", "WHO", 50, [0.0, 1.0, 0.2, 0.0]),
    ("ex:1", "exercises", "Жим штанги лёжа", None, [0.0, 0.0, 1.0, 0.0]),
    ("ex:2", "exercises", "Планка", None, [0.0, 0.2, 0.9, 0.3]),
    ("nut:Цель:0", "nutrition", "Правила питания Tulpar — Цель", None, [0.0, 0.0, 0.1, 1.0]),
]


def _index(tmp_path):
    from qdrant_client import QdrantClient, models

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    client.create_collection("coach_test", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE))
    client.upsert("coach_test", points=[
        models.PointStruct(id=n, vector=v, payload={"id": cid, "source": src, "title": title, "page": page})
        for n, (cid, src, title, page, v) in enumerate(CHUNKS)])
    return client


def test_ranking_matches_qdrant_and_uses_the_eval_hit_rule(tmp_path):
    em = _module()
    client = _index(tmp_path)
    try:
        payloads, corpus = em.load_corpus(client, "coach_test")
        assert [p["id"] for p in payloads] == sorted(c[0] for c in CHUNKS)
        np.testing.assert_allclose(np.linalg.norm(corpus, axis=1), 1.0, rtol=1e-5)
        with pytest.raises(SystemExit, match="not in the index copy"):
            em.load_corpus(client, "coach_missing")

        q = np.array([0.95, 0.3, 0.0, 0.05])
        item = {"id": "q40", "question": "Сколько минут в неделю?", "tags": ["who", "cross-lingual"],
                "expected_sources": [{"source": "who2020", "page": 29}]}  # page 30 counts: ±1, as in evals/run.py
        row = em.analyze_question(item, q, corpus, payloads, hit_rank=em._hit_rank_fn(),
                                  rng=np.random.default_rng(0), n_random=10)
        assert [t["id"] for t in row["top"]] == em.qdrant_top_ids(client, "coach_test", q, 4)
        assert row["top"][0]["id"] == "who:400:30:0" and row["hit_rank"] == 1 and row["hit@4"]
        assert row["relevant"]["page"] == 30
        assert row["random"]["n"] == 5  # every chunk but the relevant one
        assert row["random_who"]["n"] == 2
        assert row["random"]["mean"] < row["relevant"]["score"]

        miss = em.analyze_question({**item, "expected_sources": [{"source": "exercises", "title": "Планка"}]}, q,
                                   corpus, payloads, hit_rank=em._hit_rank_fn(), rng=np.random.default_rng(0))
        assert miss["hit_rank"] == 4 and miss["hit@4"] and not miss["hit@3"]
    finally:
        client.close()


def test_summary_and_eval_cross_check(tmp_path):
    em = _module()
    rows = [
        {"id": "q40", "tags": ["who"], "hit_rank": 2, "hit@3": True, "hit@4": True, "qdrant_top4_match": True,
         "relevant": {"score": 0.7}, "random": {"mean": 0.2, "p95": 0.4},
         "top": [{"source": "who2020"}] * 3 + [{"source": "exercises"}]},
        {"id": "q49", "tags": ["who"], "hit_rank": 19, "hit@3": False, "hit@4": False, "qdrant_top4_match": True,
         "relevant": {"score": 0.5}, "random": {"mean": 0.3, "p95": 0.5},
         "top": [{"source": "who2020"}, {"source": "who2020"}, {"source": "nutrition"}]},
        {"id": "q01", "tags": ["technique"], "hit_rank": 1, "hit@3": True, "hit@4": True, "qdrant_top4_match": False,
         "relevant": {"score": 0.6}, "random": {"mean": 0.1, "p95": 0.3}, "top": [{"source": "exercises"}] * 3},
    ]
    s = em.summarize(rows)
    assert (s["who_hit@3"], s["who_hit@4"], s["hit@3_all"], s["qdrant_top4_match"]) == (1, 1, 2, 2)
    assert (s["who_top3_chunks_from_who_pdf"], s["who_top3_chunks_total"]) == (5, 6)
    assert s["who_cos_relevant_median"] == 0.6 and s["who_cos_random_p95_max"] == 0.5

    saved = tmp_path / "qa.json"  # evals/run.py keeps a rank only inside the top 4
    saved.write_text(json.dumps({"rows": [{"id": "q40", "rank": 2}, {"id": "q49", "rank": None},
                                          {"id": "q01", "rank": 3}]}))
    assert em.compare_with_eval(rows, saved) == {"file": "qa.json", "compared": 3, "same_rank": 2}
    assert em.compare_with_eval(rows, tmp_path / "missing.json") is None


def test_neighbour_preservation_is_perfect_for_an_exact_map():
    em = _module()
    angles = np.sort(np.random.default_rng(3).uniform(0, np.pi, 30))  # uneven: no distance ties
    corpus = np.stack([np.cos(angles), np.sin(angles), np.zeros(30)], axis=1)  # unit vectors on an arc
    queries = np.stack([np.cos([0.3, 2.0]), np.sin([0.3, 2.0]), [0.0, 0.0]], axis=1)
    exact = em.neighbour_preservation(corpus, queries, corpus[:, :2], queries[:, :2], k=5)
    assert exact == {"chunk_knn_recall@5": 1.0, "question_top3_within_10_nearest_on_map": 1.0}
    # a random permutation scrambles neighbourhoods (a reversed arc would still keep them)
    perm = np.random.default_rng(5).permutation(30)
    shuffled = em.neighbour_preservation(corpus, queries, corpus[perm, :2], queries[:, :2], k=5)
    assert shuffled["chunk_knn_recall@5"] < 0.5


def test_pca_projection_keeps_questions_in_the_same_plane():
    em = _module()
    rng = np.random.default_rng(1)
    corpus = em.normalize(rng.normal(size=(40, 8)))
    queries = rng.normal(size=(3, 8))
    c2, q2, meta = em.project(corpus, queries, method="pca")
    assert c2.shape == (40, 2) and q2.shape == (3, 2) and meta["method"] == "pca"
    _, q_again = em.pca_2d(corpus, em.normalize(queries))
    np.testing.assert_allclose(q2, q_again)


def test_labels_do_not_overlap_and_sit_next_to_their_own_star():
    em = _module()
    stars = np.array([[100.0, 100.0], [118.0, 104.0], [109.0, 122.0], [300.0, 300.0]])
    w, h, r = [20.0] * 4, 16.0, 12.0
    offsets = em.place_labels(stars, w, h, r, frame=(0, 0, 400, 400))
    boxes = [(x + dx - 10, y + dy - 8, x + dx + 10, y + dy + 8) for (x, y), (dx, dy) in zip(stars, offsets)]
    assert all(em._overlap(a, b) == 0 for i, a in enumerate(boxes) for b in boxes[i + 1:])
    for i, ((x, y), (dx, dy)) in enumerate(zip(stars, offsets)):
        d = np.hypot(stars[:, 0] - x - dx, stars[:, 1] - y - dy)
        assert d.argmin() == i
    assert em.crowded_group(stars, 3 * r) == [0, 1, 2]


def test_pictures_render(tmp_path):
    pytest.importorskip("matplotlib")
    em = _module()
    rng = np.random.default_rng(2)
    payloads = [{"id": f"c{i}", "source": src, "title": "Жим" if src == "exercises" else "WHO", "page": i}
                for i, src in enumerate(["who2020"] * 12 + ["exercises"] * 6 + ["nutrition"] * 2)]
    c2 = rng.normal(size=(20, 2))
    q2 = np.array([[0.0, 0.0], [0.05, 0.02], [0.02, 0.06], [2.0, 2.0]])
    top = [{"id": "c0", "source": "who2020", "title": "WHO", "page": 0, "score": 0.7},
           {"id": "c13", "source": "exercises", "title": "Жим", "page": None, "score": 0.5},
           {"id": "c19", "source": "nutrition", "title": "Правила питания Tulpar — Цель", "page": None, "score": 0.4}]
    rows = [{"id": f"q{i}", "short": f"вопрос {i}", "tags": ["who"], "top": top, "hit_rank": i + 1,
             "relevant": {"score": 0.6, "page": 12}, "random": {"mean": 0.2, "p5": 0.05, "p95": 0.4}}
            for i in range(4)]
    written = em.plot_map(c2, q2, payloads, rows, {"method": "umap"}, tmp_path)
    written += em.plot_bars(rows, 0.2, 200, 42, 4, tmp_path)
    assert [p.name for p in written] == ["embedding_map.png", "embedding_map.svg", "cosine_bars.png", "cosine_bars.svg"]
    assert all(p.stat().st_size > 10_000 for p in written)
