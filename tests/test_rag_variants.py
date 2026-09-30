"""Retrieval variants (rag/variants.py): each knob off by default, each one tested without network or keys.

Jina is faked at the HTTP-helper level (`embed._post`), so the real JinaEmbedder code — batching, the disk cache,
late chunking requests — runs; vectors are deterministic pseudo-random unit vectors per input text.
"""

from __future__ import annotations

import hashlib
import json
import math

import httpx
import numpy as np
import pytest

CARDS = [
    {"id": "c1", "title": "Приседания со штангой", "muscle_group": "ноги", "equipment": "barbell",
     "text": "Упражнение: Приседания со штангой. Группа мышц: ноги. Инвентарь: barbell. Техника: опускайтесь до параллели бёдер с полом."},
    {"id": "c2", "title": "Жим гантелей лёжа", "muscle_group": "грудь", "equipment": "dumbbell",
     "text": "Упражнение: Жим гантелей лёжа. Группа мышц: грудь. Инвентарь: dumbbell. Техника: локти под углом 45 градусов."},
    {"id": "c3", "title": "Планка", "muscle_group": "пресс", "equipment": None,
     "text": "Упражнение: Планка. Группа мышц: пресс. Техника: держите корпус прямым 30–60 секунд, не прогибайте поясницу."},
]
NUTRITION = "# Правила питания Tulpar\n\n## Вода\n\nНорма воды — 32 мл на каждый килограмм веса.\n\n## Белок\n\nБелок — 2 г на килограмм веса."


def fake_vector(text: str, dim: int = 1024) -> list[float]:
    rng = np.random.default_rng(int(hashlib.sha1(text.encode()).hexdigest()[:8], 16))
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).tolist()


class FakeJina:
    """Stands in for `embed._post`: records every request body, answers /embeddings with fake vectors."""

    def __init__(self, too_long: int | None = None):
        self.bodies: list[dict] = []
        self.too_long = too_long  # late-chunking groups with more inputs than this are rejected like an over-long doc

    async def __call__(self, c, path: str, body: dict, attempts: int = 5) -> dict:
        self.bodies.append(body)
        if body.get("late_chunking") and self.too_long and len(body["input"]) > self.too_long:
            req = httpx.Request("POST", "https://api.jina.ai/v1/embeddings")
            raise httpx.HTTPStatusError("too long", request=req, response=httpx.Response(400, request=req))
        tag = "late|" + "¦".join(body["input"]) + "|" if body.get("late_chunking") else ""
        return {"data": [{"index": i, "embedding": fake_vector(tag + t)} for i, t in enumerate(body["input"])],
                "usage": {"total_tokens": sum(len(t) for t in body["input"])}}

    def inputs(self, task: str | None = None) -> list[str]:
        return [t for b in self.bodies if task in (None, b.get("task")) for t in b["input"]]


@pytest.fixture
def small_corpus(tmp_path, monkeypatch):
    from tulpar_ai.rag import index

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "exercises.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in CARDS), encoding="utf-8")
    (corpus / "nutrition.md").write_text(NUTRITION, encoding="utf-8")
    monkeypatch.setattr(index, "CORPUS", corpus)
    return corpus


@pytest.fixture
def jina(env, monkeypatch):
    from tulpar_ai.rag import embed

    fake = FakeJina()
    monkeypatch.setattr(embed, "_post", fake)
    return fake


def _set(monkeypatch, **kw):
    from tulpar_ai.config import get_settings

    s = get_settings()
    for k, v in kw.items():
        monkeypatch.setattr(s, k, v)
    return s


# ── defaults and names ───────────────────────────────────────────────────────
def test_defaults_keep_the_collection_name_and_the_answer_cache_fingerprint(env):
    from tulpar_ai.config import get_settings
    from tulpar_ai.prompts import active_version, prompt
    from tulpar_ai.rag.answer_cache import corpus_digest, fingerprint
    from tulpar_ai.rag.index import Index, corpus_fingerprint
    from parsing.chunker import PARSING_VERSION

    idx = Index()
    try:
        assert idx.variant.is_default and idx.variant.cache_tag() == ""
        assert idx.collection == f"coach_local_400_p{PARSING_VERSION}_{corpus_fingerprint()}"
        assert idx.sparse is None and not idx.late and not idx.headers and idx.dim == idx.embedder.dim
        s, v = get_settings(), active_version("answer")
        before = "|".join([  # the fingerprint formula as it was before the variants existed
            f"answer={v}:{hashlib.sha1(prompt('answer', v).encode()).hexdigest()[:10]}",
            f"sampling={s.answer_temperature}/{s.answer_top_p}/{s.answer_max_tokens}", f"text={s.text_models}",
            f"index={idx.collection}", f"retrieval={s.rag_top_k}/{s.rag_top_n}/{s.rag_rerank}/{s.rag_min_score}",
            f"corpus={corpus_digest()}"])
        assert fingerprint(idx) == hashlib.sha1(before.encode()).hexdigest()[:12]
    finally:
        idx.close()


def test_every_stored_variant_gets_its_own_collection(jina, tmp_path):
    from tulpar_ai.rag.embed import JinaEmbedder
    from tulpar_ai.rag.index import Index
    from tulpar_ai.rag.variants import RetrievalVariant

    variants = [RetrievalVariant(), RetrievalVariant(hybrid=True, bm25="lite"), RetrievalVariant(late_chunking=True),
                RetrievalVariant(embed_dim=512), RetrievalVariant(embed_dim=256), RetrievalVariant(context_headers=True),
                RetrievalVariant(hybrid=True, bm25="lite", late_chunking=True, embed_dim=256, context_headers=True)]
    names = []
    for v in variants:
        idx = Index(embedder=JinaEmbedder(), path=tmp_path / "q", variant=v)
        names.append(idx.collection)
        idx.close()
    assert len(set(names)) == len(names)
    assert all(n.startswith(names[0]) for n in names[1:])
    assert names[1].endswith("_hbm25lite") and names[3].endswith("_d512") and names[5].endswith("_ctx1")
    # multi-query is query-side only: same stored index as the default
    idx = Index(embedder=JinaEmbedder(), path=tmp_path / "q", variant=RetrievalVariant(multi_query="en"))
    assert idx.collection == names[0]
    idx.close()


def test_knobs_the_local_embedder_cannot_honour_stay_off(env, tmp_path):
    from tulpar_ai.rag.index import Index
    from tulpar_ai.rag.variants import RetrievalVariant

    idx = Index(path=tmp_path / "q", variant=RetrievalVariant(late_chunking=True, embed_dim=256))
    default = Index(path=tmp_path / "q", variant=RetrievalVariant())
    try:
        assert not idx.late and idx.dim == 768 and idx.collection == default.collection
    finally:
        idx.close()


def test_settings_are_validated(env, monkeypatch):
    from tulpar_ai.rag.variants import RetrievalVariant

    _set(monkeypatch, rag_embed_dim=300)
    with pytest.raises(ValueError):
        RetrievalVariant.from_settings()
    _set(monkeypatch, rag_embed_dim=512, rag_multi_query="de")
    with pytest.raises(ValueError):
        RetrievalVariant.from_settings()


# ── Matryoshka ───────────────────────────────────────────────────────────────
def test_truncate_renormalises():
    from tulpar_ai.rag.variants import truncate

    assert truncate([3.0, 4.0, 12.0], 2) == pytest.approx([0.6, 0.8])
    v = fake_vector("x")
    assert math.isclose(sum(x * x for x in truncate(v, 256)), 1.0, rel_tol=1e-9)
    assert truncate(v, 2048) == v


async def test_smaller_dims_reuse_cached_vectors_without_api_calls(jina, small_corpus, tmp_path, monkeypatch):
    from tulpar_ai.rag.embed import JinaEmbedder
    from tulpar_ai.rag.index import Index
    from tulpar_ai.rag.variants import RetrievalVariant, truncate

    _set(monkeypatch, rag_embed_cache="passages")
    full = Index(embedder=JinaEmbedder(), path=tmp_path / "q", variant=RetrievalVariant())
    n = await full.build()
    passages = len(jina.inputs("retrieval.passage"))
    assert passages == n
    small = Index(embedder=JinaEmbedder(), path=tmp_path / "q", variant=RetrievalVariant(embed_dim=256))
    assert await small.build() == n
    assert len(jina.inputs("retrieval.passage")) == passages  # the 256-d index cost no embedding call
    info = small.client.get_collection(small.collection)
    assert info.config.params.vectors.size == 256
    [p], _ = small.client.scroll(small.collection, limit=1, with_vectors=True, with_payload=True)
    assert np.allclose(p.vector, truncate(fake_vector(p.payload["text"]), 256), atol=1e-6)
    q = await small.embed_query("Сколько пить воды?")
    assert len(q) == 256 and math.isclose(float(np.linalg.norm(q)), 1.0, rel_tol=1e-6)
    hits = await small.search("Сколько пить воды?", 2)
    assert len(hits) == 2 and all(-1.0 <= h["score"] <= 1.0 for h in hits)
    full.close()


# ── the embedding cache ──────────────────────────────────────────────────────
async def test_embedding_cache_by_model_task_text_and_options(jina, monkeypatch):
    from tulpar_ai.config import get_settings
    from tulpar_ai.rag.emb_cache import cache_key, get_embedding_cache
    from tulpar_ai.rag.embed import JinaEmbedder

    monkeypatch.delenv("RAG_EMBED_CACHE", raising=False)  # another test may have left settings with the cache on
    get_settings.cache_clear()

    assert len({cache_key("m", "retrieval.passage", "a"), cache_key("m", "retrieval.query", "a"),
                cache_key("m2", "retrieval.passage", "a"), cache_key("m", "retrieval.passage", "a", "late:x:0")}) == 4
    emb = JinaEmbedder()
    assert get_embedding_cache("retrieval.passage") is None  # off by default
    await emb.embed(["a"], "retrieval.passage")
    await emb.embed(["a"], "retrieval.passage")
    assert len(jina.bodies) == 2

    _set(monkeypatch, rag_embed_cache="passages")
    first = await emb.embed(["a", "b"], "retrieval.passage")
    again = await emb.embed(["b", "a"], "retrieval.passage")
    assert len(jina.bodies) == 3 and np.allclose(again, [first[1], first[0]], atol=1e-6)
    await emb.embed(["q"], "retrieval.query")
    await emb.embed(["q"], "retrieval.query")
    assert len(jina.bodies) == 5  # query vectors are not written to disk in "passages" mode

    _set(monkeypatch, rag_embed_cache="all")
    await emb.embed(["q"], "retrieval.query")
    await emb.embed(["q"], "retrieval.query")
    assert len(jina.bodies) == 6


async def test_jina_retries_429_with_backoff(env, monkeypatch):
    from tulpar_ai.rag import embed

    waits = []

    async def no_sleep(s):
        waits.append(s)

    monkeypatch.setattr(embed.asyncio, "sleep", no_sleep)
    answers = [httpx.Response(429, headers={"retry-after": "3"}), httpx.Response(429),
               httpx.Response(200, json={"data": [], "usage": {"total_tokens": 7}})]

    class Client:
        async def post(self, url, headers, json):
            r = answers.pop(0)
            r.request = httpx.Request("POST", url)
            return r

    assert await embed._post(Client(), "/embeddings", {}) == {"data": [], "usage": {"total_tokens": 7}}
    assert waits == [3.0, 2.0]  # Retry-After first, then the doubled default delay


# ── late chunking ────────────────────────────────────────────────────────────
def _chunk(source, page=None, i=0, text="x" * 100):
    from tulpar_ai.rag.index import Chunk

    return Chunk(id=f"{source}:{page}:{i}", source=source, title=source, text=text, page=page, chunk_index=i)


def test_late_groups_cards_single_file_together_pdf_by_page_window():
    from tulpar_ai.rag.index import late_groups

    chunks = ([_chunk("exercises", i=i) for i in range(3)] + [_chunk("nutrition", i=i) for i in range(4)]
              + [_chunk("who2020", page=p, i=i) for p in range(1, 8) for i in range(2)])
    texts = [c.text for c in chunks]
    groups = late_groups(chunks, texts, max_chars=10_000, window_pages=3)
    assert sorted(i for g in groups for i in g) == list(range(len(chunks)))  # each chunk exactly once
    by = [[chunks[i] for i in g] for g in groups]
    assert sum(1 for g in by if g[0].source == "exercises") == 3 and all(len(g) == 1 for g in by if g[0].source == "exercises")
    assert [len(g) for g in by if g[0].source == "nutrition"] == [4]
    assert [sorted({c.page for c in g}) for g in by if g[0].source == "who2020"] == [[1, 2, 3], [4, 5, 6], [7]]
    tight = late_groups(chunks, texts, max_chars=250, window_pages=3)  # 100-char chunks: at most 2 per group
    assert all(sum(len(texts[i]) for i in g) <= 250 for g in tight)


async def test_late_chunking_requests_and_halving(jina):
    from tulpar_ai.rag.embed import JinaEmbedder

    emb = JinaEmbedder()
    out = await emb.embed_late([["a", "b", "c"], ["solo"], ["card"]], "retrieval.passage")
    assert [len(g) for g in out] == [3, 1, 1]
    assert jina.bodies[0] == {"model": "jina-embeddings-v3", "task": "retrieval.passage", "input": ["solo", "card"]}
    assert jina.bodies[1]["late_chunking"] is True and jina.bodies[1]["input"] == ["a", "b", "c"]
    assert out[1] == [fake_vector("solo")]  # lone chunks: plain embeddings, batched in one request
    assert not np.allclose(out[0][0], fake_vector("a"))  # context-dependent, not the plain vector
    jina.too_long = 2
    jina.bodies.clear()
    out = await emb.embed_late([["a", "b", "c", "d"]], "retrieval.passage")
    assert len(out[0]) == 4 and [b["input"] for b in jina.bodies] == [["a", "b", "c", "d"], ["a", "b"], ["c", "d"]]


async def test_late_chunking_index_keeps_cards_single(jina, small_corpus, tmp_path):
    from tulpar_ai.rag.embed import JinaEmbedder
    from tulpar_ai.rag.index import Index, load_chunks
    from tulpar_ai.rag.variants import RetrievalVariant

    idx = Index(embedder=JinaEmbedder(), path=tmp_path / "q", variant=RetrievalVariant(late_chunking=True))
    try:
        n = await idx.build()
        late = [b for b in jina.bodies if b.get("late_chunking")]
        nutrition = [c.text for c in load_chunks() if c.source == "nutrition"]
        assert len(late) == 1 and late[0]["input"] == nutrition  # the whole file as one document, in order
        plain = [t for b in jina.bodies if not b.get("late_chunking") for t in b["input"]]
        assert sorted(plain) == sorted(c["text"] for c in CARDS)  # cards: plain embeddings, one each
        assert n == len(CARDS) + len(nutrition) == 6
    finally:
        idx.close()


# ── contextual headers ───────────────────────────────────────────────────────
def test_who_sections_from_headings_summary_cues_and_carry_over():
    from tulpar_ai.rag.context import who_page_section

    adults, sed = who_page_section("ADULTS (aged 18–64 years)\nIn adults, physical activity confers benefits", None)
    assert adults.startswith("Взрослые 18–64") and not sed
    older, _ = who_page_section("It is recommended that:\nAll older adults should undertake regular physical activity.", adults)
    assert older.startswith("Пожилые")
    chronic, _ = who_page_section("ADULTS AND OLDER ADULTS WITH CHRONIC\nCONDITIONS (aged 18 years and older)", older)
    assert chronic.startswith("Взрослые и пожилые с хроническими")
    carried, sed = who_page_section("Supporting evidence and rationale\nSedentary behaviour was not included", chronic)
    assert carried == chronic and sed
    wrapped, _ = who_page_section("CHILDREN AND ADOLESCENTS (aged 5–17 years) AND ADULTS\n(aged 18 years and older) "
                                  "LIVING WITH DISABILITY", None)
    assert "инвалидност" in wrapped


async def test_headers_go_into_the_embedded_text_not_the_payload(jina, small_corpus, tmp_path):
    from tulpar_ai.rag.context import header
    from tulpar_ai.rag.embed import JinaEmbedder
    from tulpar_ai.rag.index import Index, load_chunks
    from tulpar_ai.rag.variants import RetrievalVariant

    chunks = load_chunks()
    card = next(c for c in chunks if c.title == "Приседания со штангой")
    water = next(c for c in chunks if "Вода" in c.title)
    assert header(card) == "Упражнение · ноги · штанга"
    assert header(next(c for c in chunks if c.title == "Планка")) == "Упражнение · пресс"
    assert header(water) == "Правила питания Tulpar · Вода"
    who = _chunk("who2020", page=42)
    assert header(who, {42: ("Взрослые 18–64 лет (adults)", True)}) == \
        "ВОЗ 2020 (WHO 2020) · Взрослые 18–64 лет (adults) · сидячий образ жизни (sedentary behaviour) · стр. 42"

    idx = Index(embedder=JinaEmbedder(), path=tmp_path / "q", variant=RetrievalVariant(context_headers=True))
    try:
        n = await idx.build()
        embedded = jina.inputs("retrieval.passage")
        assert len(embedded) == n and f"{header(card)}\n{card.text}" in embedded
        points, _ = idx.client.scroll(idx.collection, limit=100, with_payload=True)
        assert {p.payload["text"] for p in points} == {c.text for c in chunks}  # the answer sees the original text
    finally:
        idx.close()


# ── BM25 and hybrid ──────────────────────────────────────────────────────────
def test_lite_bm25_stems_and_drops_stop_words():
    from tulpar_ai.rag.sparse import LiteBm25, lite_terms

    assert lite_terms("Приседания и приседаний") == ["присед", "присед"]
    assert lite_terms("activities of the activity") == ["activ", "activ"]
    enc = LiteBm25()
    [doc] = enc.embed_documents(["Приседания со штангой: техника приседаний"])
    q = enc.embed_query("приседания")
    assert len(q.indices) == 1 and set(q.indices) <= set(doc.indices) and q.values == [1.0]
    tf = dict(zip(doc.indices, doc.values))
    assert tf[q.indices[0]] > max(v for i, v in tf.items() if i != q.indices[0])  # twice in the text: higher weight


def test_fastembed_bm25_routes_words_by_script_offline(tmp_path):
    pytest.importorskip("fastembed")  # optional: only the (off by default) hybrid variant needs it
    from tulpar_ai.rag.sparse import FastembedBm25, split_scripts

    assert split_scripts("ВОЗ: 150 минут, moderate-intensity") == ("воз минут", "150 moderate intensity")
    model = tmp_path / "bm25"  # a local model folder: stop-lists only, so nothing is downloaded
    model.mkdir()
    (model / "russian.txt").write_text("и\nв\nсо\n", encoding="utf-8")
    (model / "english.txt").write_text("the\nof\nshould\n", encoding="utf-8")
    enc = FastembedBm25(cache_dir=tmp_path / "cache", model_path=model)
    ru, en = enc.embed_documents(["Приседания со штангой", "Adults should do muscle-strengthening activities"])
    assert set(enc.embed_query("приседаний").indices) <= set(ru.indices)  # Snowball: приседания ~ приседаний
    assert set(enc.embed_query("activity").indices) <= set(en.indices)  # Snowball: activities ~ activity
    assert not set(enc.embed_query("со").indices) and not set(enc.embed_query("the").indices)  # stop-words


async def test_hybrid_index_stores_dense_and_sparse_and_fuses(env, small_corpus, tmp_path):
    from tulpar_ai.rag.index import DENSE, SPARSE, Index
    from tulpar_ai.rag.variants import RetrievalVariant

    idx = Index(path=tmp_path / "q", variant=RetrievalVariant(hybrid=True, bm25="lite"))
    try:
        assert idx.collection.endswith("_hbm25lite")
        await idx.build()
        params = idx.client.get_collection(idx.collection).config.params
        assert DENSE in params.vectors and SPARSE in params.sparse_vectors
        hits = await idx.search("планка поясница", 3)
        assert hits[0]["title"] == "Планка"
        assert all("fusion_score" in h and -1.0 <= h["score"] <= 1.0 for h in hits)
        assert hits == sorted(hits, key=lambda h: -h["fusion_score"])
        only_stop = await idx.search("и в на", 2)  # nothing for BM25: dense list alone
        assert len(only_stop) == 2
    finally:
        idx.close()


def test_rrf_fuse_sums_reciprocal_ranks_and_keeps_best_cosine():
    from tulpar_ai.rag.variants import rrf_fuse

    a = [{"id": "x", "score": 0.5}, {"id": "y", "score": 0.4}]
    b = [{"id": "y", "score": 0.7}, {"id": "z", "score": 0.6}]
    fused = rrf_fuse([a, b], k=60)
    assert [h["id"] for h in fused] == ["y", "x", "z"]
    assert fused[0]["fusion_score"] == pytest.approx(1 / 62 + 1 / 61) and fused[0]["score"] == 0.7
    assert len(rrf_fuse([a, b], limit=2)) == 2


# ── cross-lingual multi-query ────────────────────────────────────────────────
class _StubIndex:
    def __init__(self, multi_query="en"):
        from tulpar_ai.rag.embed import LocalHashEmbedder
        from tulpar_ai.rag.variants import RetrievalVariant

        self.embedder = LocalHashEmbedder()
        self.variant = RetrievalVariant(multi_query=multi_query)
        self.queries: list[str] = []

    async def search(self, query, limit):
        self.queries.append(query)
        if query.isascii():
            return [{"id": "who:12", "source": "who2020", "title": "WHO", "page": 12, "text": "150–300 minutes", "score": 0.6},
                    {"id": "who:13", "source": "who2020", "title": "WHO", "page": 13, "text": "sedentary", "score": 0.3}]
        return [{"id": "ex:1", "source": "exercises", "title": "a", "text": "a", "score": 0.5},
                {"id": "who:12", "source": "who2020", "title": "WHO", "page": 12, "text": "150–300 minutes", "score": 0.4}]


async def test_russian_question_is_also_searched_in_english(env, fake_llm, monkeypatch):
    from langsmith.run_helpers import is_traceable_function

    from tulpar_ai.rag import translate
    from tulpar_ai.rag.retrieve import retrieve

    translate.reset_memo()

    def llm(*, role, system, user, images, json_mode):
        assert role == "route" and "переводишь" in system
        return json.dumps({"query": "How many minutes a week of physical activity does WHO recommend?"})

    from tulpar_ai import llm as llm_mod

    llm_mod.set_fake(llm)  # the fake_llm fixture resets it afterwards
    idx = _StubIndex()
    hits = await retrieve("Сколько минут в неделю рекомендует ВОЗ?", top_n=2, rerank=False, index=idx)
    assert idx.queries == ["Сколько минут в неделю рекомендует ВОЗ?",
                           "How many minutes a week of physical activity does WHO recommend?"]
    assert hits[0]["id"] == "who:12" and hits[0]["queries"] == ["ru", "en"]  # in both lists, ranks 2 and 1
    assert hits[0]["rerank_score"] == 0.6  # the best cosine, not the tiny RRF number
    assert is_traceable_function(translate.to_english)

    idx.queries.clear()
    await retrieve("How much activity for adults?", rerank=False, index=idx)
    assert idx.queries == ["How much activity for adults?"]  # English: no translation
    off = _StubIndex(multi_query="off")
    await retrieve("Сколько минут в неделю?", rerank=False, index=off)
    assert off.queries == ["Сколько минут в неделю?"]


async def test_glossary_fallback_without_llm(env, fake_llm):
    from tulpar_ai import llm as llm_mod
    from tulpar_ai.rag import translate

    translate.reset_memo()
    llm_mod.set_fake(None)  # no keys in tests: every provider fails → glossary
    out = await translate.to_english("Маме 68 лет. Что по ВОЗ ей нужно кроме ходьбы, чтобы не падать?")
    assert out == "older adults WHO walking falls 68"
    assert await translate.to_english("How much?") == ""
    assert translate.is_russian("Сколько минут?") and not translate.is_russian("WHO 150 minutes")


# ── the chat graph and the answer cache ──────────────────────────────────────
async def test_sufficiency_uses_the_best_score_of_fused_hits(env, monkeypatch):
    from tulpar_ai.graph import chat

    async def fused(q):
        return [{"text": "a", "rerank_score": 0.12}, {"text": "b", "rerank_score": 0.55}]

    monkeypatch.setattr(chat, "retrieve", fused)
    out = await chat.retrieve_node({"text": "вопрос"})
    assert out["sufficient"] is True


def test_answer_cache_fingerprint_follows_query_side_variant(env):
    from tulpar_ai.rag.answer_cache import fingerprint
    from tulpar_ai.rag.index import Index
    from tulpar_ai.rag.variants import RetrievalVariant

    a = Index(variant=RetrievalVariant())
    b = Index(variant=RetrievalVariant(multi_query="en"))
    try:
        assert a.collection == b.collection and fingerprint(a) != fingerprint(b)
    finally:
        a.close()


# ── build_retriever: the contract evals use ─────────────────────────────────
async def test_build_retriever_contract(env, small_corpus, tmp_path):
    from tulpar_ai.config import get_settings
    from tulpar_ai.rag.variants import Retriever, build_retriever

    r = build_retriever(overrides={"rag_hybrid": True, "rag_bm25": "lite"}, path=tmp_path / "q", rerank=False)
    try:
        assert isinstance(r, Retriever) and r.variant.hybrid and not get_settings().rag_hybrid
        assert await r.build() == len(CARDS) + 3
        hits = await r.retrieve("Сколько воды пить на килограмм веса?", k=2)
        assert len(hits) == 2 and {"id", "payload", "score", "rerank_score", "fusion_score"} <= set(hits[0])
        assert hits[0]["payload"]["source"] in ("nutrition", "exercises") and "text" in hits[0]["payload"]
        assert "score" not in hits[0]["payload"]
        assert await r.retrieve_ids("Сколько воды пить на килограмм веса?", k=2) == [h["id"] for h in hits]
    finally:
        r.close()
