"""Corpus → chunks → vectors in an embedded Qdrant (a folder on disk).

Chunking is chosen per source, because the sources have different natural units:
  exercises.jsonl  one exercise card = one chunk (≤ ~600 chars, a self-contained answer)
  nutrition.md     split on markdown headings, then paragraphs (rules are short and atomic)
  PDF / DOCX       paragraph-first split with 120 overlap (Index defaults to 400 chars);
                   PDF keeps physical pages, DOCX uses page=None
Embedded Qdrant keeps the same API as a Qdrant server — QDRANT_URL switches to a server (see rag/qdrant.py).

Retrieval variants (rag/variants.py) change what is stored — named dense + sparse vectors, late-chunked or
header-prefixed embeddings, a smaller Matryoshka dimension — and each adds a suffix to the collection name.
The default setup keeps its old name and an unnamed dense vector, so an existing index is reused as is.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from langsmith import traceable
from qdrant_client import models

from ..config import ROOT, get_settings
from . import context
from .embed import Embedder, get_embedder
from .qdrant import close_client, get_client
from .sparse import SparseEncoder, get_sparse_encoder
from .variants import RetrievalVariant, truncate
from parsing.chunker import NS, PARSING_VERSION, WHO_FILE, WHO_SOURCE, chunk_document, document_source, split_text

CORPUS = ROOT / "corpus"
DOCUMENT_SUFFIXES = {".pdf", ".docx"}
QUERY_MEMO = 64
DENSE, SPARSE = "dense", "bm25"  # vector names in a hybrid collection

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    text: str
    page: int | None = None
    muscle_group: str | None = None
    equipment: str | None = None
    file_type: str | None = None
    chunk_index: int = 0


def load_chunks(pdf_chunk: int = 800, pdf_overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    if not CORPUS.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {CORPUS}")
    exercises = CORPUS / "exercises.jsonl"
    for line in (exercises.read_text(encoding="utf-8").splitlines() if exercises.exists() else []):
        d = json.loads(line)
        chunks.append(Chunk(id=f"ex:{d['id']}", source="exercises", title=d["title"], text=d["text"],
                            muscle_group=d.get("muscle_group"), equipment=d.get("equipment"), file_type="jsonl"))
    nutrition = CORPUS / "nutrition.md"
    md = nutrition.read_text(encoding="utf-8") if nutrition.exists() else ""
    section = "Питание"
    for block in re.split(r"\n(?=#+ )", md):
        m = re.match(r"#+ (.+)", block)
        if m:
            section = m.group(1).strip()
        for j, part in enumerate(split_text(block, 900, 100)):
            chunks.append(Chunk(id=f"nut:{section}:{j}", source="nutrition", title=f"Правила питания Tulpar — {section}",
                                text=part, file_type="md", chunk_index=j))
    for path in corpus_documents():
        chunks.extend(document_chunks(path, pdf_chunk, pdf_overlap))
    return chunks


def corpus_documents() -> list[Path]:
    return [path for path in sorted(CORPUS.rglob("*")) if _is_document(path)] if CORPUS.is_dir() else []


def document_chunks(path: Path, pdf_chunk: int, pdf_overlap: int = 120) -> list[Chunk]:
    """Chunks of one PDF/DOCX; [] with a warning when it cannot be read."""
    try:
        payloads = chunk_document(path, corpus_root=CORPUS, size=pdf_chunk, overlap=pdf_overlap)
    except Exception:  # one broken upload must not take the whole index (and every answer) down
        log.warning("Skipping unreadable document %s", path.relative_to(CORPUS), exc_info=True)
        return []
    return [Chunk(**payload) for payload in payloads]


def _is_document(path: Path) -> bool:
    """PDF/DOCX files, minus Word lock files (~$name.docx) and hidden files that sit next to real ones."""
    return (path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES
            and not path.name.startswith(("~$", ".")))


def corpus_fingerprint() -> str:
    """Short hash of the text corpus: an edited exercise card or rule gets a fresh collection on the next start."""
    h = hashlib.sha1()
    for name in ("exercises.jsonl", "nutrition.md"):
        f = CORPUS / name
        h.update(f.read_bytes() if f.exists() else b"")
    return h.hexdigest()[:8]


def late_groups(chunks: list[Chunk], texts: list[str], max_chars: int, window_pages: int) -> list[list[int]]:
    """Positions of `chunks` grouped for late chunking; every chunk is in exactly one group, order kept inside a group.

    Exercise cards are self-contained: one card, one group. Nutrition rules: the file in order. PDF/DOCX: consecutive
    chunks of one document within a window of `window_pages` pages. A group never exceeds `max_chars` of embedded
    text (a single longer chunk still gets its own group), which keeps a request inside Jina's 8192-token context.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    key0, page0, chars = None, 0, 0
    order = sorted(range(len(chunks)), key=lambda i: (chunks[i].source != "exercises", chunks[i].source,
                                                      chunks[i].page or 0, chunks[i].chunk_index, i))
    for i in order:
        c = chunks[i]
        if c.source == "exercises":
            groups.append([i])
            continue
        page = c.page or 0
        if current and (c.source != key0 or page - page0 >= window_pages or chars + len(texts[i]) > max_chars):
            groups.append(current)
            current = []
        if not current:
            key0, page0, chars = c.source, page, 0
        current.append(i)
        chars += len(texts[i])
    if current:
        groups.append(current)
    return groups


def document_path(source: str) -> Path:
    return CORPUS / (WHO_FILE if source == WHO_SOURCE else source)


class Index:
    def __init__(self, embedder: Embedder | None = None, path: Path | None = None, pdf_chunk: int = 400,
                 variant: RetrievalVariant | None = None):
        s = get_settings()
        self.embedder = embedder or get_embedder()
        self.pdf_chunk = pdf_chunk
        self.path = path or Path(s.ai_data_dir) / "qdrant"
        self.variant = variant or RetrievalVariant.from_settings(s)
        # A knob the embedder cannot honour is off, and stays out of the name: late chunking and Matryoshka are
        # Jina features; the local hashing embedder has neither.
        self.dim = self.embedder.dim
        if getattr(self.embedder, "matryoshka", False) and self.variant.embed_dim < self.embedder.dim:
            self.dim = self.variant.embed_dim
        self.late = self.variant.late_chunking and getattr(self.embedder, "late_chunking", False)
        self.headers = self.variant.context_headers
        self.sparse: SparseEncoder | None = get_sparse_encoder(self.variant.bm25) if self.variant.hybrid else None
        self.collection = f"coach_{self.embedder.id}_{pdf_chunk}_p{PARSING_VERSION}_{corpus_fingerprint()}" + self._suffix()
        self.client = get_client(self.path)
        self._qvecs: OrderedDict[str, list[float]] = OrderedDict()
        self._sections: dict[str, dict] = {}

    def _suffix(self) -> str:
        parts = []
        if self.sparse is not None:
            parts.append(f"h{self.sparse.id}")
        if self.late:
            parts.append(f"lc{self.variant.late_window_pages}x{self.variant.late_max_chars}")
        if self.dim != self.embedder.dim:
            parts.append(f"d{self.dim}")
        if self.headers:
            parts.append(f"ctx{context.HEADER_VERSION}")
        return "".join(f"_{p}" for p in parts)

    def close(self) -> None:
        close_client(self.path)

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection).count

    async def build(self, force: bool = False) -> int:
        if self.count() and not force:
            await self.add_missing_documents()
            return self.count()
        return await self._rebuild()

    # One labelled trace per real (re)index, not a bare `jina_embed` root; the no-op path above is not traced.
    @traceable(run_type="chain", name="rag_index_build")
    async def _rebuild(self) -> int:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        dense = models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
        if self.sparse is None:
            self.client.create_collection(self.collection, vectors_config=dense)
        else:  # IDF is Qdrant's side of BM25: the sparse vectors hold only the term-frequency part
            self.client.create_collection(self.collection, vectors_config={DENSE: dense}, sparse_vectors_config={
                SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)})
        chunks = load_chunks(pdf_chunk=self.pdf_chunk)
        await self._upsert(chunks)
        return len(chunks)

    async def add_missing_documents(self) -> int:
        """Index corpus documents that have no points yet; returns how many chunks were added.

        A document skipped as unreadable during a build (a transient I/O or parser failure) would otherwise
        stay out of the persisted index until PARSING_VERSION changes, because a non-empty collection is
        reused as is. Documents already present are neither re-parsed nor re-embedded.
        """
        added = 0
        for path in corpus_documents():
            source = document_source(path, CORPUS)
            if self._has_source(source):
                continue
            chunks = document_chunks(path, self.pdf_chunk)
            if not chunks:  # still unreadable, or no text layer: tried again on the next start
                continue
            try:
                await self._upsert(chunks)
            except Exception:  # the existing index still answers; the next start retries
                log.warning("Could not add %s to %s", source, self.collection, exc_info=True)
                continue
            log.info("Added %s chunks of %s missing from %s", len(chunks), source, self.collection)
            added += len(chunks)
        return added

    def _has_source(self, source: str) -> bool:
        only = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])
        return self.client.count(self.collection, count_filter=only, exact=True).count > 0

    def embed_text(self, chunk: Chunk) -> str:
        """What the embedder (and BM25) reads for a chunk; the payload keeps the original text for the answer."""
        if not self.headers:
            return chunk.text
        sections = None
        if chunk.page is not None and chunk.source not in ("exercises", "nutrition"):
            if chunk.source not in self._sections:
                self._sections[chunk.source] = context.page_sections(document_path(chunk.source), chunk.source)
            sections = self._sections[chunk.source]
        return context.with_header(chunk, sections)

    async def _passage_vectors(self, chunks: list[Chunk], texts: list[str]) -> list[list[float]]:
        if self.late:
            groups = late_groups(chunks, texts, self.variant.late_max_chars, self.variant.late_window_pages)
            grouped = await self.embedder.embed_late([[texts[i] for i in g] for g in groups], task="retrieval.passage")
            vectors: list[list[float]] = [[] for _ in chunks]
            for g, vecs in zip(groups, grouped):
                for i, v in zip(g, vecs):
                    vectors[i] = v
        else:
            vectors = await self.embedder.embed(texts, task="retrieval.passage")
        return [truncate(v, self.dim) for v in vectors] if self.dim != self.embedder.dim else vectors

    async def _upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [self.embed_text(c) for c in chunks]
        vectors = await self._passage_vectors(chunks, texts)
        if self.sparse is None:
            points = [models.PointStruct(id=str(uuid.uuid5(NS, c.id)), vector=v, payload=c.__dict__)
                      for c, v in zip(chunks, vectors)]
        else:
            sparse = self.sparse.embed_documents(texts)
            points = [models.PointStruct(id=str(uuid.uuid5(NS, c.id)), vector={DENSE: v, SPARSE: sv}, payload=c.__dict__)
                      for c, v, sv in zip(chunks, vectors, sparse)]
        self.client.upsert(self.collection, points=points)

    async def embed_query(self, query: str) -> list[float]:
        """The answer cache and the search embed the same question in one turn: remember recent vectors."""
        if query in self._qvecs:
            self._qvecs.move_to_end(query)
            return self._qvecs[query]
        [vec] = await self.embedder.embed([query], task="retrieval.query")
        if self.dim != self.embedder.dim:
            vec = truncate(vec, self.dim)
        self._qvecs[query] = vec
        if len(self._qvecs) > QUERY_MEMO:
            self._qvecs.popitem(last=False)
        return vec

    async def search(self, query: str, limit: int) -> list[dict]:
        """Top `limit` chunks. `score` is always the dense cosine; a hybrid search adds the RRF `fusion_score`."""
        vec = await self.embed_query(query)
        if self.sparse is None:
            res = self.client.query_points(self.collection, query=vec, limit=limit, with_payload=True)
            return [{**p.payload, "score": float(p.score)} for p in res.points]
        sq = self.sparse.embed_query(query)
        if not sq.indices:  # only stop-words: nothing for BM25 to match, the dense list alone
            res = self.client.query_points(self.collection, query=vec, using=DENSE, limit=limit, with_payload=True)
            return [{**p.payload, "score": float(p.score), "fusion_score": 0.0} for p in res.points]
        res = self.client.query_points(
            self.collection,
            prefetch=[models.Prefetch(query=vec, using=DENSE, limit=limit),
                      models.Prefetch(query=sq, using=SPARSE, limit=limit)],
            query=models.RrfQuery(rrf=models.Rrf(k=self.variant.rrf_k)),
            limit=limit, with_payload=True, with_vectors=[DENSE])
        qn = _unit(vec)
        return [{**p.payload, "score": _dot(qn, _unit(p.vector[DENSE])), "fusion_score": float(p.score)}
                for p in res.points]


def _unit(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))
