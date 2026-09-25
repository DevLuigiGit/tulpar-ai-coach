"""Corpus → chunks → vectors in an embedded Qdrant (a folder on disk).

Chunking is chosen per source, because the sources have different natural units:
  exercises.jsonl  one exercise card = one chunk (≤ ~600 chars, a self-contained answer)
  nutrition.md     split on markdown headings, then paragraphs (rules are short and atomic)
  PDF / DOCX       paragraph-first split with 120 overlap (Index defaults to 400 chars);
                   PDF keeps physical pages, DOCX uses page=None
Embedded Qdrant keeps the same API as a Qdrant server — QDRANT_URL switches to a server (see rag/qdrant.py).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import models

from ..config import ROOT, get_settings
from .embed import Embedder, get_embedder
from .qdrant import close_client, get_client
from parsing.chunker import NS, PARSING_VERSION, chunk_document, document_source, split_text

CORPUS = ROOT / "corpus"
DOCUMENT_SUFFIXES = {".pdf", ".docx"}
QUERY_MEMO = 64

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


class Index:
    def __init__(self, embedder: Embedder | None = None, path: Path | None = None, pdf_chunk: int = 400):
        s = get_settings()
        self.embedder = embedder or get_embedder()
        self.pdf_chunk = pdf_chunk
        self.path = path or Path(s.ai_data_dir) / "qdrant"
        self.collection = f"coach_{self.embedder.id}_{pdf_chunk}_p{PARSING_VERSION}"
        self.client = get_client(self.path)
        self._qvecs: OrderedDict[str, list[float]] = OrderedDict()

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
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(self.collection, vectors_config=models.VectorParams(
            size=self.embedder.dim, distance=models.Distance.COSINE))
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

    async def _upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = await self.embedder.embed([c.text for c in chunks], task="retrieval.passage")
        self.client.upsert(self.collection, points=[
            models.PointStruct(id=str(uuid.uuid5(NS, c.id)), vector=v, payload=c.__dict__)
            for c, v in zip(chunks, vectors)
        ])

    async def embed_query(self, query: str) -> list[float]:
        """The answer cache and the search embed the same question in one turn: remember recent vectors."""
        if query in self._qvecs:
            self._qvecs.move_to_end(query)
            return self._qvecs[query]
        [vec] = await self.embedder.embed([query], task="retrieval.query")
        self._qvecs[query] = vec
        if len(self._qvecs) > QUERY_MEMO:
            self._qvecs.popitem(last=False)
        return vec

    async def search(self, query: str, limit: int) -> list[dict]:
        vec = await self.embed_query(query)
        res = self.client.query_points(self.collection, query=vec, limit=limit, with_payload=True)
        return [{**p.payload, "score": float(p.score)} for p in res.points]
