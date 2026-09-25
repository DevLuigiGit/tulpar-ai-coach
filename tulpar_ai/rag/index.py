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
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from langsmith import traceable
from qdrant_client import models

from ..config import ROOT, get_settings
from .embed import Embedder, get_embedder
from .qdrant import close_client, get_client
from parsing.chunker import NS, chunk_document, split_text

CORPUS = ROOT / "corpus"


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
    for path in sorted(CORPUS.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".docx"}:
            continue
        chunks.extend(Chunk(**payload) for payload in chunk_document(
            path, corpus_root=CORPUS, size=pdf_chunk, overlap=pdf_overlap))
    return chunks


class Index:
    def __init__(self, embedder: Embedder | None = None, path: Path | None = None, pdf_chunk: int = 400):
        s = get_settings()
        self.embedder = embedder or get_embedder()
        self.pdf_chunk = pdf_chunk
        self.path = path or Path(s.ai_data_dir) / "qdrant"
        self.collection = f"coach_{self.embedder.id}_{pdf_chunk}"
        self.client = get_client(self.path)

    def close(self) -> None:
        close_client(self.path)

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection).count

    async def build(self, force: bool = False) -> int:
        if self.count() and not force:
            return self.count()
        return await self._rebuild()

    # One labelled trace per real (re)index, not a bare `jina_embed` root; the no-op path above is not traced.
    @traceable(run_type="chain", name="rag_index_build")
    async def _rebuild(self) -> int:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(self.collection, vectors_config=models.VectorParams(
            size=self.embedder.dim, distance=models.Distance.COSINE))
        chunks = load_chunks(pdf_chunk=self.pdf_chunk)
        vectors = await self.embedder.embed([c.text for c in chunks], task="retrieval.passage")
        self.client.upsert(self.collection, points=[
            models.PointStruct(id=str(uuid.uuid5(NS, c.id)), vector=v, payload=c.__dict__)
            for c, v in zip(chunks, vectors)
        ])
        return len(chunks)

    async def search(self, query: str, limit: int) -> list[dict]:
        [vec] = await self.embedder.embed([query], task="retrieval.query")
        res = self.client.query_points(self.collection, query=vec, limit=limit, with_payload=True)
        return [{**p.payload, "score": float(p.score)} for p in res.points]
