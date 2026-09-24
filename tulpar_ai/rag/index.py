"""Corpus → chunks → vectors in an embedded Qdrant (a folder on disk).

Chunking is chosen per source, because the sources have different natural units:
  exercises.jsonl  one exercise card = one chunk (≤ ~600 chars, a self-contained answer)
  nutrition.md     split on markdown headings, then paragraphs (rules are short and atomic)
  WHO PDF          page by page, recursive split ~800 chars with 120 overlap; the page number is kept
                   so the answer can cite «ВОЗ 2020, стр. N»
Embedded Qdrant keeps the same API as a Qdrant server — switching to Qdrant Cloud is a URL change.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from ..config import ROOT, get_settings
from .embed import Embedder, get_embedder

CORPUS = ROOT / "corpus"
NS = uuid.UUID("0d7a3c52-4b1e-4f9a-8c61-2e5d7b9a1f33")


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    text: str
    page: int | None = None
    muscle_group: str | None = None
    equipment: str | None = None


def split_text(text: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + size // 2:
                end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if c]


def load_chunks(pdf_chunk: int = 800, pdf_overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for line in (CORPUS / "exercises.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        chunks.append(Chunk(id=f"ex:{d['id']}", source="exercises", title=d["title"], text=d["text"],
                            muscle_group=d.get("muscle_group"), equipment=d.get("equipment")))
    md = (CORPUS / "nutrition.md").read_text(encoding="utf-8")
    section = "Питание"
    for block in re.split(r"\n(?=#+ )", md):
        m = re.match(r"#+ (.+)", block)
        if m:
            section = m.group(1).strip()
        for j, part in enumerate(split_text(block, 900, 100)):
            chunks.append(Chunk(id=f"nut:{section}:{j}", source="nutrition", title=f"Правила питания Tulpar — {section}", text=part))
    pdf = CORPUS / "who_2020_physical_activity.pdf"
    if pdf.exists():
        for pno, page in enumerate(PdfReader(str(pdf)).pages, start=1):
            text = page.extract_text() or ""
            for j, part in enumerate(split_text(text, pdf_chunk, pdf_overlap)):
                if len(part) > 80:
                    chunks.append(Chunk(id=f"who:{pdf_chunk}:{pno}:{j}", source="who2020",
                                        title="WHO guidelines on physical activity and sedentary behaviour (2020)",
                                        text=part, page=pno))
    return chunks


class Index:
    def __init__(self, embedder: Embedder | None = None, path: Path | None = None, pdf_chunk: int = 400):
        s = get_settings()
        self.embedder = embedder or get_embedder()
        self.pdf_chunk = pdf_chunk
        self.path = path or Path(s.ai_data_dir) / "qdrant"
        self.collection = f"coach_{self.embedder.id}_{pdf_chunk}"
        self.client = QdrantClient(path=str(self.path))

    def close(self) -> None:
        self.client.close()

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection).count

    async def build(self, force: bool = False) -> int:
        if self.count() and not force:
            return self.count()
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
