"""Existing Tulpar splitter and document metadata; no embeddings or storage."""

import json
import uuid
from pathlib import Path

from .cleaner import normalize_text
from .parser import parse_document

# Preserve the namespace used by existing document IDs and Qdrant point IDs.
NS = uuid.UUID("0d7a3c52-4b1e-4f9a-8c61-2e5d7b9a1f33")

# Bump whenever parsing, cleaning or splitting changes the chunks: the index collection name carries it,
# so a persisted index built by an older parser is rebuilt instead of being served forever.
PARSING_VERSION = 2

# A PDF fragment this short is a running header, a page number or an ISBN line: no answer in it, but it
# still competes for the top-k slots. The pre-package indexer dropped these as well.
MIN_PDF_CHUNK_CHARS = 80

_SEPARATORS = ("\n\n", "\n", ". ", " ")

WHO_FILE = "who_2020_physical_activity.pdf"
WHO_SOURCE = "who2020"


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    text = normalize_text(text)
    if len(text) <= size:
        return [text] if text else []
    return [c for c in (text[start:end].strip() for start, end in _spans(text, size, overlap)) if c]


def _spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Chunk boundaries: paragraph → line → sentence → word, the overlap starts on a word boundary.

    Every chunk must end after the previous one: otherwise, when the only separator in the window is
    the one that closed the previous chunk, the next chunk is just the overlap again (a duplicate).
    """
    spans: list[tuple[int, int]] = []
    start, prev_end = 0, 0
    while True:
        end = min(len(text), start + size)
        if end < len(text):
            end = _cut(text, start, end, lo=max(start + overlap, prev_end) + 1, floor=max(start + 1, prev_end))
        spans.append((start, end))
        if end >= len(text):
            return spans
        start, prev_end = _next_start(text, start, end, overlap), end


def _cut(text: str, start: int, end: int, *, lo: int, floor: int) -> int:
    for separator in _SEPARATORS:
        cut = text.rfind(separator, lo, end)
        if cut >= 0:
            return cut + len(separator)
    cut = text.rfind(" ", floor, end)
    return cut + 1 if cut >= 0 else end


def _next_start(text: str, start: int, end: int, overlap: int) -> int:
    next_start = max(end - overlap, start + 1)
    while next_start > start + 1 and not text[next_start - 1].isspace():
        next_start -= 1
    if next_start <= start + 1 and not text[next_start - 1].isspace():
        next_start = max(end - overlap, start + 1)
        while next_start < end and not text[next_start - 1].isspace():
            next_start += 1
    return next_start


def document_source(path: Path, corpus_root: Path) -> str:
    """The `source` payload of a document's chunks: its corpus-relative path, the WHO PDF keeps its old alias."""
    source = Path(path).relative_to(corpus_root).as_posix()
    return WHO_SOURCE if source == WHO_FILE else source


def chunk_document(path: Path, *, corpus_root: Path, size: int = 400, overlap: int = 120) -> list[dict]:
    """Return payloads for the existing Chunk, preserving document IDs and metadata.

    Source paths are relative to corpus_root, distinguishing equal filenames in
    different directories. chunk_index is zero-based within each parsed page
    (the entire DOCX is one page=None block); a dropped short PDF fragment keeps
    its index unused, so the IDs of the other chunks do not move. WHO aliases remain compatible.
    """
    path, corpus_root = Path(path), Path(corpus_root)
    source = document_source(path, corpus_root)
    who = source == WHO_SOURCE
    chunks = []
    for parsed in parse_document(path):
        page = parsed["page"]
        for chunk_index, part in enumerate(split_text(parsed["text"], size, overlap)):
            if parsed["file_type"] == "pdf" and len(part) <= MIN_PDF_CHUNK_CHARS:
                continue
            identity = json.dumps([source, size, overlap, page, chunk_index], ensure_ascii=False)
            chunk_id = f"who:{size}:{page}:{chunk_index}" if who else f"doc:{uuid.uuid5(NS, identity)}"
            chunks.append({
                **parsed,
                "id": chunk_id,
                "source": source,
                "title": "WHO guidelines on physical activity and sedentary behaviour (2020)" if who else path.name,
                "text": part,
                "chunk_index": chunk_index,
            })
    return chunks
