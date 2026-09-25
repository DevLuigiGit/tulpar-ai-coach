"""Existing Tulpar splitter and document metadata; no embeddings or storage."""

import json
import uuid
from pathlib import Path

from .cleaner import normalize_text
from .parser import parse_document

# Preserve the namespace used by existing document IDs and Qdrant point IDs.
NS = uuid.UUID("0d7a3c52-4b1e-4f9a-8c61-2e5d7b9a1f33")


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    text = normalize_text(text)
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            for separator in ("\n\n", "\n", ". ", " "):
                cut = text.rfind(separator, start + overlap + 1, end)
                if cut >= 0:
                    end = cut + len(separator)
                    break
            else:
                cut = text.rfind(" ", start + 1, end)
                if cut >= 0:
                    end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        while next_start > start + 1 and not text[next_start - 1].isspace():
            next_start -= 1
        if next_start <= start + 1 and not text[next_start - 1].isspace():
            next_start = max(end - overlap, start + 1)
            while next_start < end and not text[next_start - 1].isspace():
                next_start += 1
        start = next_start
    return [c for c in out if c]


def chunk_document(path: Path, *, corpus_root: Path, size: int = 400, overlap: int = 120) -> list[dict]:
    """Return payloads for the existing Chunk, preserving document IDs and metadata.

    Source paths are relative to corpus_root, distinguishing equal filenames in
    different directories. chunk_index is zero-based within each parsed page
    (the entire DOCX is one page=None block). WHO aliases remain compatible.
    """
    path, corpus_root = Path(path), Path(corpus_root)
    source = path.relative_to(corpus_root).as_posix()
    who = source == "who_2020_physical_activity.pdf"
    chunks = []
    for parsed in parse_document(path):
        page = parsed["page"]
        for chunk_index, part in enumerate(split_text(parsed["text"], size, overlap)):
            identity = json.dumps([source, size, overlap, page, chunk_index], ensure_ascii=False)
            chunk_id = f"who:{size}:{page}:{chunk_index}" if who else f"doc:{uuid.uuid5(NS, identity)}"
            chunks.append({
                **parsed,
                "id": chunk_id,
                "source": "who2020" if who else source,
                "title": "WHO guidelines on physical activity and sedentary behaviour (2020)" if who else path.name,
                "text": part,
                "chunk_index": chunk_index,
            })
    return chunks


