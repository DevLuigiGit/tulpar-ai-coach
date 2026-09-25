"""Extract PDF pages and DOCX paragraphs/tables with source metadata."""

from pathlib import Path

import pymupdf
from docx import Document
from docx.text.paragraph import Paragraph


def parse_document(path: Path) -> list[dict]:
    path = Path(path)
    file_type = path.suffix.lower().lstrip(".")
    metadata = {"source": path.name, "file_type": file_type}
    if file_type == "pdf":
        with pymupdf.open(path) as document:
            return [{**metadata, "page": number, "text": page.get_text()}
                    for number, page in enumerate(document, start=1)]
    if file_type == "docx":
        document = Document(path)
        blocks = []
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                blocks.append(block.text)
            else:
                blocks.append("\n".join("\t".join(cell.text for cell in row.cells) for row in block.rows))
        return [{**metadata, "page": None, "text": "\n\n".join(blocks)}]
    raise ValueError(f"Unsupported document type: {path.suffix}")
