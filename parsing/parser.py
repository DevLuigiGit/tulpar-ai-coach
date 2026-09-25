"""Extract PDF pages and DOCX paragraphs/tables with source metadata.

PDF text comes from PDFium via pypdfium2 (Apache-2.0 / BSD-3-Clause). PyMuPDF is AGPL-3.0, which would
force the whole commercial Tulpar SaaS open once this package is reused there; on the WHO PDF both give
the same text (EVALS.md). OCR is not performed: a scanned PDF yields empty pages and a warning.
"""

import logging
from pathlib import Path

import pypdfium2
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

log = logging.getLogger(__name__)

# PDFium drops the line break after a hyphen that ends a line and marks the hyphen with U+0002.
_PDFIUM_EOL_HYPHEN = "\x02"


def parse_document(path: Path) -> list[dict]:
    path = Path(path)
    file_type = path.suffix.lower().lstrip(".")
    metadata = {"source": path.name, "file_type": file_type}
    if file_type == "pdf":
        blocks = [{**metadata, "page": number, "text": text}
                  for number, text in enumerate(_pdf_pages(path), start=1)]
    elif file_type == "docx":
        blocks = [{**metadata, "page": None, "text": _container_text(Document(path))}]
    else:
        raise ValueError(f"Unsupported document type: {path.suffix}")
    if not any(block["text"].strip() for block in blocks):
        log.warning("%s has no extractable text (a scan without a text layer?); OCR is not performed", path.name)
    return blocks


def _pdf_pages(path: Path) -> list[str]:
    document = pypdfium2.PdfDocument(path)
    try:
        pages = []
        for index in range(len(document)):
            page = document[index]
            textpage = page.get_textpage()
            try:
                pages.append(textpage.get_text_bounded().replace(_PDFIUM_EOL_HYPHEN, "-"))
            finally:
                textpage.close()
                page.close()
        return pages
    finally:
        document.close()


def _container_text(container) -> str:
    """Paragraphs and tables of a document or a table cell, in document order."""
    blocks = []
    for block in container.iter_inner_content():
        blocks.append(block.text if isinstance(block, Paragraph) else _table_text(block))
    return "\n\n".join(blocks)


def _table_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells, seen = [], set()
        for cell in row.cells:
            # python-docx repeats a horizontally merged cell once per grid column it spans.
            if id(cell._tc) in seen:
                continue
            seen.add(id(cell._tc))
            cells.append(_container_text(cell).replace("\n\n", "\n"))
        rows.append("\t".join(cells))
    return "\n".join(rows)
