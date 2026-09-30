"""Titles of corpus documents in citations (parsing/chunker.py)."""


def test_docx_title_comes_from_its_properties(tmp_path):
    """Citations show a document's own title («Schoenfeld et al., 2021 (Sports). …»), not «file.docx»."""
    from docx import Document

    from parsing.chunker import chunk_document, document_title

    titled, plain = tmp_path / "evidence" / "a.docx", tmp_path / "b.docx"
    titled.parent.mkdir()
    for path, title in ((titled, "Schoenfeld et al., 2021 (Sports). Loading recommendations"), (plain, "")):
        doc = Document()
        doc.core_properties.title = title
        doc.add_paragraph("Moderate loads of about 8–12 repetitions are a practical choice for hypertrophy. " * 3)
        doc.save(path)
    assert document_title(titled) == "Schoenfeld et al., 2021 (Sports). Loading recommendations"
    assert document_title(plain) == "b.docx"
    chunks = chunk_document(titled, corpus_root=tmp_path)
    assert chunks and all(c["title"].startswith("Schoenfeld") and c["source"] == "evidence/a.docx" for c in chunks)
