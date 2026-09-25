"""PDF backends compared on the extracted text itself, page by page, with no embedder and no network.

Retrieval evals cannot tell the backends apart. With the local n-gram embedder the Russian questions never
reach the English WHO PDF (0 of 12 WHO questions hit on every backend), so those runs do not depend on
the PDF text at all. With Jina the backends differ by about one question out of 12, which is noise.
This script runs every installed backend through the same cleaner and splitter as the index and reports:

  pages_with_text / words / chunks  how much text survives normalisation and the short-fragment filter
  unconfirmed_share                 share of a backend's words that no other backend has on the same page:
                                    glued words ("physicalactivity"), split words ("physi cal"), stray glyphs
  order_similarity_vs_pdfium        difflib ratio of word sequences against pdfium (the production
                                    backend), weighted by page length: reading order and dropped text
  extra_vs_pdfium / _on_neighbour   words a backend has beyond pdfium on a page, and how many of them pdfium
                                    finds on the previous or next page (text taken from a facing page)

Usage: .venv/bin/python evals/parsing_backends.py [--pdf corpus/who_2020_physical_activity.pdf]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsing.chunker import MIN_PDF_CHUNK_CHARS, split_text  # noqa: E402
from parsing.cleaner import normalize_text  # noqa: E402
from parsing.parser import _pdf_pages  # noqa: E402

REFERENCE = "pdfium"
_WORD = re.compile(r"[a-z0-9]+")


def _pymupdf_pages(path: Path) -> list[str]:
    import pymupdf  # AGPL: only for this comparison, not a runtime dependency

    with pymupdf.open(path) as document:
        return [page.get_text() for page in document]


def _pypdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(path).pages]


BACKENDS = {"pdfium": _pdf_pages, "pymupdf": _pymupdf_pages, "pypdf": _pypdf_pages}


def page_words(text: str) -> list[str]:
    return _WORD.findall(normalize_text(text).lower())


def page_chunks(text: str, size: int = 400, overlap: int = 120) -> list[str]:
    """What the index would store for this page (same splitter and short-fragment filter)."""
    return [c for c in split_text(text, size, overlap) if len(c) > MIN_PDF_CHUNK_CHARS]


def unconfirmed(pages: list[list[str]], others: list[list[list[str]]]) -> tuple[int, Counter]:
    """Word occurrences on a page that no other backend has on the same page (as many times)."""
    total, examples = 0, Counter()
    for i, words in enumerate(pages):
        seen = Counter()
        for other in others:
            if i < len(other):
                seen |= Counter(other[i])
        extra = Counter(words) - seen
        total += sum(extra.values())
        examples.update(extra)
    return total, examples


def extra_on_neighbour(reference: list[list[str]], pages: list[list[str]]) -> tuple[int, int]:
    """Words beyond the reference on each page, and how many of those the reference has one page away."""
    extra_total = neighbour_total = 0
    for i, words in enumerate(pages):
        extra = Counter(words) - Counter(reference[i] if i < len(reference) else [])
        nearby = Counter()
        for j in (i - 1, i + 1):
            if 0 <= j < len(reference):
                nearby |= Counter(reference[j])
        extra_total += sum(extra.values())
        neighbour_total += sum((extra & nearby).values())
    return extra_total, neighbour_total


def order_similarity(a: list[list[str]], b: list[list[str]]) -> float:
    matched = weight = 0.0
    for i in range(max(len(a), len(b))):
        x, y = (a[i] if i < len(a) else []), (b[i] if i < len(b) else [])
        if not x and not y:
            continue
        size = len(x) + len(y)
        matched += SequenceMatcher(None, x, y, autojunk=False).ratio() * size
        weight += size
    return round(matched / weight, 4) if weight else 1.0


def compare(raw: dict[str, list[str]]) -> dict[str, dict]:
    words = {name: [page_words(p) for p in pages] for name, pages in raw.items()}
    report = {}
    for name, pages in raw.items():
        total_words = sum(len(w) for w in words[name])
        extra, examples = unconfirmed(words[name], [w for other, w in words.items() if other != name])
        report[name] = {
            "pages": len(pages),
            "pages_with_text": sum(1 for p in pages if normalize_text(p)),
            "words": total_words,
            "chunks": sum(len(page_chunks(p)) for p in pages),
            "unconfirmed_words": extra,
            "unconfirmed_share": round(extra / total_words, 4) if total_words else 0.0,
            "unconfirmed_examples": [w for w, _ in examples.most_common(12)],
        }
        if REFERENCE in words and name != REFERENCE:
            report[name]["order_similarity_vs_pdfium"] = order_similarity(words[REFERENCE], words[name])
            extra_total, neighbour = extra_on_neighbour(words[REFERENCE], words[name])
            report[name]["extra_vs_pdfium"] = extra_total
            report[name]["extra_vs_pdfium_on_neighbour_page"] = neighbour
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=ROOT / "corpus" / "who_2020_physical_activity.pdf")
    ap.add_argument("--out", type=Path, default=ROOT / "evals" / "results" / "parsing_backends_who.json")
    args = ap.parse_args()
    raw, skipped = {}, {}
    for name, extract in BACKENDS.items():
        try:
            raw[name] = extract(args.pdf)
        except ImportError as e:
            skipped[name] = str(e)
    result = {"pdf": args.pdf.name, "backends": compare(raw), "skipped": skipped}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("| backend | pages with text | words | chunks | unconfirmed words | order vs pdfium "
          "| extra vs pdfium (on neighbour page) |")
    print("|---|---|---|---|---|---|---|")
    for name, r in result["backends"].items():
        order = r.get("order_similarity_vs_pdfium", "—")
        extra = f"{r['extra_vs_pdfium']} ({r['extra_vs_pdfium_on_neighbour_page']})" if "extra_vs_pdfium" in r else "—"
        print(f"| {name} | {r['pages_with_text']}/{r['pages']} | {r['words']} | {r['chunks']} "
              f"| {r['unconfirmed_words']} ({r['unconfirmed_share']:.2%}) | {order} | {extra} |")
    for name, r in result["backends"].items():
        print(f"{name}: {', '.join(r['unconfirmed_examples'])}")
    if skipped:
        print("skipped:", skipped)


if __name__ == "__main__":
    main()
