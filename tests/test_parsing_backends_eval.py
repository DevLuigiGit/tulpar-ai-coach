"""The PDF backend comparison must separate backends by their text, not by what retrieval happens to find."""

import importlib.util
import json
import sys
from pathlib import Path

from parsing.tests.pdfgen import write_pdf

ROOT = Path(__file__).resolve().parents[1]
LONG = "Adults should do at least 150 to 300 minutes of moderate-intensity aerobic physical activity per week."


def _module():
    spec = importlib.util.spec_from_file_location("evals_parsing_backends", ROOT / "evals" / "parsing_backends.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evals_parsing_backends"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_glued_and_off_page_words_are_unconfirmed():
    pb = _module()
    report = pb.compare({
        "pdfium": [LONG, "Page two text."],
        "pymupdf": [LONG, "Page two text."],
        "pypdf": [LONG.replace("physical activity", "physicalactivity"), "Page two text. Recommendations viii"],
    })
    assert report["pdfium"]["unconfirmed_words"] == 0
    assert report["pymupdf"]["order_similarity_vs_pdfium"] == 1.0
    assert set(report["pypdf"]["unconfirmed_examples"]) == {"physicalactivity", "recommendations", "viii"}
    assert report["pypdf"]["order_similarity_vs_pdfium"] < 1.0


def test_text_from_a_facing_page_is_found_on_the_neighbour_page():
    pb = _module()
    reference = [["intro"], ["at", "a", "glance"], ["adults", "150"]]
    spread = [["intro"], ["at", "a", "glance", "adults", "150", "stray"], ["adults", "150"]]
    assert pb.extra_on_neighbour(reference, spread) == (3, 2)


def test_same_word_twice_on_a_page_counts_once_as_extra():
    pb = _module()
    total, examples = pb.unconfirmed([["and", "and", "and"]], [[["and"]], [["and", "and"]]])
    assert total == 1 and examples == {"and": 1}


def test_main_writes_report_and_lists_missing_backends(tmp_path, monkeypatch, capsys):
    pb = _module()
    pdf = write_pdf(tmp_path / "doc.pdf", [f"{LONG}\n{LONG}", ""])

    def missing(path):
        raise ImportError("not installed")

    monkeypatch.setattr(pb, "BACKENDS", {"pdfium": pb.BACKENDS["pdfium"], "pymupdf": missing})
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["parsing_backends.py", "--pdf", str(pdf), "--out", str(out)])
    pb.main()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["skipped"] == {"pymupdf": "not installed"}
    pdfium = report["backends"]["pdfium"]
    assert pdfium["pages"] == 2 and pdfium["pages_with_text"] == 1 and pdfium["chunks"] >= 1
    assert "| pdfium | 1/2 |" in capsys.readouterr().out
