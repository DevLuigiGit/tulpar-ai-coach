"""Open-access articles for the knowledge base: Europe PMC full text (JATS XML) → DOCX in corpus/evidence/.

    python tools/fetch_open_articles.py            # fetch every article of ARTICLES, write corpus/evidence/*.docx
    python tools/fetch_open_articles.py --list     # what would be fetched, with licences

Why these: the questions the simulated clients asked and the knowledge base could not answer (docs/user-simulation.md,
finding 4) — sets and repetitions, rest between sets, a weight plateau, eating late, daily steps, breaks from sitting.
Every article is CC BY 4.0 except the steps review (CC BY-NC 4.0: fine for this non-commercial project, like the WHO
PDF; a commercial Tulpar would need another source).

What goes in: title, abstract and every top-level section except methods and results (narrative reviews name their
sections by topic, «Hypertrophy», so a whitelist would drop the substance). Tables, figures, citation markers and the
reference list are left out — statistics, not advice, and noise for retrieval. The DOCX says so on its first line, as
CC BY requires.
The index picks new documents up on the next start (Index.add_missing_documents), the title comes from the DOCX.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "corpus" / "evidence"
API = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

ARTICLES = [
    {"slug": "schoenfeld_2021_repetition_continuum", "short": "Schoenfeld et al., 2021 (Sports)", "pmcid": "PMC7927075", "doi": "10.3390/sports9020032",
     "cite": "Schoenfeld BJ, Grgic J, Van Every DW, Plotkin DL. Sports. 2021;9(2):32", "license": "CC BY 4.0",
     "topic": "подходы и повторения для силы и роста мышц"},
    {"slug": "singer_2024_rest_intervals", "short": "Singer et al., 2024 (Frontiers in Sports and Active Living)", "pmcid": "PMC11349676", "doi": "10.3389/fspor.2024.1429789",
     "cite": "Singer A, Wolf M, Generoso L, et al. Front Sports Act Living. 2024;6:1429789", "license": "CC BY 4.0",
     "topic": "отдых между подходами"},
    {"slug": "melby_2017_weight_regain", "short": "Melby et al., 2017 (Nutrients)", "pmcid": "PMC5452198", "doi": "10.3390/nu9050468",
     "cite": "Melby CL, Paris HL, Foright RM, Peth J. Nutrients. 2017;9(5):468", "license": "CC BY 4.0",
     "topic": "плато и возврат веса после похудения"},
    {"slug": "lopez_minguez_2019_meal_timing", "short": "Lopez-Minguez et al., 2019 (Nutrients)", "pmcid": "PMC6893547", "doi": "10.3390/nu11112624",
     "cite": "Lopez-Minguez J, Gómez-Abellán P, Garaulet M. Nutrients. 2019;11(11):2624", "license": "CC BY 4.0",
     "topic": "время приёмов пищи и поздний ужин"},
    {"slug": "taylor_2021_sitting_breaks", "short": "Taylor et al., 2021 (PLoS One)", "pmcid": "PMC7781669", "doi": "10.1371/journal.pone.0244841",
     "cite": "Taylor FC, Dunstan DW, Fletcher E, et al. PLoS One. 2021;16(1):e0244841", "license": "CC BY 4.0",
     "topic": "перерывы в сидячей работе"},
    {"slug": "xu_2024_daily_steps_umbrella", "short": "Xu et al., 2024 (BMJ Open)", "pmcid": "PMC11474941", "doi": "10.1136/bmjopen-2024-088524",
     "cite": "Xu et al. BMJ Open. 2024", "license": "CC BY-NC 4.0", "topic": "шаги в день и здоровье"},
]

SKIP = ("method", "result", "statist", "search strateg", "material", "supplement", "funding", "conflict", "author",
        "data avail", "abbreviation", "acknowledg", "review board", "consent", "footnote", "reference", "supporting",
        "ethic", "contributor", "competing", "provenance", "patient and public", "associated data", "publisher")


def _text(el: ET.Element | None) -> str:
    """Paragraph text without citation markers («[12]», «Smith et al. [3]» keeps the words, drops the numbers)."""
    if el is None:
        return ""
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        if node.tag in ("table-wrap", "fig", "disp-formula", "fn"):
            return
        if node.tag == "xref" and node.get("ref-type") == "bibr":
            parts.append(node.tail or "")
            return
        parts.append(node.text or "")
        for child in node:
            walk(child)
        parts.append(node.tail or "")

    walk(el)
    parts[-1] = ""  # the element's own tail belongs to its parent
    return " ".join("".join(parts).split()).replace(" ,", ",").replace(" .", ".").replace("( )", "").replace("[ ]", "")


def _wanted(title: str) -> bool:
    """Every top-level section but the statistics: narrative reviews name theirs by topic («Hypertrophy»)."""
    t = title.lower()
    return not any(k in t for k in SKIP)


def to_docx(xml: bytes, meta: dict, path: Path) -> dict:
    root = ET.fromstring(xml)
    title = _text(root.find(".//article-meta/title-group/article-title"))
    doc = Document()
    full = f"{meta['short']}. {title}"
    doc.core_properties.title = full if len(full) <= 250 else full[:249].rstrip() + "…"  # the DOCX limit is 255
    doc.add_heading(title, level=1)
    doc.add_paragraph(f"Источник: {meta['cite']}. DOI: {meta['doi']}. Лицензия: {meta['license']}. Текст из Europe "
                      f"PMC ({meta['pmcid']}) без изменений, кроме удалённых разделов о методах и результатах, "
                      "таблиц, рисунков, ссылок на литературу и списка литературы.")
    kept = []
    abstract = root.find(".//article-meta/abstract")
    if abstract is not None:
        doc.add_heading("Abstract", level=2)
        for p in abstract.iter("p"):
            doc.add_paragraph(_text(p))
        kept.append("Abstract")

    def section(sec: ET.Element, level: int) -> None:
        heading = _text(sec.find("title"))
        if heading and not _wanted(heading):
            return
        if heading:
            doc.add_heading(heading, level=min(level, 3))
            kept.append(heading)
        for child in sec:
            if child.tag == "p":
                text = _text(child)
                if text:
                    doc.add_paragraph(text)
            elif child.tag == "sec":
                section(child, level + 1)

    body = root.find(".//body")
    for sec in (body.findall("sec") if body is not None else []):
        section(sec, 2)
    doc.save(path)
    return {"title": title, "sections": kept}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for a in ARTICLES:
            print(f"{a['pmcid']:12s} {a['license']:12s} {a['topic']:45s} {a['cite']}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for a in ARTICLES:
        req = urllib.request.Request(API.format(pmcid=a["pmcid"]), headers={"User-Agent": "tulpar-ai-coach corpus"})
        with urllib.request.urlopen(req, timeout=60) as r:
            xml = r.read()
        info = to_docx(xml, a, OUT / f"{a['slug']}.docx")
        print(f"{a['slug']}: {len(xml) // 1024} KiB XML → {', '.join(info['sections'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
