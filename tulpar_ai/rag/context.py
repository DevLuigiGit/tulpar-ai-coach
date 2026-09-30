"""Contextual chunks (RAG_CONTEXT_HEADERS=on): a short deterministic header in front of the text that is EMBEDDED.

A 400-character WHO fragment often says «should do at least 150–300 minutes» without saying who: the population is
in the page heading. The header puts it back for the embedder (and BM25): «ВОЗ 2020 · Взрослые 18–64 лет (adults)
· стр. 42». The answer model still sees the original chunk text — the header is never stored in the payload's text.

No LLM is involved: headers come from the chunk's own metadata and, for PDFs, from the page's heading lines, so the
same corpus always gives the same headers (and the same cached embeddings).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from parsing.chunker import WHO_SOURCE
from parsing.parser import parse_document

HEADER_VERSION = 1  # bump when the header text changes: it is part of the index collection name

EQUIPMENT_RU = {"barbell": "штанга", "dumbbell": "гантели", "cable": "блочный тренажёр", "machine": "тренажёр",
                "bodyweight": "свой вес", "kettlebell": "гиря"}

# WHO 2020 sections, most specific first: «ADULTS AND OLDER ADULTS WITH CHRONIC CONDITIONS» is the chronic
# section, not the older adults one. Labels are bilingual: the questions are Russian, the page is English.
WHO_SECTIONS: list[tuple[str, str]] = [
    (r"DISABILITY", "Дети, подростки и взрослые с инвалидностью (living with disability)"),
    (r"CHRONIC", "Взрослые и пожилые с хроническими заболеваниями (adults with chronic conditions)"),
    (r"PREGNANT|POSTPARTUM", "Беременные и недавно родившие женщины (pregnant and postpartum women)"),
    (r"OLDER ADULTS", "Пожилые люди 65+ (older adults)"),
    (r"CHILDREN|ADOLESCENTS", "Дети и подростки 5–17 лет (children and adolescents)"),
    (r"\bADULTS\b", "Взрослые 18–64 лет (adults)"),
    (r"GLOSSARY", "Глоссарий терминов (glossary)"),
    (r"EXECUTIVE SUMMARY", "Краткое изложение рекомендаций (executive summary)"),
    (r"BACKGROUND|RATIONALE|TARGET AUDIENCE|SCOPE OF|METHODS|THE EVIDENCE", "Введение и методы (background, methods)"),
    (r"CERTAINTY OF EVIDENCE|BENEFITS AND HARMS|RESEARCH NEEDS|DISSEMINATION|IMPLEMENTATION|SURVEILLANCE",
     "Внедрение и пробелы в данных (implementation, research needs)"),
    (r"^REFERENCES", "Список литературы (references)"),
    (r"MANAGEMENT OF GUIDELINE|DECLARATION OF INTEREST|ANNEX", "Приложения (annexes)"),
    (r"CONTENTS|ACKNOWLEDGEMENTS|ABBREVIATIONS|WHO GUIDELINES ON", "Служебные страницы (front matter)"),
    (r"^RECOMMENDATIONS$", "Рекомендации (recommendations)"),
]
# Summary pages with no heading of their own open with the population in plain case: «All older adults should…».
WHO_CUES: list[tuple[str, str]] = [
    (r"living with disability", WHO_SECTIONS[0][1]),
    (r"chronic conditions", WHO_SECTIONS[1][1]),
    (r"pregnan|postpartum", WHO_SECTIONS[2][1]),
    (r"older adults", WHO_SECTIONS[3][1]),
    (r"children and adolescents should", WHO_SECTIONS[4][1]),
    (r"\badults should|all adults", WHO_SECTIONS[5][1]),
]
SEDENTARY = "сидячий образ жизни (sedentary behaviour)"
POPULATIONS = {label for _, label in WHO_SECTIONS[:6]}
_TITLE = re.compile(r"(who )?guidelines on\s+physical activity and\s+sedentary behaviour")  # title and page footer

_HEADING = re.compile(r"^[A-Z][A-Z ,\-–]{4,}")  # a line opening with an upper-case run of 5+ characters


def heading_lines(page_text: str) -> list[str]:
    lines = [" ".join(line.replace("\xa0", " ").split()) for line in (page_text or "").splitlines()]
    return [line for line in lines if _HEADING.match(line) and sum(c.isalpha() for c in _HEADING.match(line).group()) >= 5]


def who_page_section(page_text: str, previous: str | None) -> tuple[str | None, bool]:
    """(section label, is the page about sedentary behaviour) for one WHO page; unknown → the previous page's section.

    Upper-case headings decide first (a heading may wrap: «CHILDREN AND ADOLESCENTS (aged 5–17 years) AND ADULTS /
    (aged 18 years and older) LIVING WITH DISABILITY», so the first lines count too); a summary page that opens with
    «It is recommended that: All older adults…» is read by its opening words; otherwise the section carries over.
    """
    lines = [" ".join(line.replace("\xa0", " ").split()) for line in (page_text or "").splitlines() if line.strip()]
    joined = "\n".join(heading_lines(page_text) + lines[:3])
    section = next((label for rx, label in WHO_SECTIONS if re.search(rx, joined, re.M)), None)
    start = _TITLE.sub(" ", " ".join(lines)[:400].lower())
    if section is None and re.match(r"\s*(it is recommended that|all |adults |older adults |children )", start):
        section = next((label for rx, label in WHO_CUES if re.search(rx, start)), None)
    section = section or previous
    population = section in POPULATIONS
    sedentary = population and (bool(re.search(r"^SEDENTARY BEHAVIOUR", joined, re.M)) or "sedentary" in start[:300])
    return section, sedentary


def generic_page_section(page_text: str, previous: str | None) -> tuple[str | None, bool]:
    heads = heading_lines(page_text)
    return (heads[0][:80].strip(" ,-–").capitalize() if heads else previous), False


@lru_cache(maxsize=32)
def _page_sections(path: str, mtime: float, who: bool) -> dict[int, tuple[str | None, bool]]:
    out: dict[int, tuple[str | None, bool]] = {}
    previous: str | None = None
    guess = who_page_section if who else generic_page_section
    for block in parse_document(Path(path)):
        if block.get("page") is None:
            continue
        section, sedentary = guess(block["text"], previous)
        out[block["page"]] = (section, sedentary)
        previous = section
    return out


def page_sections(path: Path, source: str) -> dict[int, tuple[str | None, bool]]:
    """Section guessed for every page of a PDF (heading lines, carried over to pages without one)."""
    try:
        return _page_sections(str(path), path.stat().st_mtime, source == WHO_SOURCE)
    except Exception:  # an unreadable document gets page-only headers, never a failed build
        return {}


def header(chunk, sections: dict[int, tuple[str | None, bool]] | None = None) -> str:
    """The context line for one chunk (an index.Chunk)."""
    if chunk.source == "exercises":
        parts = ["Упражнение", chunk.muscle_group, EQUIPMENT_RU.get(chunk.equipment or "", chunk.equipment)]
    elif chunk.source == "nutrition":
        section = chunk.title.split("—", 1)[-1].strip() if "—" in chunk.title else chunk.title
        parts = ["Правила питания Tulpar", section]
    else:
        section, sedentary = (sections or {}).get(chunk.page or -1, (None, False))
        name = "ВОЗ 2020 (WHO 2020)" if chunk.source == WHO_SOURCE else chunk.title
        parts = [name, section, SEDENTARY if sedentary else None, f"стр. {chunk.page}" if chunk.page else None]
    return " · ".join(p for p in parts if p)


def with_header(chunk, sections: dict[int, tuple[str | None, bool]] | None = None) -> str:
    return f"{header(chunk, sections)}\n{chunk.text}"
