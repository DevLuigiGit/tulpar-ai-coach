"""Check evals/golden/qa.jsonl against the corpus: JSON per line, unique ids, exact exercise titles,
must_include tokens present in the reference answer and in the cited card / nutrition.md, and for every
cited WHO page the English phrases that carry the fact. Run: .venv/bin/python evals/check_qa_golden.py"""
import json, re, sys
from collections import Counter
from pypdf import PdfReader

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
cards = {}
for l in open(f"{ROOT}/corpus/exercises.jsonl", encoding="utf-8"):
    d = json.loads(l); cards[d["title"]] = d["text"].lower()
nutrition = open(f"{ROOT}/corpus/nutrition.md", encoding="utf-8").read().lower()
pdf = PdfReader(f"{ROOT}/corpus/who_2020_physical_activity.pdf")

def norm(s):
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"\s*([–-])\s*", r"\1", s).lower()

WHO = {  # qid -> {page: [english phrases that must be on that page]}
 "q40": {12: ["150–300 minutes of moderate-intensity", "more than 300 minutes"], 42: ["150–300 minutes of moderate-intensity", "more than 300 minutes"]},
 "q41": {12: ["75–150 minutes of vigorous-intensity", "more than 150 minutes of vigorous"], 42: ["75–150 minutes of vigorous-intensity", "more than 150 minutes of vigorous"]},
 "q42": {12: ["involve all major muscle groups on 2 or more days a week"], 42: ["involve all major muscle groups on 2 or more days a week"]},
 "q43": {14: ["functional balance and strength training at moderate or greater intensity, on 3 or more days a week", "prevent falls", "2 or more days"], 53: ["functional balance and strength training at moderate or greater intensity, on 3 or more days a week", "prevent falls", "2 or more days"]},
 "q44": {11: ["average of 60 minutes per day", "at least 3 days a week", "recreational screen time"], 35: ["average of 60 minutes per day", "at least 3 days a week"]},
 "q45": {12: ["adults should limit the amount of time spent being sedentary", "any intensity (including light intensity)", "more than the recommended levels"], 48: ["adults should limit the amount of time spent being sedentary"]},
 "q46": {51: ["60–75 minutes per day", "more than 8 hours per day", "more than 1 million"]},
 "q47": {16: ["at least 150 minutes of moderate-intensity aerobic physical activity throughout the week", "gentle stretching", "habitually engaged in vigorous"], 57: ["at least 150 minutes of moderate-intensity aerobic physical activity throughout the week", "gentle stretching", "habitually engaged in vigorous"]},
 "q48": {16: ["supine position after the first trimester", "excessive heat", "physical contact", "stay hydrated"], 58: ["supine position after the first trimester", "excessive heat", "physical contact", "stay hydrated"]},
 "q49": {18: ["150–300 minutes of moderate-intensity", "75–150 minutes of vigorous", "type-2 diabetes", "2 or more days", "start by doing small amounts"], 62: ["150–300 minutes of moderate-intensity", "75–150 minutes of vigorous", "type-2 diabetes"]},
 "q50": {46: ["bouts of least 10 minutes duration has been removed", "any duration"]},
 "q51": {9: ["between 3 and less than 6 times", "5 or 6 on a scale of 0–10", "6.0 or more mets", "7 or 8 on a scale of 0–10"]},
}

errs = []
rows = []
for i, line in enumerate(open(f"{ROOT}/evals/golden/qa.jsonl", encoding="utf-8"), 1):
    try:
        rows.append(json.loads(line))
    except Exception as e:
        errs.append(f"line {i}: bad JSON {e}")
ids = [r["id"] for r in rows]
dups = [k for k, v in Counter(ids).items() if v > 1]
if dups: errs.append(f"duplicate ids {dups}")
keys = {"id", "question", "answerable", "expected_sources", "must_include", "reference_answer", "tags"}
for r in rows:
    q = r["id"]
    if set(r) != keys: errs.append(f"{q}: keys {set(r) ^ keys}")
    if not r["answerable"]:
        if r["expected_sources"] or r["must_include"]: errs.append(f"{q}: unanswerable must have empty sources/tokens")
        if "тренер" not in r["reference_answer"]: errs.append(f"{q}: unanswerable ref must hand off to trainer")
        continue
    if not (1 <= len(r["must_include"]) <= 3): errs.append(f"{q}: must_include size")
    ref = r["reference_answer"].lower().replace("–", "-")
    for t in r["must_include"]:
        if t != t.lower(): errs.append(f"{q}: token not lowercase {t}")
        if t.lower() not in ref: errs.append(f"{q}: token {t!r} missing from reference_answer")
    srcs = r["expected_sources"]
    if not srcs: errs.append(f"{q}: no sources")
    texts = []  # Russian source texts the tokens must come from
    for s in srcs:
        if s["source"] == "exercises":
            if s["title"] not in cards: errs.append(f"{q}: no exercise titled {s['title']!r}")
            else: texts.append(cards[s["title"]])
        elif s["source"] == "nutrition":
            texts.append(nutrition)
        elif s["source"] == "who2020":
            page = s["page"]
            if not 1 <= page <= len(pdf.pages): errs.append(f"{q}: who2020 page {page} out of range"); continue
            text = norm(pdf.pages[page - 1].extract_text() or "")
            for ph in WHO.get(q, {}).get(page, []):
                if norm(ph) not in text: errs.append(f"{q}: phrase {ph!r} not on who2020 p.{page}")
            if page not in WHO.get(q, {}): errs.append(f"{q}: who2020 p.{page} has no phrases in WHO map")
        else:
            errs.append(f"{q}: unknown source {s['source']!r}")
    for t in r["must_include"]:
        if texts and not any(t.lower() in x for x in texts): errs.append(f"{q}: token {t!r} not in any cited Russian source")
for q in WHO:
    if q not in ids: errs.append(f"WHO map has {q} not in file")
tags = Counter(t for r in rows for t in r["tags"])
print("rows", len(rows), "unique ids", len(set(ids)))
print("tags", dict(sorted(tags.items(), key=lambda x: -x[1])))
print("answerable", sum(r["answerable"] for r in rows), "unanswerable", sum(not r["answerable"] for r in rows))
print("ERRORS:" if errs else "OK", *errs, sep="\n")
sys.exit(1 if errs else 0)
