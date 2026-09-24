"""Rewrite the results block of EVALS.md from evals/results/*.json.

    python evals/report.py
Only the text between <!-- results:start --> and <!-- results:end --> is replaced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
from run import KEYS, table  # noqa: E402

RES = ROOT / "evals" / "results"
DOC = ROOT / "EVALS.md"
ORDER = [("program", "Валидатор плана (Skill)"), ("router", "Маршрутизатор"), ("qa", "RAG: вопросы и ответы"),
         ("draft", "Черновики изменений программы"), ("vision", "Фото еды")]
EXPS = [("exp_rag_rerank", "qa", "A/B №1: реранкер включён или выключен"),
        ("exp_chunk_size", "qa", "Размер фрагмента PDF, только поиск"),
        ("exp_answer_temperature", "qa", "Temperature узла ответа, 3 повтора"),
        ("exp_answer_top_p", "qa", "top_p узла ответа"),
        ("exp_route_temperature", "router", "Temperature маршрутизатора, 3 повтора")]


def block() -> str:
    out = []
    for name, title in ORDER:
        f = RES / f"{name}.json"
        if f.exists():
            s = json.loads(f.read_text(encoding="utf-8"))["summary"]
            out.append(f"### {title}\n\n{table(name, [('текущая конфигурация', s)])}\n")
    for name, suite, title in EXPS:
        f = RES / f"{name}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            arms = [(", ".join(f"{k}={v}" for k, v in a["arm"].items()), a["summary"]) for a in d["arms"]]
            out.append(f"### {title}\n\n{table(suite, arms)}\n")
    for f in sorted(RES.glob("router_*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))["summary"]
        out.append(f"### Маршрутизатор: {f.stem.removeprefix('router_')}\n\n{table('router', [(f.stem, s)])}\n")
    for f in sorted(RES.glob("vision_*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))["summary"]
        out.append(f"### Фото еды: {f.stem.removeprefix('vision_')}\n\n{table('vision', [(f.stem, s)])}\n")
    return "\n".join(out) or "_Результатов пока нет: запустите прогоны из раздела «Как запустить»._\n"


def main() -> None:
    doc = DOC.read_text(encoding="utf-8")
    a, b = doc.index("<!-- results:start -->"), doc.index("<!-- results:end -->")
    DOC.write_text(doc[: a + len("<!-- results:start -->")] + "\n\n" + block() + "\n" + doc[b:], encoding="utf-8")
    print("EVALS.md updated")


if __name__ == "__main__":
    main()
