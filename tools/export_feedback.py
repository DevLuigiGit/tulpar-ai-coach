"""Data flywheel: answers the clients marked 👎 → candidates for the golden sets.

    python tools/export_feedback.py                       # data/app.sqlite → evals/golden/candidates_from_feedback.jsonl
    python tools/export_feedback.py --db path/app.sqlite --since 2026-09-01 --out /tmp/c.jsonl

Reads the service's SQLite (read-only), writes one JSON line per 👎 answer. A candidate is NOT a golden
case: a person reviews it, fills `expected_*` / `reference_answer` and moves it to qa.jsonl or
router.jsonl by hand. Phones, e-mails, IIN and known client/trainer names are masked, because the file
lands in the repository. The whole file is rewritten on every run (sorted by message id), so re-running
is safe.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulpar_ai.pii import mask  # noqa: E402

OUT = ROOT / "evals" / "golden" / "candidates_from_feedback.jsonl"

SQL = """
SELECT f.message_id, f.client_id, f.comment, f.run_id, f.source, f.updated_at, m.text AS answer, m.payload_json,
       (SELECT u.text FROM messages u WHERE u.client_id = m.client_id AND u.role = 'user' AND u.id < m.id
        ORDER BY u.id DESC LIMIT 1) AS question
FROM feedback f JOIN messages m ON m.id = f.message_id
WHERE f.rating = 'down' AND f.updated_at >= ?
ORDER BY f.message_id
"""


def _names(con: sqlite3.Connection) -> dict[str, str]:
    """Demo-mode names from this service's DB; in tulpar mode names are not stored here."""
    try:
        rows = con.execute("SELECT name, role FROM demo_users").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {name: "[тренер]" if role == "trainer" else "[клиент]" for name, role in rows if name}


def candidate(row: sqlite3.Row, names: dict[str, str]) -> dict:
    payload = json.loads(row["payload_json"] or "{}")
    intent = payload.get("intent")
    question = mask(row["question"] or "", names)
    base = {
        "id": f"fb{row['message_id']}",
        "status": "needs_review",
        # a question answered from the knowledge base → qa.jsonl; a wrong branch (escalation, refusal…) → router.jsonl
        "target": "qa" if payload.get("kind") == "answer" else "router",
        "feedback": {"rating": "down", "comment": mask(row["comment"] or "", names) or None, "source": row["source"],
                     "at": row["updated_at"], "langsmith_run_id": row["run_id"]},
        "observed": {"kind": payload.get("kind"), "intent": intent, "answer": mask(row["answer"] or "", names),
                     "citations": [{"source": c.get("source"), "title": c.get("title"), "page": c.get("page")}
                                   for c in payload.get("citations") or []]},
    }
    if base["target"] == "qa":
        return {**base, "question": question, "answerable": None, "expected_sources": [], "must_include": [],
                "reference_answer": "", "tags": ["from_feedback"]}
    return {**base, "text": question, "expected_intent": None, "red_flag": None, "tags": ["from_feedback"],
            "note": ""}


def export(db: Path, out: Path, since: str = "") -> int:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        names = _names(con)
        rows = [candidate(r, names) for r in con.execute(SQL, (since,))]
    finally:
        con.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    from tulpar_ai.config import get_settings

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, default=None, help="SQLite of the service (default: AI_DATA_DIR/app.sqlite)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--since", default="", help="only votes updated on/after this ISO date")
    args = ap.parse_args(argv)
    db = args.db or get_settings().data_path("app.sqlite")
    if not Path(db).exists():
        sys.exit(f"нет базы {db}")
    n = export(Path(db), args.out, args.since)
    print(f"{n} кандидатов → {args.out}")


if __name__ == "__main__":
    main()
