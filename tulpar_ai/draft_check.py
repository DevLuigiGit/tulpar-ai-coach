"""The summary of a program draft must not promise what its operations do not do.

Screenshot of 30.09: «В плане есть приседания со штангой и жим ногами — нужно их заменить…», while the only
operation replaced the squat. The trainer reads the summary first, so a promise without an operation is an error
like a validator error: the graph sends the draft back with this message (≤3 drafts), and if it stays, the card
shows it next to the Skill warnings.

The check is deterministic: a clause of the summary that speaks of a change («заменить», «убрать», «вместо»…) and
names an exercise of the current plan, while no operation touches that exercise. A clause that keeps something
(«оставляем», «без изменений») is not a promise.
"""

from __future__ import annotations

import re

CHANGE = re.compile(r"замен|убер|убира|удал|исключ|поменя|сократ|облегч|вместо|уменьш|увелич", re.I)
KEEP = re.compile(r"остав|сохран|не трог|без изменен|не меня|менять не", re.I)
_CLAUSES = re.compile(r"[.;!?\n]+")


def _norm(text: str | None) -> str:
    return (text or "").lower().replace("ё", "е")


def _stems(name: str) -> list[str]:
    """Word starts that survive case endings: «жим ногами» → «жим», «ногам» (finds «жиме ногами» too)."""
    out = []
    for w in re.findall(r"[а-яa-z]+", _norm(name)):
        if len(w) >= 3:
            out.append(w[:5] if len(w) > 5 else w[: max(3, len(w) - 1)])
    return out


def mentions(text: str, name: str) -> bool:
    words = re.findall(r"[а-яa-z]+", _norm(text))
    stems = _stems(name)
    return bool(stems) and all(any(w.startswith(s) for w in words) for s in stems)


def _head(name: str) -> str | None:
    stems = _stems(name)
    return stems[0] if stems else None


def summary_mismatches(plan: dict, ops: list[dict], summary: str | None) -> list[dict]:
    """E_SUMMARY_MISMATCH per exercise the summary promises to change without an operation on it.

    An exercise is named either in full («жим ногами») or by its first word when no other exercise of the plan
    starts the same way («выпады» for «Выпады с гантелями», but not «жим» when the plan also has «Жим штанги лёжа»)."""
    touched_ids = {o.get("wex_id") for o in ops if o.get("wex_id")}
    exercises = [e for d in plan.get("days", []) for e in d.get("exercises", [])]
    touched = {_norm(e.get("exercise_name")) for e in exercises if e.get("id") in touched_ids}
    heads: dict[str, set[str]] = {}
    for e in exercises:
        if _head(e.get("exercise_name") or ""):
            heads.setdefault(_head(e["exercise_name"]), set()).add(_norm(e["exercise_name"]))
    out: list[dict] = []
    for clause in _CLAUSES.split(summary or ""):
        if not CHANGE.search(clause) or KEEP.search(clause):
            continue
        words = re.findall(r"[а-яa-z]+", _norm(clause))
        for e in exercises:
            name = e.get("exercise_name") or ""
            key, head = _norm(name), _head(name)
            if not key or key in touched or any(v["name"] == key for v in out):
                continue
            by_head = head and len(heads.get(head, ())) == 1 and any(w.startswith(head) for w in words)
            if not (mentions(clause, name) or by_head):
                continue
            out.append({"code": "E_SUMMARY_MISMATCH", "severity": "error", "op_index": None, "name": key,
                        "message": f"описание обещает изменить «{name}», но операций с этим упражнением нет: "
                                   "добавь операцию или убери это из описания"})
    for v in out:
        v.pop("name")
    return out
