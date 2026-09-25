#!/usr/bin/env python3
"""Validate a proposed change to a Tulpar training plan. Standard library only.

Input JSON: {"plan": <Plan>, "ops": [<PlanOp>...], "client": {"level", "place", "trainer_note": {"injury", "body"}}}
  Plan   = {"id", "title", "days": [{"day_index", "title", "exercises": [{"id", "exercise_id", "exercise_name",
            "target_sets", "target_reps"}]}]}
  PlanOp = {"op": "replace_exercise"|"set_volume"|"remove_exercise"|"add_exercise", "day_index": int,
            "wex_id"?: str, "exercise_id"?: str, "sets"?: int, "reps"?: int, "reason"?: str}

Usage:
  python validate_plan.py proposal.json      # or "-" for stdin
Exit code: 0 no errors (warnings allowed), 1 errors found, 2 unreadable input.
The catalogue is read from ../references/exercises_catalog.json next to this script.
"""

from __future__ import annotations

import copy
import json
import sys
import uuid
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent.parent / "references" / "exercises_catalog.json"

# Mirrors Tulpar: backend/app/generator.py LEVEL_VOLUME and backend/app/fitness.py PLACE_EQUIPMENT.
LEVEL_VOLUME = {
    "beginner": {"sets": 3, "reps": 11, "ex_per_day": 4},
    "inter": {"sets": 4, "reps": 9, "ex_per_day": 5},
    "adv": {"sets": 4, "reps": 7, "ex_per_day": 6},
}
PLACE_EQUIPMENT = {
    "gym": {"bodyweight", "dumbbell", "barbell", "machine", "cable", "band", "kettlebell", "other"},
    "home": {"bodyweight", "band", "dumbbell", "kettlebell"},
    "other": {"bodyweight", "dumbbell", "barbell", "band", "kettlebell"},
}
# Body part → word stems as they appear in Russian notes and contraindication texts.
BODY_PARTS = {
    "колени": ["колен"],
    "спина": ["спин", "поясниц", "позвон", "грыж", "протрузи"],
    "плечи": ["плеч"],
    "запястья": ["запяст", "кист"],
    "локти": ["локт"],
    "шея": ["шея", "шеи", "шею", "шейн"],
    "сердце": ["сердц", "давлен", "гипертон"],
    "тазобедренный сустав": ["тазобедр"],
    "голеностоп": ["голеностоп", "лодыж"],
}
REPS_MEASURES = {"weight_reps", "reps", None}
# The catalogue labels core work two ways; a swap between them is still the same muscle group.
GROUP_ALIASES = {"кор": "пресс"}
# Kind of work: swapping strength for cardio or stretching is never a like-for-like replacement.
NON_STRENGTH = {"кардио", "растяжка"}


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, dict]:
    return {e["id"]: e for e in json.loads(path.read_text(encoding="utf-8"))}


def body_parts(text: str | None) -> set[str]:
    t = (text or "").lower()
    return {part for part, stems in BODY_PARTS.items() if any(s in t for s in stems)}


def injured_parts(client: dict) -> set[str]:
    note = client.get("trainer_note") or {}
    parts = body_parts(note.get("body"))
    for extra in client.get("restrictions") or []:
        parts |= body_parts(extra)
    return parts


def muscle_group(item: dict | None, catalog: dict[str, dict]) -> str | None:
    """Group of a plan exercise or a catalogue entry; the catalogue wins over what the plan row says."""
    if not item:
        return None
    ex = catalog.get(item.get("exercise_id") or item.get("id") or "") or item
    g = ex.get("muscle_group") or item.get("muscle_group")
    return GROUP_ALIASES.get(g, g)


def _find(plan: dict, day_index: int, wex_id: str | None):
    day = next((d for d in plan.get("days", []) if d.get("day_index") == day_index), None)
    if day is None:
        return None, None
    wex = next((e for e in day.get("exercises", []) if e.get("id") == wex_id), None) if wex_id else None
    return day, wex


def apply_ops(plan: dict, ops: list[dict], catalog: dict[str, dict]) -> tuple[dict, list[dict]]:
    """Apply ops to a COPY of the plan. Returns (new_plan, reference errors)."""
    new = copy.deepcopy(plan)
    errors: list[dict] = []
    for i, op in enumerate(ops):
        kind = op.get("op")
        day, wex = _find(new, op.get("day_index"), op.get("wex_id"))
        if day is None:
            errors.append(err("E_BAD_REF", f"нет дня с day_index={op.get('day_index')}", i))
            continue
        if kind in ("replace_exercise", "set_volume", "remove_exercise") and wex is None:
            errors.append(err("E_BAD_REF", f"в дне «{day.get('title')}» нет упражнения wex_id={op.get('wex_id')}", i))
            continue
        if kind in ("replace_exercise", "add_exercise"):
            ex = catalog.get(op.get("exercise_id") or "")
            if ex is None:
                errors.append(err("E_UNKNOWN_EXERCISE", f"упражнения exercise_id={op.get('exercise_id')} нет в каталоге", i))
                continue
        if kind == "replace_exercise":
            wex.update({"exercise_id": ex["id"], "exercise_name": ex["name"], "muscle_group": ex.get("muscle_group"),
                        "equipment": ex.get("equipment")})
            if op.get("sets"):
                wex["target_sets"] = op["sets"]
            if op.get("reps"):
                wex["target_reps"] = op["reps"]
        elif kind == "set_volume":
            if not op.get("sets") and not op.get("reps"):
                errors.append(err("E_OP_FIELDS", "set_volume без sets и reps", i))
                continue
            if op.get("sets"):
                wex["target_sets"] = op["sets"]
            if op.get("reps"):
                wex["target_reps"] = op["reps"]
        elif kind == "remove_exercise":
            day["exercises"] = [e for e in day["exercises"] if e.get("id") != wex["id"]]
        elif kind == "add_exercise":
            day["exercises"].append({
                "id": op.get("wex_id") or str(uuid.uuid4()), "exercise_id": ex["id"], "exercise_name": ex["name"],
                "muscle_group": ex.get("muscle_group"), "equipment": ex.get("equipment"),
                "target_sets": op.get("sets") or 3, "target_reps": op.get("reps") or 10,
            })
        else:
            errors.append(err("E_OP_FIELDS", f"неизвестная операция {kind!r}", i))
    return new, errors


def err(code: str, message: str, op_index: int | None = None) -> dict:
    return {"code": code, "severity": "error" if code.startswith("E_") else "warning", "message": message, "op_index": op_index}


def validate(plan: dict, ops: list[dict], client: dict, catalog: dict[str, dict] | None = None) -> list[dict]:
    catalog = catalog if catalog is not None else load_catalog()
    out: list[dict] = []
    if not ops:
        return [err("E_EMPTY_OPS", "в предложении нет ни одной операции")]
    new, ref_errors = apply_ops(plan, ops, catalog)
    out.extend(ref_errors)

    level = client.get("level") or "beginner"
    vol = LEVEL_VOLUME.get(level, LEVEL_VOLUME["beginner"])
    allowed = PLACE_EQUIPMENT.get(client.get("place") or "")
    hurt = injured_parts(client)

    touched = {(op.get("day_index"), op.get("exercise_id")) for op in ops if op.get("op") in ("replace_exercise", "add_exercise")}
    for i, op in enumerate(ops):
        ex = catalog.get(op.get("exercise_id") or "")
        if op.get("op") in ("replace_exercise", "add_exercise") and ex:
            if allowed is not None and ex.get("equipment") and ex["equipment"] not in allowed:
                out.append(err("E_EQUIPMENT", f"«{ex['name']}» требует {ex['equipment']}, а клиент тренируется: {client.get('place')}", i))
            clash = hurt & body_parts(ex.get("contraindications"))
            if clash:
                out.append(err("E_CONTRAINDICATION",
                               f"«{ex['name']}» противопоказано при проблемах: {', '.join(sorted(clash))} (заметка тренера)", i))
        if op.get("op") == "replace_exercise" and ex:
            # A wrong exercise_id is the typical model slip: the reason names one exercise, the id is another.
            _, old = _find(plan, op.get("day_index"), op.get("wex_id"))
            was, becomes = muscle_group(old, catalog), muscle_group(ex, catalog)
            if was and becomes and was != becomes:
                pair = f"«{ex['name']}» ({ex.get('muscle_group')}) заменяет «{old.get('exercise_name')}» ({was})"
                if (was in NON_STRENGTH) != (becomes in NON_STRENGTH) or {was, becomes} <= NON_STRENGTH:
                    out.append(err("E_MUSCLE_GROUP", f"{pair}: другой вид нагрузки, замена должна быть из той же группы мышц", i))
                else:  # e.g. RDL (ноги) → weighted back extension (спина): same chain, the trainer judges
                    out.append(err("W_MUSCLE_GROUP", f"{pair}: другая группа мышц, проверьте, что нагрузка та же", i))
        sets, reps = op.get("sets"), op.get("reps")
        if sets is not None and not (1 <= sets <= vol["sets"] + 2):
            out.append(err("E_VOLUME", f"{sets} подходов вне допустимого диапазона 1–{vol['sets'] + 2} для уровня {level}", i))
        if reps is not None and not (3 <= reps <= 25):
            out.append(err("E_VOLUME", f"{reps} повторений вне диапазона 3–25", i))
        elif reps is not None and abs(reps - vol["reps"]) > 4:
            out.append(err("W_VOLUME_LEVEL", f"{reps} повторений далеко от нормы уровня {level} (~{vol['reps']})", i))

    for day in new.get("days", []):
        exs = day.get("exercises", [])
        if not exs:
            out.append(err("E_DAY_SIZE", f"день «{day.get('title')}» остался пустым"))
        elif len(exs) > vol["ex_per_day"] + 3:
            out.append(err("E_DAY_SIZE", f"в дне «{day.get('title')}» {len(exs)} упражнений — слишком много для уровня {level}"))
        ids = [e.get("exercise_id") for e in exs if e.get("exercise_id")]
        dups = {x for x in ids if ids.count(x) > 1}
        for d in dups:
            out.append(err("E_DUPLICATE", f"в дне «{day.get('title')}» дважды «{catalog.get(d, {}).get('name', d)}»"))
        for e in exs:
            ex = catalog.get(e.get("exercise_id") or "")
            if not ex or (day.get("day_index"), ex["id"]) in touched:
                continue
            clash = hurt & body_parts(ex.get("contraindications"))
            if clash:
                out.append(err("W_REMAINING_CONTRA",
                               f"в плане остаётся «{ex['name']}», противопоказанное при: {', '.join(sorted(clash))}"))
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        raw = sys.stdin.read() if argv[1] == "-" else Path(argv[1]).read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        print(f"cannot read input: {e}", file=sys.stderr)
        return 2
    violations = validate(data["plan"], data.get("ops", []), data.get("client", {}))
    print(json.dumps({"ok": not any(v["severity"] == "error" for v in violations), "violations": violations},
                     ensure_ascii=False, indent=2))
    return 1 if any(v["severity"] == "error" for v in violations) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
