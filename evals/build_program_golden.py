"""Build evals/golden/program.jsonl: real Tulpar plans with injected mistakes and the codes the validator
must raise. Deterministic — rerun after the catalogue changes."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
plans = json.loads((ROOT / "fixtures" / "plans.json").read_text(encoding="utf-8"))
cat = json.loads((ROOT / "skills" / "tulpar-program-builder" / "references" / "exercises_catalog.json").read_text(encoding="utf-8"))
by_name = {e["name"]: e for e in cat}


def plan(level: str, place: str) -> dict:
    p = next((p for p in plans if p["meta"].get("level") == level and p["meta"].get("place") == place), None)
    return copy.deepcopy(p or plans[0])


def contra(word: str, equipment: set[str]) -> dict:
    return next(e for e in cat if word in (e.get("contraindications") or "").lower()
                and (e.get("equipment") in equipment or e.get("equipment") is None))


def safe(group: str, equipment: set[str], exclude: set[str] = frozenset()) -> dict:
    return next(e for e in sorted(cat, key=lambda e: e["name"]) if e["muscle_group"] == group
                and (e.get("equipment") in equipment or e.get("equipment") is None)
                and not e.get("contraindications") and e["id"] not in exclude)


GYM = {"barbell", "dumbbell", "machine", "cable", "bodyweight"}
knee = {"level": "inter", "place": "gym", "trainer_note": {"injury": True, "body": "Болит левое колено после бега"}}
back = {"level": "inter", "place": "gym", "trainer_note": {"injury": True, "body": "Протрузия в пояснице, осевую нагрузку исключить"}}
home = {"level": "beginner", "place": "home", "trainer_note": {"injury": False, "body": "Тренируется дома с гантелями"}}
beginner = {"level": "beginner", "place": "gym", "trainer_note": None}

cases = []
gp = plan("inter", "gym")
d0 = gp["days"][0]
w0 = d0["exercises"][0]
g0 = w0["muscle_group"]
alt = safe(g0, GYM, {e["exercise_id"] for e in d0["exercises"]})
cases.append(dict(id="valid_replace", note="Корректная замена на упражнение той же группы без противопоказаний",
                  client=knee, plan=gp, ops=[{"op": "replace_exercise", "day_index": 0, "wex_id": w0["id"],
                                             "exercise_id": alt["id"], "sets": 4, "reps": 9}], expect_errors=[]))
kx = contra("колен", GYM)
cases.append(dict(id="knee_contra", note=f"Клиенту с больным коленом добавляют «{kx['name']}»", client=knee, plan=gp,
                  ops=[{"op": "add_exercise", "day_index": 0, "exercise_id": kx["id"], "sets": 3, "reps": 10}],
                  expect_errors=["E_CONTRAINDICATION"]))
# a back-contraindicated exercise replacing one of the same muscle group in a day that does not have it yet,
# so the case tests only the contraindication
bx, bd, bw = next((e, d, w) for e in cat if "спин" in (e.get("contraindications") or "").lower()
                  and (e.get("equipment") in GYM or e.get("equipment") is None)
                  for d in gp["days"] if e["id"] not in {x["exercise_id"] for x in d["exercises"]}
                  for w in d["exercises"] if w["muscle_group"] == e["muscle_group"])
cases.append(dict(id="back_contra", note=f"Клиенту с протрузией ставят «{bx['name']}»", client=back, plan=gp,
                  ops=[{"op": "replace_exercise", "day_index": bd["day_index"], "wex_id": bw["id"], "exercise_id": bx["id"]}],
                  expect_errors=["E_CONTRAINDICATION"]))
hp = plan("beginner", "home")
machine = next(e for e in cat if e.get("equipment") == "machine" and not e.get("contraindications"))
# same muscle group as the replaced exercise, so the case tests only the equipment
hm, hd, hw = next((e, d, w) for d in hp["days"] for w in d["exercises"] for e in cat
                  if e.get("equipment") == "machine" and not e.get("contraindications")
                  and e["muscle_group"] == w["muscle_group"])
cases.append(dict(id="home_machine", note=f"Домашнему клиенту ставят тренажёр «{hm['name']}»", client=home, plan=hp,
                  ops=[{"op": "replace_exercise", "day_index": hd["day_index"], "wex_id": hw["id"],
                        "exercise_id": hm["id"]}], expect_errors=["E_EQUIPMENT"]))
cases.append(dict(id="unknown_exercise", note="Выдуманный exercise_id", client=knee, plan=gp,
                  ops=[{"op": "add_exercise", "day_index": 1, "exercise_id": "00000000-0000-0000-0000-000000000000"}],
                  expect_errors=["E_UNKNOWN_EXERCISE"]))
cases.append(dict(id="bad_ref", note="Операция над упражнением, которого нет в плане", client=knee, plan=gp,
                  ops=[{"op": "set_volume", "day_index": 0, "wex_id": "not-in-plan", "sets": 3}], expect_errors=["E_BAD_REF"]))
bp = plan("beginner", "gym")
cases.append(dict(id="volume_beginner", note="Новичку 9 подходов", client=beginner, plan=bp,
                  ops=[{"op": "set_volume", "day_index": 0, "wex_id": bp["days"][0]["exercises"][0]["id"], "sets": 9}],
                  expect_errors=["E_VOLUME"]))
dup = d0["exercises"][1]
cases.append(dict(id="duplicate", note="В день добавляют упражнение, которое там уже есть", client=beginner, plan=gp,
                  ops=[{"op": "add_exercise", "day_index": 0, "exercise_id": dup["exercise_id"], "sets": 3, "reps": 10}],
                  expect_errors=["E_DUPLICATE"]))
cases.append(dict(id="empty_day", note="Из дня удаляют все упражнения", client=beginner, plan=gp,
                  ops=[{"op": "remove_exercise", "day_index": 0, "wex_id": e["id"]} for e in d0["exercises"]],
                  expect_errors=["E_DAY_SIZE"]))
cases.append(dict(id="empty_ops", note="Черновик без операций", client=knee, plan=gp, ops=[], expect_errors=["E_EMPTY_OPS"]))
cases.append(dict(id="reps_out_of_range", note="40 повторений", client=beginner, plan=bp,
                  ops=[{"op": "set_volume", "day_index": 0, "wex_id": bp["days"][0]["exercises"][0]["id"], "reps": 40}],
                  expect_errors=["E_VOLUME"]))
cases.append(dict(id="home_machine_and_knee", note="Две ошибки сразу: тренажёр дома и противопоказание по колену",
                  client={**home, "trainer_note": {"injury": True, "body": "Колено после операции"}}, plan=hp,
                  ops=[{"op": "add_exercise", "day_index": 0, "exercise_id": machine["id"]},
                       {"op": "add_exercise", "day_index": 1, "exercise_id": contra("колен", {"bodyweight", "dumbbell"})["id"]}],
                  expect_errors=["E_EQUIPMENT", "E_CONTRAINDICATION"]))

# The slip seen in a live draft: the reason describes a machine quad exercise, the id is a cardio burpee.
ld, lw = next((d, w) for d in gp["days"] for w in d["exercises"] if w["muscle_group"] == "ноги")
burpee = by_name["Бёрпи с отжиманием"]
cases.append(dict(id="wrong_group", note=f"«{lw['exercise_name']}» заменяют на «{burpee['name']}» из группы {burpee['muscle_group']}",
                  client=knee, plan=gp, ops=[{"op": "replace_exercise", "day_index": ld["day_index"], "wex_id": lw["id"],
                                             "exercise_id": burpee["id"], "reason": "изоляция квадрицепса в тренажёре"}],
                  expect_errors=["E_MUSCLE_GROUP"]))

out = ROOT / "evals" / "golden" / "program.jsonl"
out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
print(f"{len(cases)} cases → {out.relative_to(ROOT)}")
