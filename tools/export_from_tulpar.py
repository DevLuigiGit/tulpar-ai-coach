"""Export the data this project needs from a local Tulpar checkout — READ-ONLY.

The script only reads files from the Tulpar repository (seed literals, data JSON, docs) and writes
into THIS repository (fixtures/, corpus/, skills/.../references/). It never imports Tulpar code,
never touches its database, git, Docker or Railway.

    python tools/export_from_tulpar.py --tulpar ~/Projects/tulpar-saas

What comes out:
  fixtures/exercises.json   196 exercises: muscle group, equipment, compound flag, RU technique,
                            benefit, contraindications (ids are stable uuid5 from the name)
  fixtures/foods.json       1305 catalogue foods, КБЖУ per 100 g
  fixtures/plans.json       12 ready-made Tulpar plans with exercise ids resolved
  fixtures/demo.json        demo trainer + client, active plan, trainer note, 14 days of diary,
                            recent sessions and weights — generated deterministically
  corpus/exercises.jsonl    one RAG document per exercise that has Russian text
  corpus/nutrition.md       nutrition rules (safe pace, per-100 g snapshot, floors)
  skills/tulpar-program-builder/references/exercises_catalog.json
                            compact catalogue for the offline plan validator
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import uuid
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS = uuid.UUID("5b8f2c1e-7a4d-4e3b-9c2a-1f0e6d5c4b3a")  # namespace for stable ids


def sid(kind: str, name: str) -> str:
    return str(uuid.uuid5(NS, f"{kind}:{name}"))


def seed_literals(seed_py: Path, names: set[str]) -> dict:
    tree = ast.parse(seed_py.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        target = value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        if target in names and value is not None:
            out[target] = ast.literal_eval(value)
    missing = names - out.keys()
    if missing:
        raise SystemExit(f"seed.py: literals not found: {sorted(missing)}")
    return out


EQUIPMENT_HINTS = [  # order matters: the first match wins
    ("cable", ("блок", "кроссовер", "канат")),
    ("machine", ("тренажер", "тренажёр", "гакк", "смита", "дорожк", "велотренаж", "эллипс", "степпер", "гиперэкстенз")),
    ("barbell", ("штанг", "грифом", "гриф ")),
    ("kettlebell", ("гир",)),
    ("band", ("резин", "эспандер", "лент")),
    ("dumbbell", ("гантел",)),
]


def infer_equipment(name: str) -> str | None:
    n = name.lower()
    for eq, stems in EQUIPMENT_HINTS:
        if any(s in n for s in stems):
            return eq
    return None


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("rows", []) if isinstance(data, dict) else []


def export(tulpar: Path) -> None:
    seed_py = tulpar / "backend" / "scripts" / "seed.py"
    lit = seed_literals(
        seed_py,
        {"BASE_EXERCISES", "EXERCISE_EQUIPMENT", "EXERCISE_COMPOUND", "EXERCISE_CONTENT", "BASE_FOODS", "BASE_PLANS"},
    )
    data_dir = tulpar / "backend" / "data"
    ru_text: dict[str, dict] = {}
    for fname in ("b18_ru_authored.json", "b18_ru_technique.json", "b18_ru_reworded.json", "b18_ru_from_external.json"):
        for r in rows(data_dir / fname):
            name = r.get("our_name")
            text = r.get("description_ru") or r.get("description_after")
            if name and text and name not in ru_text:
                ru_text[name] = {"text": text, "origin": r.get("text_origin") or fname,
                                 "muscle_group": r.get("muscle_group"), "equipment": r.get("equipment")}

    exercises = []
    for name, measurement, met, muscle, slug in lit["BASE_EXERCISES"]:
        content = lit["EXERCISE_CONTENT"].get(name)
        extra = ru_text.get(name, {})
        exercises.append({
            "id": sid("exercise", name),
            "name": name,
            "slug": slug,
            "muscle_group": muscle,
            "measurement_type": measurement,
            "met": met,
            "equipment": lit["EXERCISE_EQUIPMENT"].get(name) or extra.get("equipment") or infer_equipment(name),
            "equipment_inferred": not (lit["EXERCISE_EQUIPMENT"].get(name) or extra.get("equipment")) and bool(infer_equipment(name)),
            "is_compound": name in lit["EXERCISE_COMPOUND"],
            "description": (content[0] if content else None) or extra.get("text"),
            "benefit": content[1] if content else None,
            "contraindications": content[2] if content else None,
            "text_origin": "tulpar-seed" if content else extra.get("origin"),
        })
    by_name = {e["name"]: e for e in exercises}

    foods = [
        {"id": sid("food", n), "name": n, "kcal": k, "protein": p, "fat": f, "carbs": c}
        for n, k, p, f, c in lit["BASE_FOODS"]
    ]

    plans = []
    for p in lit["BASE_PLANS"]:
        days = []
        for di, d in enumerate(p["days"]):
            exs = []
            for ei, (ex_name, sets, reps) in enumerate(d["ex"]):
                ex = by_name.get(ex_name)
                exs.append({
                    "id": sid("wex", f"{p['title']}|{di}|{ei}"),
                    "exercise_id": ex["id"] if ex else None,
                    "exercise_name": ex_name,
                    "muscle_group": ex["muscle_group"] if ex else None,
                    "equipment": ex["equipment"] if ex else None,
                    "target_sets": sets,
                    "target_reps": reps,
                })
            days.append({"id": sid("day", f"{p['title']}|{di}"), "day_index": di, "title": d["title"],
                         "weekday": d.get("weekday"), "exercises": exs})
        plans.append({"id": sid("plan", p["title"]), "title": p["title"], "mode": p.get("mode"),
                      "meta": p.get("meta", {}), "days": days})

    demo = build_demo(plans, foods, by_name)

    (ROOT / "fixtures").mkdir(exist_ok=True)
    dump(ROOT / "fixtures" / "exercises.json", exercises)
    dump(ROOT / "fixtures" / "foods.json", foods)
    dump(ROOT / "fixtures" / "plans.json", plans)
    dump(ROOT / "fixtures" / "demo.json", demo)

    corpus = ROOT / "corpus"
    corpus.mkdir(exist_ok=True)
    with (corpus / "exercises.jsonl").open("w", encoding="utf-8") as fh:
        n_docs = 0
        for e in exercises:
            if not e["description"]:
                continue
            parts = [f"Упражнение: {e['name']}.", f"Группа мышц: {e['muscle_group']}."]
            if e["equipment"]:
                parts.append(f"Инвентарь: {e['equipment']}.")
            parts.append(f"Техника: {e['description']}")
            if e["benefit"]:
                parts.append(f"Польза: {e['benefit']}")
            if e["contraindications"]:
                parts.append(f"Противопоказания: {e['contraindications']}")
            fh.write(json.dumps({"id": e["id"], "title": e["name"], "source": "exercises",
                                 "muscle_group": e["muscle_group"], "equipment": e["equipment"],
                                 "text": " ".join(parts)}, ensure_ascii=False) + "\n")
            n_docs += 1

    nutrition = (tulpar / "docs" / "nutrition.md").read_text(encoding="utf-8")
    # keep only the rules; drop the line about which Ollama models the backend used (stale, internal)
    nutrition = "\n".join(l for l in nutrition.splitlines() if "Модели Ollama" not in l)
    (corpus / "nutrition.md").write_text(
        "<!-- Источник: Tulpar docs/nutrition.md, выгружено tools/export_from_tulpar.py -->\n" + nutrition,
        encoding="utf-8")

    refs = ROOT / "skills" / "tulpar-program-builder" / "references"
    refs.mkdir(parents=True, exist_ok=True)
    dump(refs / "exercises_catalog.json", [
        {k: e[k] for k in ("id", "name", "muscle_group", "equipment", "is_compound", "measurement_type", "contraindications")}
        for e in exercises
    ])

    print(f"exercises={len(exercises)} (with RU text: {n_docs}) foods={len(foods)} plans={len(plans)}")


def build_demo(plans: list[dict], foods: list[dict], by_name: dict) -> dict:
    """Deterministic demo world: one trainer, two clients. Mirrors Tulpar's seed_demo profile."""
    rnd = random.Random(42)
    base = next(p for p in plans if p["meta"].get("level") == "inter" and p["meta"].get("place") == "gym"
                and p["meta"].get("days_per_week") == 3) if any(
        p["meta"].get("level") == "inter" and p["meta"].get("days_per_week") == 3 for p in plans) else plans[0]
    trainer = {"id": sid("user", "demo-trainer"), "role": "trainer", "name": "Арман Тренер",
               "telegram_id": "demo-trainer", "gym": "Iron Gym Almaty"}
    clients = [
        {"id": sid("user", "demo-student"), "role": "client", "name": "Айдар Ким", "telegram_id": "demo-student",
         "sex": "male", "age": 29, "height_cm": 178, "goal": "cut", "level": "inter", "place": "gym",
         "training_days": [0, 2, 4], "activity": "moderate", "weight_kg": 84.2, "goal_weight_kg": 79,
         "trainer_note": {"injury": True,
                          "body": "Жалуется на левое колено после бега. Глубокие приседания со штангой и прыжки пока исключаем, "
                                  "жим ногами — только в неполной амплитуде."},
         "plan_template": base["title"]},
        {"id": sid("user", "demo-student-2"), "role": "client", "name": "Дана Сейткали", "telegram_id": "demo-student-2",
         "sex": "female", "age": 34, "height_cm": 165, "goal": "keep", "level": "beginner", "place": "home",
         "training_days": [1, 3, 5], "activity": "light", "weight_kg": 61.5, "goal_weight_kg": 60,
         "trainer_note": {"injury": False, "body": "Тренируется дома: гантели и резинка. Хочет больше кардио."},
         "plan_template": next((p["title"] for p in plans if p["meta"].get("place") == "home"), plans[0]["title"])},
    ]
    today = date(2026, 9, 24)
    staple = [f for f in foods if any(w in f["name"].lower() for w in
              ("плов", "гречк", "курин", "овсян", "яйц", "творог", "банан", "яблок", "рис", "лагман", "манты",
               "салат", "кефир", "хлеб", "говядин"))] or foods[:60]
    for c in clients:
        tpl = next(p for p in plans if p["title"] == c["plan_template"])
        plan = json.loads(json.dumps(tpl))
        plan["id"] = sid("active-plan", c["id"])
        for d in plan["days"]:
            d["id"] = sid("active-day", f"{c['id']}|{d['day_index']}")
            for i, e in enumerate(d["exercises"]):
                e["id"] = sid("active-wex", f"{c['id']}|{d['day_index']}|{i}")
        c["active_plan"] = plan
        diary = []
        for back in range(14, 0, -1):
            on = (today - timedelta(days=back)).isoformat()
            for meal in ("breakfast", "lunch", "dinner"):
                f = rnd.choice(staple)
                diary.append({"on_date": on, "meal": meal, "food_id": f["id"], "name": f["name"],
                              "grams": rnd.choice([150, 200, 250, 300]), "kcal": f["kcal"], "protein": f["protein"],
                              "fat": f["fat"], "carbs": f["carbs"]})
        c["diary"] = diary
        sessions = []
        for back in (12, 10, 7, 5, 2):
            day = plan["days"][back % len(plan["days"])]
            sets = []
            for e in day["exercises"][:4]:
                for s in range(1, (e["target_sets"] or 3) + 1):
                    sets.append({"exercise_name": e["exercise_name"], "set_index": s,
                                 "reps": max(1, (e["target_reps"] or 10) - rnd.choice([0, 0, 1, 2])),
                                 "weight": rnd.choice([None, 20, 30, 40, 50]), "rpe": rnd.choice([7, 8, 8, 9])})
            sessions.append({"completed_at": (today - timedelta(days=back)).isoformat(), "day_title": day["title"],
                             "avg_rpe": 8, "sets": sets})
        c["sessions"] = sessions
        w0 = c["weight_kg"] + 1.6
        c["weights"] = [{"measured_at": (today - timedelta(days=3 * i)).isoformat(),
                         "weight_kg": round(w0 - 0.16 * (10 - i) + rnd.uniform(-0.2, 0.2), 1)} for i in range(10, 0, -1)]
        del c["plan_template"]
    return {"note": "Демо-данные, сгенерированные детерминированно из сидов Tulpar. Реальных людей здесь нет.",
            "today": today.isoformat(), "trainer": trainer, "clients": clients}


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tulpar", default=str(Path.home() / "Projects" / "tulpar-saas"))
    export(Path(ap.parse_args().tulpar).expanduser())
