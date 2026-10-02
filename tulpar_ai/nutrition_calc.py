"""A client's daily norm by the rules of corpus/nutrition.md (Tulpar's targets.py and water.py), and the profile
source the answer node cites for questions about the client himself: «сколько калорий мне есть?».

The model gets the finished numbers instead of doing the arithmetic: a model multiplying 10 × 84,2 + 6,25 × 178 …
in its head is where made-up norms come from. No name goes in — ClientContext.for_llm() has none.
"""

from __future__ import annotations

from .pii import mask

ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "high": 1.725, "active": 1.725, "athlete": 1.9}
ACTIVITY_RU = {"sedentary": "сидячий образ жизни", "light": "лёгкая", "moderate": "умеренная", "high": "высокая",
               "active": "высокая", "athlete": "спортсмен"}
GOAL = {"cut": (-0.20, "снижение веса"), "lose": (-0.20, "снижение веса"), "keep": (0.0, "поддержание веса"),
        "maintain": (0.0, "поддержание веса"), "gain": (0.10, "набор массы"), "strength": (0.05, "сила")}
LEVEL_RU = {"beginner": "начинающий", "inter": "средний", "intermediate": "средний", "advanced": "продвинутый"}
PLACE_RU = {"gym": "в зале", "home": "дома"}
DAYS_RU = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]


def _r(x: float) -> int:
    return int(x + 0.5)


def targets(p: dict) -> dict | None:
    """Norm per corpus/nutrition.md; None without sex, age, height and weight — Tulpar asks to fill the profile then."""
    try:
        w, h, a = float(p["weight_kg"]), float(p["height_cm"]), float(p["age"])
    except (KeyError, TypeError, ValueError):
        return None
    if p.get("sex") not in ("male", "female") or not (w > 0 and h > 0 and a > 0):
        return None
    bmr = 10 * w + 6.25 * h - 5 * a + (5 if p["sex"] == "male" else -161)
    factor = ACTIVITY.get(p.get("activity") or "moderate", 1.55)
    maintenance = bmr * factor
    delta, goal = GOAL.get(p.get("goal") or "keep", (0.0, "поддержание веса"))
    kcal = maintenance * (1 + delta)
    protein = 2 * w
    fat = 0.25 * kcal / 9
    carbs = (kcal - protein * 4 - fat * 9) / 4
    water = min(5000, max(1500, round(32 * w / 100) * 100))
    return {"bmr": _r(bmr), "factor": factor, "maintenance": _r(maintenance), "goal": goal, "goal_delta": delta,
            "kcal": _r(kcal), "protein_g": _r(protein), "fat_g": _r(fat), "carbs_g": _r(carbs),
            "water_ml": water, "water_training_ml": min(5000, water + 500)}


def _num(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def _years(n) -> str:
    n = int(n)
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} год"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} года"
    return f"{n} лет"


def _percent(x: float) -> str:
    v = round(x * 100)
    return f"+{v}%" if v > 0 else f"−{-v}%" if v < 0 else "0%"


def _litres(ml: int) -> str:
    return _num(round(ml / 1000, 1)) + " л"


def profile_source(ctx: dict) -> str | None:
    """ClientContext.for_llm() → the text of the «profile» source. None when there is nothing useful in it."""
    p = ctx.get("profile") or {}
    facts = []
    if p.get("sex") in ("male", "female"):
        facts.append("мужчина" if p["sex"] == "male" else "женщина")
    if p.get("age"):
        facts.append(_years(p["age"]))
    if p.get("height_cm"):
        facts.append(f"рост {_num(p['height_cm'])} см")
    if p.get("weight_kg"):
        facts.append(f"вес {_num(p['weight_kg'])} кг")
    if p.get("goal") in GOAL:
        goal = f"цель — {GOAL[p['goal']][1]}"
        if p.get("goal_weight_kg"):
            goal += f" (желаемый вес {_num(p['goal_weight_kg'])} кг)"
        facts.append(goal)
    if p.get("level") in LEVEL_RU:
        facts.append(f"уровень — {LEVEL_RU[p['level']]}")
    if p.get("place") in PLACE_RU:
        facts.append(f"тренируется {PLACE_RU[p['place']]}")
    if p.get("activity") in ACTIVITY_RU:
        facts.append(f"активность — {ACTIVITY_RU[p['activity']]}")
    days = [DAYS_RU[d] for d in p.get("training_days") or [] if isinstance(d, int) and 0 <= d < 7]
    if days:
        facts.append("дни тренировок: " + ", ".join(days))
    lines = ["Профиль клиента: " + "; ".join(facts) + "."] if facts else []
    note = ctx.get("trainer_note")
    body = note.get("body") if isinstance(note, dict) else note
    if body:
        lines.append(f"Ограничения и заметка тренера: {mask(str(body))}")
    if p.get("limitations"):
        lines.append(f"Травмы и ограничения со слов клиента (анкета): {mask(str(p['limitations']))}")
    t = targets(p)
    if t:
        lines.append(
            f"Норма по правилам питания Tulpar для этого профиля: базовый обмен {t['bmr']} ккал; поддержание веса "
            f"{t['maintenance']} ккал (коэффициент активности {_num(t['factor'])}); с учётом цели «{t['goal']}» "
            f"({_percent(t['goal_delta'])}) — {t['kcal']} ккал в день; "
            f"белок {t['protein_g']} г (2 г на кг веса), жиры {t['fat_g']} г (25% калорий), углеводы {t['carbs_g']} г; "
            f"вода {_litres(t['water_ml'])} в день, в день тренировки {_litres(t['water_training_ml'])}.")
    elif p.get("onboarded") is False:  # a new client who has not filled in the questionnaire yet
        lines.append("Анкета клиента не заполнена: пол, возраст, рост и вес неизвестны, поэтому личную норму калорий, "
                     "белка, жиров, углеводов и воды посчитать нельзя. На вопрос о его собственной норме это и есть "
                     "ответ: скажи, что норму посчитаем после анкеты, и попроси заполнить её в приложении "
                     "(кнопка «Открыть»).")
    return "\n".join(lines) or None
