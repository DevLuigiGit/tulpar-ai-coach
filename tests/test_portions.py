"""Household measures in meal messages (simulation finding 1): «две ложки сахара» was logged as 150 g ≈ 580 kcal.

Numbers are checked by evals/meal_eval.py on evals/golden/meal_text*.jsonl; these tests pin the deterministic parts.
"""

import json
from pathlib import Path

import pytest

from tulpar_ai.gateway.base import rank_foods
from tulpar_ai.portions import estimate_grams, measure_in_text

FOODS = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "foods.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name, qty, unit, grams", [
    ("сахар", 2, "ложка", 14),
    ("мёд", 1, "ложка", 10),  # denser than water
    ("масло оливковое", 1, "столовая ложка", 15),
    ("печенье", 2, "штука", 24),
    ("манты", 5, "штука", 225),
    ("хлеб ржаной", 1, "кусок", 30),
    ("борщ", 0.5, "тарелка", 150),
    ("кефир", 1, "стакан", 200),
])
def test_estimate(name, qty, unit, grams):
    assert estimate_grams(name, qty, unit) == grams


@pytest.mark.parametrize("name, qty, unit", [("гречка", None, None), ("гречка", 1, "штука"), ("сахар", 0, "ложка"),
                                             ("сахар", "много", "ложка"), ("кусок чего-то", 1, "кусок")])
def test_no_guess_without_a_known_weight(name, qty, unit):
    assert estimate_grams(name, qty, unit) is None


@pytest.mark.parametrize("text, name, qty, unit", [
    ("Горсть грецких орехов", "грецкие орехи", 1, "горсть"),
    ("Съел пять мантов", "манты", 5, "штука"),
    ("Полтарелки борща", "борщ", 0.5, "тарелка"),
    ("Выпил кофе, положил две ложки сахара", "сахар", 2, "ложка"),
    ("Кусок черного хлеба", "черный хлеб", 1, "кусок"),
    ("Две столовые ложки арахисовой пасты", "арахисовая паста", 2, "столовая ложка"),
    ("12 пельменей", "пельмени", 12, "штука"),
    ("Съела яблоко", "яблоко", 1, "штука"),
])
def test_measure_read_off_the_text(text, name, qty, unit):
    q, u, _ = measure_in_text(text, name)
    assert (q, u) == (qty, unit)


@pytest.mark.parametrize("text, name", [("200 г гречки", "гречка"), ("10 г гречки", "гречка"), ("съел гречку", "гречка"),
                                        ("Две котлеты с пюре", "пюре")])
def test_no_measure_invented(text, name):
    assert measure_in_text(text, name) is None


@pytest.mark.parametrize("query, expected", [
    ("хлеб", "Хлеб белый"),  # was «Хлебцы»: «хлеб» and «хлебцы» shared a 4-letter prefix
    ("кусок черного хлеба", "Хлеб ржаной"),  # was «Хлебцы»
    ("чай с сахаром", "Чай чёрный с сахаром"),  # was «Иван-чай с сахаром»: the shorter name won the tie
    ("молоко", "Молоко 2.5%"),  # was «Молоко козье»: «2.5» counted as two extra words
    ("яйцо", "Яйцо куриное"),  # was «Яйцо утиное»
    ("батон", "Батон нарезной"),
])
def test_everyday_names(query, expected):
    assert rank_foods(query, FOODS, 1)[0].name == expected


def test_addition_alone_is_not_a_match():
    assert all(f.name.startswith("Пельмени") for f in rank_foods("пельмени со сметаной", FOODS, 5))


async def test_card_shows_the_estimate(app_state, monkeypatch):
    from tulpar_ai.graph import chat

    async def fake_json_call(role, system, user, **kw):
        return {"items": [{"name": "сахар", "grams": None, "qty": 2, "unit": "ложка", "said": "две ложки"},
                          {"name": "пельмени со сметаной", "grams": None, "qty": 6, "unit": "штука", "said": "шесть"}]}, {}

    monkeypatch.setattr(chat, "json_call", fake_json_call)
    _, gw = app_state
    client = await gw.demo_user("client")
    res = await chat.meal_text({"client_id": client.id, "text": "кофе, две ложки сахара и шесть пельменей со сметаной"})
    items = {i["name"]: i for i in res["meal"]["items"]}
    assert items["Сахар"]["grams"] == 14 and items["Сахар"]["grams_source"] == "estimate"
    assert items["Пельмени отварные"]["grams"] == 72  # the dish keeps its grams, the sour cream is not guessed
    assert "примерно, «две ложки»" in res["reply"]
