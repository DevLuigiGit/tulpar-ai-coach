"""Food name matching on the demo food table: plain drinks must not pick up honey, sugar or a latte."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tulpar_ai.gateway.base import MATCH_THRESHOLD, name_parts, rank_foods

FOODS = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "foods.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("query, expected", [
    ("чай", "Чай чёрный без сахара"),  # was «Чай с мёдом», 26 instead of 1 kcal/100 g
    ("кофе", "Кофе чёрный без сахара"),  # was «Раф-кофе», 92 kcal/100 g
    ("чай без сахара", "Чай чёрный без сахара"),  # was «Иван-чай с сахаром»: «без» was a stop word
    ("Кофе чёрный без сахара", "Кофе чёрный без сахара"),  # was «Кофе чёрный с сахаром»
    ("кола без сахара", "Кола без сахара"),
    ("кола", "Кола"),
    ("чай с молоком", "Чай с молоком без сахара"),
    ("гусь с кожей", "Гусь с кожей сырой"),
    ("плова", "Плов"),
    ("какао", "Какао с молоком"),
])
def test_top_match(query, expected):
    top = rank_foods(query, FOODS, 3)[0]
    assert top.name == expected and top.score >= MATCH_THRESHOLD, [(f.name, f.score) for f in rank_foods(query, FOODS, 3)]


def test_name_parts():
    assert name_parts("Чай с лимоном и сахаром") == ({"чай"}, {"лимоно", "сахаро"}, set())
    assert name_parts("Гусь без кожи сырой") == ({"гусь", "сырой"}, set(), {"кожи"})
    assert name_parts("Мокко (кофе с шоколадом)") == ({"мокко", "кофе"}, {"шокола"}, set())
    assert name_parts("Салат из огурцов и помидоров с маслом") == ({"салат", "огурцо", "помидо"}, {"маслом"}, set())


async def test_meal_card_logs_plain_tea(app_state):
    """The demo turn «Съел 200 г плова и чай» (fake LLM extracts «плов» and «чай»)."""
    from tulpar_ai.graph import runner

    _, gw = app_state
    client = await gw.demo_user("client")
    res = await runner.run_chat_turn(client.id, text="Съел 200 г плова и чай")
    names = [i["name"] for i in res["meal"]["items"]]
    assert names == ["Плов", "Чай чёрный без сахара"], names


async def test_composite_dish_falls_back_to_its_parts(app_state):
    """«гречка с курицей» is not a catalog row: its parts are found and the grams are split as a guess."""
    from tulpar_ai.graph.chat import _resolve_items

    _, gw = app_state
    client = await gw.demo_user("client")
    items, unknown = await _resolve_items(client.id, [{"name": "гречка с курицей", "grams": 200}])
    names = [i["name"].lower() for i in items]
    assert any("греч" in n for n in names) and any("кур" in n for n in names), names
    assert not unknown
    assert all(i["grams"] == 100 and i["grams_source"] == "default" and i["asked_as"] == "гречка с курицей" for i in items)
