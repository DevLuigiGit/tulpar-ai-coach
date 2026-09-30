"""The daily norm the answer node cites for «сколько калорий мне есть?» — the rules of corpus/nutrition.md."""

import pytest

from tulpar_ai.nutrition_calc import profile_source, targets

AIDAR = {"sex": "male", "age": 29, "height_cm": 178, "weight_kg": 84.2, "activity": "moderate", "goal": "cut",
         "level": "inter", "place": "gym", "training_days": [0, 2, 4], "goal_weight_kg": 79}
DANA = {"sex": "female", "age": 34, "height_cm": 165, "weight_kg": 61.5, "activity": "light", "goal": "keep"}


def test_norm_by_the_corpus_rules():
    # 10 × 84,2 + 6,25 × 178 − 5 × 29 + 5 = 1814,5; × 1,55 = 2812; снижение веса −20% → 2250
    t = targets(AIDAR)
    assert (t["bmr"], t["maintenance"], t["kcal"]) == (1815, 2812, 2250)
    assert (t["protein_g"], t["water_ml"], t["water_training_ml"]) == (168, 2700, 3200)  # 2 г/кг; 32 мл/кг, +500
    d = targets(DANA)  # 10 × 61,5 + 6,25 × 165 − 5 × 34 − 161 = 1315,25; × 1,375; поддержание
    assert (d["bmr"], d["kcal"], d["protein_g"], d["water_ml"]) == (1315, 1808, 123, 2000)


@pytest.mark.parametrize("missing", ["sex", "age", "height_cm", "weight_kg"])
def test_no_norm_without_the_profile(missing):
    assert targets({k: v for k, v in AIDAR.items() if k != missing}) is None


def test_water_is_clamped():
    assert targets({**DANA, "weight_kg": 40})["water_ml"] == 1500
    assert targets({**AIDAR, "weight_kg": 200})["water_ml"] == 5000


def test_profile_source_has_no_name_and_keeps_the_trainer_note():
    text = profile_source({"profile": AIDAR, "trainer_note": {"injury": True, "body": "Прыжки пока исключаем."}})
    assert "29 лет" in text and "2250 ккал" in text and "Прыжки пока исключаем" in text
    assert "Айдар" not in text
    assert profile_source({"profile": {}, "trainer_note": None}) is None


async def test_personal_answer_is_not_cached(monkeypatch):
    from tulpar_ai.graph import chat

    stored = []

    class Cache:
        async def put(self, *a):
            stored.append(a)

    monkeypatch.setattr(chat, "get_answer_cache", lambda: Cache())
    base = {"text": "Сколько калорий мне есть?", "kind": "answer", "reply": "2250 ккал [2]"}
    await chat.cache_store({**base, "citations": [{"n": 2, "title": "Ваш профиль", "source": chat.PROFILE_SOURCE}]})
    assert stored == []
    await chat.cache_store({**base, "citations": [{"n": 1, "title": "Правила питания Tulpar", "source": "nutrition"}]})
    assert len(stored) == 1
