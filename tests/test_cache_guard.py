"""Lexical guard of the answer cache: one-word flips that embeddings score at 0.95+ must never share an answer."""

import json
from pathlib import Path

import pytest

from tulpar_ai.rag.cache_guard import conflicts, slots

ROOT = Path(__file__).resolve().parents[1]

# Pairs from the review probes with their Jina retrieval.query cosine: every one cleared the 0.95 threshold.
FLIPS = [
    ("Я женщина, 30 лет. Какой минимум калорий мне поставит Tulpar?",
     "Я мужчина, 30 лет. Какой минимум калорий мне поставит Tulpar?", "sex"),  # 0.9887
    ("Сколько минут в неделю нужно заниматься взрослому по ВОЗ?",
     "Сколько минут в день нужно заниматься взрослому по ВОЗ?", "period"),  # 0.9778
    ("Сколько раз в неделю ВОЗ советует делать силовые тренировки?",
     "Сколько раз в день ВОЗ советует делать силовые тренировки?", "period"),  # 0.9681
    ("Сколько минут в неделю нужно заниматься взрослому по ВОЗ?",
     "Сколько минут в месяц нужно заниматься взрослому по ВОЗ?", "period"),  # 0.9612
    ("Сколько секунд держать вакуум живота?", "Сколько минут держать вакуум живота?", "period"),  # 0.9735
    ("Можно ли тренироваться натощак?", "Нельзя ли тренироваться натощак?", "modal"),  # 0.9757
    ("Можно ли делать присед со штангой каждый день?", "Нужно ли делать присед со штангой каждый день?", "modal"),
    ("Можно ли делать становую тягу каждый день?", "Нужно ли делать становую тягу каждый день?", "modal"),
    ("Как Tulpar считает мою дневную норму калорий?", "Как Tulpar считает мою недельную норму калорий?", "period"),
    ("Какой максимальный темп похудения в неделю разрешает приложение?",
     "Какой максимальный темп похудения в месяц разрешает приложение?", "period"),
    ("Мне 45 лет, вес 90 кг, рост 180. Сколько калорий мне нужно в день?",
     "Мне 45 лет, вес 60 кг, рост 165. Сколько калорий мне нужно в день?", "number"),
    ("Что есть до тренировки?", "Что есть после тренировки?", "timing"),
    ("Сколько белка в день нужно при похудении?", "Сколько белка в день нужно при наборе массы?", "goal"),
    ("Как правильно жать штангу лёжа?", "Как правильно жать штангу стоя?", "position"),
    ("Сколько раз в неделю ВОЗ советует делать силовые тренировки?",
     "Сколько раз в неделю ВОЗ советует делать силовые тренировки детям?", "group"),
    ("Можно ли тренироваться без разминки?", "Можно ли тренироваться с разминкой?", "negation"),
]

# Real paraphrases that must stay cacheable.
SAME = [
    ("Я женщина, 30 лет. Какой минимум калорий мне поставит Tulpar?",
     "Мне 30 лет, я женщина. Какой минимум калорий мне поставит Tulpar?"),
    ("Зачем делать тягу гантели одной рукой, а не двумя сразу?",
     "В чём смысл тяги гантели одной рукой вместо двух рук сразу?"),  # "а не" names an alternative
    ("Есть ли противопоказания у ягодичного мостика?", "Кому не стоит делать ягодичный мостик?"),
    ("Как Tulpar считает мою дневную норму калорий?", "По какой формуле приложение рассчитывает мою суточную норму калорий?"),
    ("Сколько должен двигаться ребёнок 10 лет в день?", "Какая норма ежедневной активности для ребёнка десяти лет?"),
    ("Если продукт в базе потом исправят, поменяется ли мой прошлый дневник питания?",
     "Изменится ли старый дневник питания, если в базе потом поправят продукт?"),
]


@pytest.mark.parametrize("a,b,slot", FLIPS)
def test_flip_is_a_conflict(a, b, slot):
    assert slot in conflicts(a, b)
    assert conflicts(a, a) == [] and conflicts(b, b) == []


@pytest.mark.parametrize("a,b", SAME)
def test_paraphrase_passes(a, b):
    assert conflicts(a, b) == []


def test_slot_mentioned_on_one_side_only_is_a_conflict():
    """A question that names no sex must not get the answer written for women."""
    assert conflicts("Какой минимум калорий в день Tulpar ставит женщине?", "Какой минимум калорий в день ставит Tulpar?") == ["sex"]


def test_negated_modals_are_their_own_values():
    assert slots("Не нужно ли делать разминку?") == {"modal": frozenset({"not_needed"})}
    assert slots("Кому не стоит делать мостик?") == {"modal": frozenset({"forbidden"})}
    assert slots("Можно ли не делать разминку?") == {"modal": frozenset({"allowed"}), "negation": frozenset({"not"})}


def test_lookalike_words_are_not_slots():
    for text in ["Как вести дневник питания?", "Как часто тренироваться?", "Что делать в понедельник?",
                 "Как Tulpar считает индекс массы тела?", "Сколько стоит абонемент?"]:
        assert slots(text) == {}, text


def test_numbers_keep_order_and_read_number_words():
    assert slots("5 подходов по 10 повторений")["number"] == ("5", "10")
    assert "number" in conflicts("5 подходов по 10 повторений", "10 подходов по 5 повторений")
    assert conflicts("Сколько ккал в одном кг жира?", "Сколько ккал в 1 кг жира?") == []


def test_golden_flips_are_caught():
    """Every near-miss of evals/golden/cache_pairs.jsonl labelled with a slot is caught on that slot."""
    rows = [json.loads(line) for line in (ROOT / "evals" / "golden" / "cache_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    labelled = [r for r in rows if r.get("slot")]
    assert len(labelled) >= 35
    missed = [r["id"] for r in labelled if r["slot"] not in conflicts(r["a"], r["b"])]
    assert missed == []
