"""Dizziness reaches the trainer before any model call, even inside a technique question (held-out row h07)."""

import pytest

from tulpar_ai.graph.chat import HARD


@pytest.mark.parametrize("text", [
    "как правильно дышать при жиме лежа? а то я задерживаю дыхание и голова кружится немного",
    "На жиме лёжа после каждого подхода голова кружится и темнеет в глазах",
    "После кардио сильное головокружение",
    "кружится голова после становой",
    "Жаттығудан кейін басым айналады",
])
def test_dizziness_is_a_hard_marker(text):
    assert HARD.search(text)


@pytest.mark.parametrize("text", ["Круговая тренировка на всё тело", "Кружка протеина после зала", "Голова болит? нет, всё ок"])
def test_no_false_alarm(text):
    assert not HARD.search(text)
