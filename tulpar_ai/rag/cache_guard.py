"""Lexical guard for the answer cache: one-word flips that sentence embeddings barely notice.

A cosine threshold cannot separate "Я женщина, 30 лет. Какой минимум калорий…" from the same question about a man
(0.9887 with Jina), "минут в неделю" from "минут в день" (0.9778) or "Можно ли" from "Нельзя ли" (0.9757): the
sentence is otherwise identical, the answer is not. So a cached answer is served only when both questions agree on
every slot below. A slot mentioned on one side only is a disagreement too: a question that names no sex must not
get the answer written for women, a question with no time unit must not get the per-week figure.

The lists cover the flips found in evals (sex, population group, time unit, modality, negation, numbers, goal,
timing around a workout, body position, direction). Anything else (one exercise for another) is left to the
threshold, which evals/golden/cache_pairs.jsonl measures.
"""

from __future__ import annotations

import re

# slot -> value -> pattern. Patterns are matched on whole words of the lower-cased text with ё folded to е.
SLOTS: dict[str, dict[str, str]] = {
    "sex": {
        "female": r"женщин\w*|женск\w*|девушк\w*|девочк\w*",
        "male": r"мужчин\w*|мужск\w*|парень|парн(?:я|ю|ем|е|и|ей|ям|ями|ях)|мальчик\w*",
    },
    "group": {
        "child": r"ребен\w*|дет(?:и|ей|ям|ьми|ях|ск\w*|ств\w*)|школьник\w*|малыш\w*|девочк\w*|мальчик\w*",
        "teen": r"подрост\w*|тинейдж\w*",
        "adult": r"взросл\w*",
        "elderly": r"пожил\w*|пенсионер\w*",
        "pregnant": r"беремен\w*|кормящ\w*",
    },
    "period": {
        "second": r"секунд\w*|сек",
        "minute": r"минут\w*|мин",
        "hour": r"(?:пол)?час(?:а|ов|ы|у|ом|ам|ами)?",
        "day": r"день|дня|дней|дню|днем|сутк\w*|суточн\w*|ежедневн\w*|дневн(?:ая|ую|ой|ое|ые|ых|ым|ыми|ого|ому)",
        "week": r"недел\w*|еженедельн\w*",
        "month": r"месяц\w*|ежемесячн\w*|месячн\w*",
        "year": r"год|года|году|годом|ежегодн\w*|годов\w*",
    },
    # Order matters: negated phrases are matched and cut out before their positive forms.
    "modal": {
        "not_needed": r"не нужно|не надо|не обязательно|необязательно|не необходимо",
        "forbidden": r"нельзя|запрещ\w*|противопоказ\w*|не стоит|не следует|не рекомендуется|не рекомендуют|вредно",
        "needed": r"нужно|надо|необходимо|обязательно|следует|стоит ли",
        "allowed": r"можно|допустимо|разрешено|разрешается",
    },
    "goal": {
        "lose": r"похуд\w*|худе\w*|сброс\w*|дефицит\w*|жиросжиг\w*|сушк\w*",
        "gain": r"набор\w*|набра\w*|набир\w*|профицит\w*|масс(?:а|у|ы|е|ой)(?! тела)",
    },
    "timing": {
        "before": r"перед|до (?:тренировк\w*|заняти\w*|еды|сна|завтрак\w*|обед\w*|ужин\w*|пробежк\w*|зала)",
        "after": r"после",
        "during": r"во время|в процессе",
        "fasted": r"натощак",
    },
    "position": {"lying": r"лежа", "standing": r"стоя", "sitting": r"сидя"},
    "direction": {"forward": r"вперед", "backward": r"назад"},
}

# Checked after the modal phrases are cut out; "X, а не Y" names an alternative and does not negate the question.
NEGATION = re.compile(r"(?<!\bа )\b(?:не|ни|нет|без)\b")

NUMBER_WORDS = {
    1: "один одна одно одного одной одному одним одном одну",
    2: "два две двух двум двумя",
    3: "три трех трем тремя",
    4: "четыре четырех четырем четырьмя",
    5: "пять пяти пятью",
    6: "шесть шести шестью",
    7: "семь семи семью",
    8: "восемь восьми восемью",
    9: "девять девяти девятью",
    10: "десять десяти десятью",
}
_NUM_WORD = {w: str(n) for n, forms in NUMBER_WORDS.items() for w in forms.split()}
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?|\b(?:" + "|".join(sorted(_NUM_WORD, key=len, reverse=True)) + r")\b")

_COMPILED = {slot: {v: re.compile(rf"\b(?:{p})\b") for v, p in values.items()} for slot, values in SLOTS.items()}


def _norm(text: str) -> str:
    return " ".join((text or "").lower().replace("ё", "е").split())


def slots(text: str) -> dict[str, object]:
    """Slot values a question commits to; slots it does not mention are absent."""
    t = _norm(text)
    out: dict[str, object] = {}
    for slot, values in _COMPILED.items():
        found = set()
        for v, rx in values.items():
            if rx.search(t):
                found.add(v)
                if slot == "modal":  # cut the phrase out: "не нужно" must not also count as "нужно" or as a bare "не"
                    t = rx.sub(" ", t)
        if found:
            out[slot] = frozenset(found)
    if NEGATION.search(t):
        out["negation"] = frozenset({"not"})
    nums = tuple(_NUM_WORD.get(m, m.replace(",", ".")) for m in _NUMBER.findall(t))
    if nums:
        out["number"] = nums  # ordered: "5 подходов по 10" is not "10 подходов по 5"
    return out


def conflicts(a: str, b: str) -> list[str]:
    """Slots on which two questions disagree; empty means a cached answer to one may serve the other."""
    sa, sb = slots(a), slots(b)
    return sorted(k for k in sa.keys() | sb.keys() if sa.get(k) != sb.get(k))
