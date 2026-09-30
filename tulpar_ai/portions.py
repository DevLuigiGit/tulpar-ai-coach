"""Household measures → grams for the food diary: «две ложки сахара», «пара печенек», «пять мантов».

The meal_text prompt only reads the count and the unit off the message; the weight comes from these tables, so the
same words always give the same grams (the model used to answer «две ложки сахара» with 150 g once and 16 g the next
time). Typical weights, rounded; the card shows them as an estimate the client can correct.
"""

from __future__ import annotations

import re

# grams of one unit, whatever the product
UNIT_GRAMS = {
    "чайная ложка": 5, "ложка": 7, "столовая ложка": 15, "горсть": 30,
    "стакан": 200, "чашка": 200, "кружка": 300, "бокал": 200, "бутылка": 500,
    "тарелка": 300, "миска": 300, "порция": 250,
}
# denser than water: honey, jam and syrup weigh more per spoon
_DENSE = re.compile(r"м[её]д|варень|джем|сироп|сгущ")

# grams of one piece; the first pattern that matches the product name wins, so specific ones go first
PIECE_GRAMS: list[tuple[re.Pattern, float]] = [(re.compile(p), g) for p, g in [
    (r"перепел", 12), (r"яйц", 55),
    (r"банан", 120), (r"яблок", 180), (r"груш", 170), (r"апельсин", 150), (r"мандарин", 70), (r"персик", 150),
    (r"киви", 75), (r"огур", 100), (r"помидор|томат", 120), (r"картоф", 100),
    (r"^печень[ея]|^печенек", 12), (r"конфет", 12), (r"пряник", 35), (r"вафл", 15), (r"зефир", 35),
    (r"мант", 45), (r"пельмен", 12), (r"вареник", 25), (r"хинкал", 70), (r"баурсак", 25), (r"самс", 120),
    (r"беляш", 100), (r"чебурек", 130), (r"пирож", 70), (r"блинчик", 70), (r"блин", 50), (r"оладь|оладий", 40),
    (r"сырник", 50), (r"котлет", 80), (r"сардельк", 100), (r"сосиск", 50), (r"булк|булочк", 60),
    (r"круассан", 60), (r"пончик", 60), (r"маффин|кекс", 70), (r"тост", 30), (r"хлебц", 10),
    (r"грудк", 200), (r"бедр", 120), (r"йогурт", 125), (r"сырок", 45), (r"батончик", 50), (r"шоколадк", 90),
]]
# grams of one slice («кусок», «ломтик»)
SLICE_GRAMS: list[tuple[re.Pattern, float]] = [(re.compile(p), g) for p, g in [
    (r"хлеб|батон|багет", 30), (r"пицц", 110), (r"торт|пирог|шарлотк|медовик|наполеон|чизкейк", 100),
    (r"арбуз|дын", 250), (r"колбас|ветчин|сыр", 20),
    (r"мяс|куриц|курин|говяд|свин|баранин|конин|рыб|лосос|семг|стейк|казы", 100),
]]

_UNIT_ALIASES = {"шт": "штука", "штук": "штука", "штуки": "штука", "ч. л.": "чайная ложка", "ч.л.": "чайная ложка",
                 "ст. л.": "столовая ложка", "ст.л.": "столовая ложка", "ломтик": "кусок", "долька": "кусок"}


def _norm(text: str) -> str:
    return (text or "").lower().replace("ё", "е").strip()


def _lookup(table: list[tuple[re.Pattern, float]], name: str) -> float | None:
    n = _norm(name)
    return next((g for p, g in table if p.search(n)), None)


def unit_weight(name: str, unit: str | None) -> float | None:
    """Grams of ONE unit of this product, or None when the tables do not know it."""
    u = _norm(unit)
    u = _UNIT_ALIASES.get(u, u)
    if not u:
        return None
    if u == "штука":
        return _lookup(PIECE_GRAMS, name)
    if u == "кусок":
        return _lookup(SLICE_GRAMS, name)
    g = UNIT_GRAMS.get(u)
    if g is not None and "ложка" in u and _DENSE.search(_norm(name)):
        g = round(g * 1.4)
    return g


def estimate_grams(name: str, qty, unit: str | None) -> float | None:
    """qty × unit weight, e.g. («сахар», 2, «ложка») → 14. None when the count or the unit weight is unknown."""
    try:
        q = float(qty)
    except (TypeError, ValueError):
        return None
    if not 0 < q <= 50:
        return None
    w = unit_weight(name, unit)
    return round(q * w) if w else None


# ── the same count read off the text, for when the model leaves qty/unit empty (it does, even at temperature 0) ──
_NUMBERS = [(r"\d+(?:[.,]\d+)?", None), (r"полтор\w*", 1.5), (r"половин\w*|пол", 0.5), (r"одн\w*|один", 1),
            (r"дв[ае]|двух", 2), (r"пар[ауы]?|парочк\w*", 2), (r"тр[ие]|тр[её]х", 3), (r"четыр\w*", 4), (r"пят\w*", 5),
            (r"шест\w*", 6), (r"сем\w*", 7), (r"восем\w*", 8), (r"девят\w*", 9), (r"десят\w*", 10)]
_UNITS = [(r"чайн\w*\s+ложк\w*|ч\.\s?л\.", "чайная ложка"), (r"столов\w*\s+ложк\w*|ст\.\s?л\.", "столовая ложка"),
          (r"ложк\w*|ложечк\w*", "ложка"), (r"стакан\w*", "стакан"), (r"чашк\w*", "чашка"), (r"кружк\w*", "кружка"),
          (r"бокал\w*", "бокал"), (r"бутылк\w*", "бутылка"), (r"тарелк\w*", "тарелка"), (r"миск\w*", "миска"),
          (r"порци\w*", "порция"), (r"горст\w*", "горсть"), (r"кус\w*|ломтик\w*|дольк\w*", "кусок"),
          (r"шт\.?|штук\w*", "штука")]
_NUM_RE = "|".join(f"(?:{p})" for p, _ in _NUMBERS)
_UNIT_RE = "|".join(f"(?:{p})" for p, _ in _UNITS)
_WEIGHT_UNIT = r"(?:г|гр|грамм\w*|кг|мл|л|литр\w*)\b"


def _number(word: str | None) -> float:
    if not word:
        return 1.0
    w = _norm(word)
    for p, q in _NUMBERS:
        if re.fullmatch(p, w):
            return float(w.replace(",", ".")) if q is None else float(q)
    return 1.0


def _stems(name: str) -> list[str]:
    """Word starts that survive Russian case endings: «манты» → «мант» (finds «мантов»), «черный» → «черн»."""
    out = []
    for w in re.findall(r"[а-яa-z]+", _norm(name)):
        n = len(w) - 1 if len(w) <= 5 else min(5, len(w) - 2)
        if n >= 3:
            out.append(re.escape(w[:n]))
    return out


def measure_in_text(text: str, name: str) -> tuple[float, str, str | None] | None:
    """(qty, unit, words) for the product `name` in the client's message: «горсть грецких орехов» → (1, «горсть»),
    «пять мантов» → (5, «штука»), «полтарелки борща» → (0.5, «тарелка»), «съела яблоко» → (1, «штука»).
    None when the text names no count for it and it is not a piece product."""
    t = _norm(text)
    for stem in _stems(name):
        near = rf"\s+(?:[а-яa-z]+\s+){{0,2}}?{stem}"
        m = re.search(rf"(?:\b({_NUM_RE})[\s-]*)?\b({_UNIT_RE}){near}", t)
        if m:
            unit = next(u for p, u in _UNITS if re.fullmatch(p, m.group(2)))
            return _number(m.group(1)), unit, t[m.start(): m.end(2)]
        m = re.search(rf"\bпол-?({_UNIT_RE}){near}", t)  # «полтарелки», «полстакана»
        if m:
            unit = next(u for p, u in _UNITS if re.fullmatch(p, m.group(1)))
            return 0.5, unit, t[m.start(): m.end(1)]
        m = re.search(rf"\b({_NUM_RE})\s+(?!{_WEIGHT_UNIT})(?:[а-яa-z]+\s+)?{stem}", t)  # «пять мантов»
        if m:
            return _number(m.group(1)), "штука", m.group(1)
    if _lookup(PIECE_GRAMS, name):  # a piece product named without a count is one piece: «съела яблоко»
        return 1.0, "штука", None
    return None
