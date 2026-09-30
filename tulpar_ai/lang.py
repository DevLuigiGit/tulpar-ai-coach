"""Is this text Kazakh? One cheap rule shared by speech recognition (stt.py) and the voice of replies (tts.py).

Kazakh Cyrillic has nine letters Russian does not (ә ғ қ ң ө ұ ү һ і); about half of Kazakh words contain one.
Counting WORDS rather than letters keeps a Russian reply that quotes one Kazakh dish name Russian.
"""

from __future__ import annotations

import re

from .config import get_settings

KK_LETTERS = frozenset("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
_WORD = re.compile(r"[^\W\d_]+", re.U)


def kk_word_share(text: str) -> float:
    """Share of words with a Kazakh-only letter: 0.3–0.7 for Kazakh sentences, 0 for Russian."""
    words = _WORD.findall(text or "")
    return sum(any(ch in KK_LETTERS for ch in w) for w in words) / len(words) if words else 0.0


def is_kazakh(text: str, min_share: float | None = None) -> bool:
    return kk_word_share(text) >= (get_settings().kk_min_word_share if min_share is None else min_share)
