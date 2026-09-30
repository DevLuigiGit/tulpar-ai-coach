"""BM25 as sparse vectors for hybrid search (RAG_HYBRID=on).

Qdrant keeps the sparse vectors next to the dense ones in the same collection and applies IDF itself
(`modifier=IDF`), so a document vector holds only the BM25 term-frequency part and a query vector is 1.0 per term.

The corpus is bilingual: Russian cards and rules, the English WHO PDF, Russian questions and — with
RAG_MULTI_QUERY=en — their English translations. One Snowball stemmer cannot serve both, so words are split by
script: Cyrillic words go to the Russian stemmer and stop-list, the rest (Latin words, numbers) to the English ones,
and the two sparse vectors are summed (a Russian and an English stem never share a hash in practice).

Two implementations, recorded in the collection name so an index is never queried with the other one's hashes:
  fastembed  `Qdrant/bm25` from fastembed: Snowball stemmers + stop-lists downloaded once from Hugging Face
             into AI_DATA_DIR/fastembed. Default when it loads.
  lite       no download and no extra package: the same tokenisation with a small built-in stop-list and a
             crude suffix-stripping stemmer for Russian and English. Used offline (tests, CI) or on RAG_BM25=lite.
"""

from __future__ import annotations

import logging
import re
import zlib
from collections import Counter
from pathlib import Path
from typing import Protocol

from qdrant_client import models

from ..config import get_settings

log = logging.getLogger(__name__)

K1, B = 1.2, 0.75
# Average number of indexed terms per chunk after stop-words, measured on the corpus of 2026-09-29 with 400-character
# WHO chunks (1476 chunks: 37.1 with the fastembed stemmers, 37.4 with lite). A constant, not a live corpus statistic,
# so a document added later is weighted exactly like the first build.
AVG_LEN = 37.0

_CYR = re.compile(r"[а-яё]", re.I)
_WORD = re.compile(r"\w+", re.U)


def _fold(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def split_scripts(text: str) -> tuple[str, str]:
    """(Cyrillic words, everything else) of a text, each as a space-joined string."""
    ru, other = [], []
    for w in _WORD.findall(_fold(text)):
        (ru if _CYR.search(w) else other).append(w)
    return " ".join(ru), " ".join(other)


class SparseEncoder(Protocol):
    id: str

    def embed_documents(self, texts: list[str]) -> list[models.SparseVector]: ...

    def embed_query(self, text: str) -> models.SparseVector: ...


def _merge(parts: list[dict[int, float]]) -> models.SparseVector:
    acc: dict[int, float] = {}
    for p in parts:
        for i, v in p.items():
            acc[i] = acc.get(i, 0.0) + float(v)
    idx = sorted(acc)
    return models.SparseVector(indices=idx, values=[acc[i] for i in idx])


class FastembedBm25:
    """fastembed `Qdrant/bm25`, one model per language, words routed by script (see the module docstring)."""

    id = "bm25fe"

    def __init__(self, cache_dir: Path | None = None, model_path: Path | None = None):
        from fastembed.sparse.bm25 import Bm25

        kw = {"cache_dir": str(cache_dir or Path(get_settings().ai_data_dir) / "fastembed"), "avg_len": AVG_LEN,
              "k": K1, "b": B}
        if model_path is not None:  # a local copy of the model folder (stop-lists): no download at all
            kw["specific_model_path"] = str(model_path)
        self.ru = Bm25("Qdrant/bm25", language="russian", **kw)
        self.en = Bm25("Qdrant/bm25", language="english", **kw)

    @staticmethod
    def _as_dict(emb) -> dict[int, float]:
        return {int(i): float(v) for i, v in zip(emb.indices, emb.values)}

    def embed_documents(self, texts: list[str]) -> list[models.SparseVector]:
        split = [split_scripts(t) for t in texts]
        ru = list(self.ru.embed([r for r, _ in split]))
        en = list(self.en.embed([e for _, e in split]))
        return [_merge([self._as_dict(a), self._as_dict(b)]) for a, b in zip(ru, en)]

    def embed_query(self, text: str) -> models.SparseVector:
        r, e = split_scripts(text)
        [qr] = list(self.ru.query_embed(r))
        [qe] = list(self.en.query_embed(e))
        return _merge([self._as_dict(qr), self._as_dict(qe)])


# ── lite: no download, no dependency ─────────────────────────────────────────
RU_STOP = set("""
а без более бы был была были было быть в вам вас весь во вот все всего всех вы где да даже для до его ее если есть еще
же за здесь и из или им их к как ко когда кто ли либо мне может мы на над надо наш не него нее нет ни них но ну о об
однако он она они оно от очень по под при с со так также такой там те тем то того тоже той только том ты у уже хотя
чем что чтобы чье чья эта эти это я ей ему ним нем
""".split())
EN_STOP = set("""
a about above after again all am an and any are as at be been before being below between both but by can could did do
does doing down during each few for from further had has have having he her here hers him his how i if in into is it
its itself just me more most my no nor not now of off on once only or other our out over own same she should so some
such than that the their them then there these they this those through to too under until up very was we were what
when where which while who whom why will with would you your
""".split())
RU_SUFFIXES = sorted("""
иями ями ами ией иям ием иях ого его ому ему ыми ими ая яя ую юю ое ее ые ие ый ий ой ей ых их ом ем ам ям ах ях ов ев
ия ья ие ье ию ью а я о е ы и у ю ь й ть ться тся ешь ет ем ете ут ют ишь ит им ите ат ят ал ала али ало ил ила или
ило ость ости ение ения ений ением ениям ание ания аний анием
""".split(), key=len, reverse=True)
EN_SUFFIXES = sorted("ational tional ations ation ments ment ness ities ity ies ing edly ed ly es s".split(), key=len,
                     reverse=True)


def lite_stem(word: str) -> str:
    """Strip one known suffix, keeping a stem of at least 3 letters (Russian) / 3 letters (English)."""
    suffixes = RU_SUFFIXES if _CYR.search(word) else EN_SUFFIXES
    if word.isdigit():
        return word
    for suf in suffixes:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def lite_terms(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(_fold(text)):
        if w in RU_STOP or w in EN_STOP or len(w) > 40:
            continue
        out.append(lite_stem(w))
    return out


def _token_id(term: str) -> int:
    return zlib.crc32(term.encode())


class LiteBm25:
    id = "bm25lite"

    def embed_documents(self, texts: list[str]) -> list[models.SparseVector]:
        out = []
        for t in texts:
            terms = lite_terms(t)
            tf = Counter(terms)
            norm = K1 * (1 - B + B * len(terms) / AVG_LEN)
            out.append(_merge([{_token_id(w): n * (K1 + 1) / (n + norm) for w, n in tf.items()}]))
        return out

    def embed_query(self, text: str) -> models.SparseVector:
        return _merge([{_token_id(w): 1.0 for w in set(lite_terms(text))}])


_encoders: dict[str, SparseEncoder] = {}


def get_sparse_encoder(kind: str | None = None) -> SparseEncoder:
    """RAG_BM25=auto|fastembed|lite. auto falls back to lite (with a warning) when fastembed cannot load its model."""
    s = get_settings()
    kind = (kind or s.rag_bm25 or "auto").lower()
    key = f"{kind}:{s.ai_data_dir}"
    if key in _encoders:
        return _encoders[key]
    enc: SparseEncoder
    if kind == "lite":
        enc = LiteBm25()
    else:
        try:
            enc = FastembedBm25()
        except Exception:
            if kind == "fastembed":
                raise
            log.warning("fastembed BM25 unavailable: using the built-in lite BM25", exc_info=True)
            enc = LiteBm25()
    _encoders[key] = enc
    return enc


def reset_sparse_encoders() -> None:
    _encoders.clear()
