"""The Kazakh voice eval: text metrics, derived policies and the golden file — no network."""

from __future__ import annotations

import collections
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _eval():
    spec = importlib.util.spec_from_file_location("evals_voice_eval", ROOT / "evals" / "voice_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evals_voice_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_golden_phrases_cover_every_intent_in_each_language():
    rows = [json.loads(l) for l in (ROOT / "evals" / "golden" / "voice_phrases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({r["id"] for r in rows}) == len(rows) == 40
    assert collections.Counter(r["lang"] for r in rows) == {"kk": 20, "mixed": 10, "ru": 10}
    for lang in ("kk", "mixed", "ru"):
        assert {r["expected_intent"] for r in rows if r["lang"] == lang} == {
            "meal_text", "question", "program_request", "escalate", "other"}
    assert all(r["voice"].startswith("kk-KZ") for r in rows if r["lang"] != "ru")
    assert all(r["voice"].startswith("ru-RU") for r in rows if r["lang"] == "ru")
    assert all(r["red_flag"] == (r["expected_intent"] == "escalate") for r in rows)


def test_normalize_and_error_rates():
    ve = _eval()
    assert ve.normalize("Сәлем, Ёлка!  Жаттығу-күні…") == "сәлем елка жаттығу күні"
    assert ve.edit_distance("kitten", "sitting") == 3
    assert ve.edit_distance(["a", "b"], ["a", "b"]) == 0
    c = ve.error_counts("Көп рахмет!", "коп рахмет")
    assert c == {"char_edits": 1, "chars": 10, "word_edits": 1, "words": 2}


def _row(pid, group, variant, text, pred, expected="question", detected=None, error=None, ms=100):
    return {"id": pid, "group": group, "variant": variant, "text": text, "detected": detected, "error": error,
            "ms": ms, "expected": expected, "pred": pred, "text_pred": expected,
            "char_edits": 0, "chars": 10, "word_edits": 0, "words": 2}


def test_derived_policies_pick_rows_without_new_calls(env):
    ve = _eval()
    phrases = [{"id": "a", "text": "Салмақ тастау үшін қанша ақуыз керек?", "lang": "kk"},
               {"id": "b", "text": "Сколько белка нужно?", "lang": "ru"},
               {"id": "c", "text": "Қанша су ішу керек?", "lang": "kk"}]
    by = {"ru": {"a": _row("a", "kk", "ru", "салмак", "other"), "b": _row("b", "ru", "ru", "сколько", "question"),
                 "c": _row("c", "kk", "ru", "канша", "other")},
          "kk": {"a": _row("a", "kk", "kk", "салмақ", "question"), "b": _row("b", "ru", "kk", "сколко", "other"),
                 "c": _row("c", "kk", "kk", "қанша", "question")},
          "auto": {"a": _row("a", "kk", "auto", "салмақ", "question", detected="kk"),
                   "b": _row("b", "ru", "auto", "сколько", "question", detected="ru"),
                   "c": _row("c", "kk", "auto", "kansha", "other", detected="tr")}}
    hint = {r["id"]: r for r in ve.derive("hint", by, phrases)}
    assert (hint["a"]["chosen"], hint["b"]["chosen"], hint["c"]["chosen"]) == ("kk", "ru", "kk")
    auto = {r["id"]: r for r in ve.derive("auto+ru", by, phrases)}
    assert (auto["a"]["chosen"], auto["b"]["chosen"], auto["c"]["chosen"]) == ("auto", "auto", "ru")
    assert auto["c"]["calls"] == 2 and auto["c"]["ms"] == 200 and auto["c"]["text"] == "канша"
    s = ve.summarize(list(auto.values()))
    assert s["stt_calls"] == 4 and s["intent_accuracy"] == round(100 * 2 / 3, 1)


def test_summary_counts_script_escalations_and_errors(env):
    ve = _eval()
    rows = [_row("a", "kk", "kk", "Көп рахмет", "other", expected="other"),
            _row("b", "kk", "ru", "Коп рахмет", "other", expected="other"),
            _row("c", "mixed", "kk", "тізем ауырады", "escalate", expected="escalate"),
            _row("d", "ru", "kk", "", None, error="stt 429")]
    rows[3].pop("pred")
    s = ve.summarize(rows)
    assert s["n"] == 4 and s["stt_errors"] == 1
    assert s["kk_script"] == round(100 * 2 / 3, 1)  # Kazakh letters kept in 2 of 3 kk/mixed transcripts
    assert s["intent_accuracy"] == 100.0 and s["escalation_recall"] == 100.0 and s["false_escalation_rate"] == 0.0
