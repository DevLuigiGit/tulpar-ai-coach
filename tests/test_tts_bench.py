"""The voice benchmark: answer selection, stats, the results file — fake TTS, no network."""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

import pytest

from tulpar_ai import tts

ROOT = Path(__file__).resolve().parents[1]
MP3 = b"\xff\xf3" + b"\x00" * 5998  # 6000 bytes = 1.0 s at 48 kbit/s


def _bench():
    spec = importlib.util.spec_from_file_location("evals_tts_bench", ROOT / "evals" / "tts_bench.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evals_tts_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


def _source(path: Path) -> Path:
    rows = [{"id": "q1", "answered": True, "answer": "Ходите **150 минут** в неделю [1]."},
            {"id": "q2", "answered": False, "answer": "Недостаточно данных."},
            {"id": "q3", "answered": True, "answer": "  "},
            {"id": "q1", "answered": True, "answer": "Повтор того же вопроса."},
            {"id": "q4", "answered": True, "answer": "Белок: 1,6 г на кг массы тела [2]."},
            {"id": "q5", "answered": True, "answer": "Спите 7–9 часов."}]
    path.write_text(json.dumps({"summary": {}, "rows": rows}, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def fake_tts():
    def fake(text: str, voice: str) -> bytes:
        if "Белок" in text:
            raise TimeoutError("edge-tts timeout")
        return MP3

    tts.set_fake(fake)
    yield
    tts.set_fake(None)


def test_pick_answers_skips_unanswered_empty_and_repeats(tmp_path):
    picked = _bench().pick_answers(_source(tmp_path / "qa.json"), n=2)
    assert [a["id"] for a in picked] == ["q1", "q4"]
    assert picked[0]["answer"].startswith("Ходите")


async def test_bench_measures_size_and_latency_and_keeps_failures(env, fake_tts, tmp_path):
    b = _bench()
    answers = b.pick_answers(_source(tmp_path / "qa.json"), n=10)
    res = await b.bench(answers, repeats=2, audio_dir=tmp_path / "audio")
    s, rows = res["summary"], res["rows"]
    assert (s["n"], s["ok"], s["errors"]) == (6, 4, 2)
    # Stats are over successful calls only: a timeout must not drag the size median to zero.
    assert s["bytes_p50"] == s["bytes_min"] == s["bytes_max"] == 6000
    assert s["audio_s_p50"] == 1.0 and s["kb_p50"] == round(6000 / 1024, 1)
    assert s["p50_ms"] is not None and s["min_ms"] <= s["p50_ms"] <= s["max_ms"]
    assert s["voice"] == "ru-RU-SvetlanaNeural" and s["format"].startswith("mp3")
    failed = [r for r in rows if r["error"]]
    assert {r["id"] for r in failed} == {"q4"} and all(r["bytes"] == 0 for r in failed)
    assert "TimeoutError" in failed[0]["error"]
    # Citations and markdown are gone before synthesis, so the spoken text is shorter.
    assert all(r["speech_chars"] < r["answer_chars"] for r in rows if r["id"] == "q1")
    assert sorted(p.name for p in (tmp_path / "audio").iterdir()) == ["q1_0.mp3", "q1_1.mp3", "q5_0.mp3", "q5_1.mp3"]


async def test_main_writes_results_json_and_summary(env, fake_tts, tmp_path, monkeypatch):
    b = _bench()
    results = tmp_path / "results"
    monkeypatch.setattr(b.run, "RESULTS", results)
    monkeypatch.setattr(sys, "argv", ["tts_bench.py", "--source", str(_source(tmp_path / "qa.json")),
                                      "--n", "3", "--repeats", "1", "--tag", "t"])
    assert await b.main() == 0
    saved = json.loads((results / "tts_t.json").read_text(encoding="utf-8"))
    assert saved["summary"]["bytes_p50"] == 6000 and saved["summary"]["n"] == 3
    table = (results / "SUMMARY.md").read_text(encoding="utf-8")
    assert "## tts_t" in table and "| bytes_p50 |" in table


def test_committed_tts_results_back_the_reported_numbers():
    """The latency and size figures in the docs must be reproducible from the committed run."""
    saved = json.loads((ROOT / "evals" / "results" / "tts.json").read_text(encoding="utf-8"))
    s, rows = saved["summary"], saved["rows"]
    ok = [r for r in rows if not r["error"]]
    assert s["n"] == len(rows) >= 5 and s["ok"] == len(ok) > 0
    assert s["p50_ms"] == round(statistics.median(r["ms"] for r in ok), 1)
    assert s["bytes_p50"] == round(statistics.median(r["bytes"] for r in ok), 1)
    assert (s["bytes_min"], s["bytes_max"]) == (min(r["bytes"] for r in ok), max(r["bytes"] for r in ok))
    assert all(r["audio_s"] == round(r["bytes"] * 8 / 48_000, 2) for r in ok)
    assert "## tts —" in (ROOT / "evals" / "results" / "SUMMARY.md").read_text(encoding="utf-8")
