"""Kazakh voice: speech recognition quality and end-to-end routing on voice notes.

    python evals/voice_eval.py [--variants ru,kk,auto] [--audio-dir data/voice-kk/audio] [--format ogg|mp3]
                               [--repeats 3] [--no-router] [--reroute] [--limit N] [--sleep 3] [--tag T]
    python evals/voice_eval.py --tts-check      # which edge-tts voice is understood better → voice_kk_tts.json

Steps, each cached on disk so a re-run does not pay twice for the same call:
  1. synthesize every phrase of evals/golden/voice_phrases.jsonl with its edge-tts voice (kk-KZ Aigul/Daulet for
     Kazakh and mixed phrases, ru-RU Svetlana/Dmitry for the Russian controls) and convert it to OGG/Opus mono
     like a Telegram voice note;
  2. transcribe it with Groq Whisper through tulpar_ai/stt.py under each language variant
     (ru = production before this change, kk, auto = no language, Whisper detects it);
  3. route the source text and every transcript with the live router (precheck + route, as `evals/run.py router`),
     `--repeats` times each, majority vote (identical transcripts share verdicts).
Policies that need no extra calls are derived from the same rows: `hint` (stt.choose_language on the client's own
text) and `auto+ru` (auto, re-asked with ru when Whisper detects neither ru nor kk).

Synthesized speech is cleaner than a real voice note from a gym: every number here is an upper bound.
Audio lives in data/ (not in git). Writes evals/results/voice_kk[_tag].json, appends tables to SUMMARY.md.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import run  # noqa: E402

GROUPS = ("kk", "mixed", "ru")
VARIANTS = {"ru": "ru", "kk": "kk", "auto": None}
run.KEYS["voice"] = ["n", "cer", "wer", "kk_script", "intent_accuracy", "escalation_recall", "false_escalation_rate",
                     "agree_with_text", "stability", "stt_calls", "stt_p50_ms", "stt_errors"]


# ── text metrics ─────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    t = (text or "").lower().replace("ё", "е")
    t = re.sub(r"[^\w\s]|_", " ", t)
    return " ".join(t.split())


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_counts(ref: str, hyp: str) -> dict:
    r, h = normalize(ref), normalize(hyp)
    return {"char_edits": edit_distance(r, h), "chars": len(r),
            "word_edits": edit_distance(r.split(), h.split()), "words": len(r.split())}


# ── audio ────────────────────────────────────────────────────────────────────
async def synthesize(text: str, voice: str) -> bytes:
    import edge_tts

    buf = bytearray()
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    if not buf:
        raise RuntimeError("edge-tts returned no audio")
    return bytes(buf)


def to_ogg(mp3: Path, ogg: Path) -> None:
    # Telegram voice notes are OGG/Opus mono; 32 kbit/s voip is close to what the app records.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ac", "1", "-ar", "48000",
                    "-c:a", "libopus", "-b:a", "32k", "-application", "voip", str(ogg)], check=True)


async def prepare_audio(phrases: list[dict], audio_dir: Path, fmt: str) -> dict[str, Path]:
    audio_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "ogg" and not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is needed for --format ogg (or pass --format mp3)")
    out = {}
    for p in phrases:
        key = hashlib.sha1(f"{p['text']}|{p['voice']}".encode()).hexdigest()[:8]
        mp3 = audio_dir / f"{p['id']}_{key}.mp3"
        if not mp3.exists():
            for attempt in range(6):  # the edge endpoint drops a stream now and then (NoAudioReceived)
                try:
                    mp3.write_bytes(await synthesize(p["text"], p["voice"]))
                    break
                except Exception:
                    if attempt == 5:
                        raise
                    await asyncio.sleep(2 + 2 * attempt)
        path = mp3
        if fmt == "ogg":
            path = mp3.with_suffix(".ogg")
            if not path.exists():
                to_ogg(mp3, path)
        out[p["id"]] = path
    return out


# ── cached calls ─────────────────────────────────────────────────────────────
class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        self.data[key] = value
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")


async def transcribe(path: Path, language: str | None, pause: float) -> dict:
    from tulpar_ai import stt

    audio = path.read_bytes()
    for attempt in range(6):
        t0 = time.perf_counter()
        try:
            res = await stt.transcribe_detailed(audio, filename=path.name, language=language)
            return {**res, "ms": round((time.perf_counter() - t0) * 1000), "error": None}
        except Exception as e:  # 429 from the shared key, 5xx, network: back off and retry
            msg = f"{type(e).__name__}: {e}"[:200]
            retryable = "stt 429" in msg or "stt 5" in msg or not msg.startswith("RuntimeError")
            if not retryable or attempt == 5:
                return {"text": "", "language": None, "ms": round((time.perf_counter() - t0) * 1000), "error": msg}
            await asyncio.sleep(min(60, pause * 4 * 2 ** attempt))


async def route_text(text: str) -> dict:
    from tulpar_ai import llm
    from tulpar_ai.graph.chat import precheck, route

    for attempt in range(3):
        with llm.record() as calls:
            st = {"text": text}
            st.update(await precheck(st))
            out = await route(st)
        # route() falls back to regex rules when no provider answers: that is not the live router, retry
        if not str(out.get("reason", "")).startswith("heuristic") or attempt == 2:
            model = f"{calls[-1]['provider']}:{calls[-1]['model']}" if calls else "rules"
            return {"intent": out["intent"], "reason": out.get("reason", ""), "model": model}
        await asyncio.sleep(10)


# ── metrics ──────────────────────────────────────────────────────────────────
def majority(verdicts: list[dict]) -> dict:
    votes = [v["intent"] for v in verdicts]
    intent = max(votes, key=lambda x: (votes.count(x), -votes.index(x)))  # a tie goes to the earliest verdict
    first = next(v for v in verdicts if v["intent"] == intent)
    return {"intent": intent, "all": votes, "stable": len(set(votes)) == 1, "reason": first["reason"],
            "model": first["model"]}


def summarize(rows: list[dict]) -> dict:
    from tulpar_ai.stt import kk_word_share

    ok = [r for r in rows if not r["error"]]
    chars = sum(r["chars"] for r in ok) or 1
    words = sum(r["words"] for r in ok) or 1
    routed = [r for r in rows if r.get("pred")]
    esc = [r for r in routed if r["expected"] == "escalate"]
    non = [r for r in routed if r["expected"] != "escalate"]
    kk_rows = [r for r in ok if r["group"] != "ru"]
    ms = [r["ms"] for r in rows if r.get("ms") is not None]
    out = {"n": len(rows), "cer": round(100 * sum(r["char_edits"] for r in ok) / chars, 1),
           "wer": round(100 * sum(r["word_edits"] for r in ok) / words, 1),
           # Kazakh and mixed phrases only: the transcript keeps Kazakh letters (not transliterated or translated)
           "kk_script": run.pct(kk_word_share(r["text"]) > 0 for r in kk_rows) if kk_rows else None,
           "stt_calls": sum(r.get("calls", 1) for r in rows), "stt_errors": len(rows) - len(ok),
           "stt_p50_ms": statistics.median(ms) if ms else None}
    if routed:
        out |= {"intent_accuracy": run.pct(r["pred"] == r["expected"] for r in routed),
                "escalation_recall": run.pct(r["pred"] == "escalate" for r in esc) if esc else None,
                "false_escalation_rate": run.pct(r["pred"] == "escalate" for r in non) if non else None,
                "agree_with_text": run.pct(r["pred"] == r["text_pred"] for r in routed),
                "stability": run.pct(r.get("stable", True) for r in routed)}
    langs = [r["detected"] for r in rows if r.get("detected")]
    if langs:
        out["detected"] = {k: langs.count(k) for k in sorted(set(langs))}
    return out


def derive(policy: str, by_variant: dict[str, dict[str, dict]], phrases: list[dict]) -> list[dict]:
    """Rows of a policy built from already measured variants (no extra calls)."""
    from tulpar_ai.stt import AUTO_ACCEPT, choose_language

    rows = []
    for p in phrases:
        if policy == "hint":
            # Assumption: the client types the way they speak; production looks at their recent messages.
            lang = choose_language("hint", recent_texts=[p["text"]])
            rows.append({**by_variant[lang][p["id"]], "policy": policy, "chosen": lang})
        elif policy == "auto+ru":
            auto = by_variant["auto"][p["id"]]
            if auto["error"] or auto.get("detected") in AUTO_ACCEPT:
                rows.append({**auto, "policy": policy, "chosen": "auto"})
            else:  # production would ask again with ru: two calls for this note
                rows.append({**by_variant["ru"][p["id"]], "policy": policy, "chosen": "ru", "calls": 2,
                             "ms": (auto["ms"] or 0) + (by_variant["ru"][p["id"]]["ms"] or 0)})
    return rows


# ── voice of replies: which edge-tts voice is understood better ──────────────
# Reply-like texts that contain Kazakh letters. `lang` is the language most of the words are in; the round trip
# (synthesize with each voice → Whisper) is scored against the text, Whisper told that language (auto for mixed).
TTS_TEXTS = [
    {"id": "t1", "lang": "kk", "text": "Тренер ответил: Сәлеметсіз бе! Ертеңгі жаттығуды жеңілдетеміз, аяқ күнін алып тастаймын."},
    {"id": "t2", "lang": "kk", "text": "Тренер ответил: Жарайсыз! Осы аптада кардионы екі рет қосамыз, әр жолы жиырма минуттан."},
    {"id": "t3", "lang": "kk", "text": "Тренер ответил: Тізеңіз ауырса, жаттығуды тоқтатып, дәрігерге көрініңіз."},
    {"id": "t4", "lang": "ru", "text": "Нашёл 1 поз., всего ≈ 480 ккал:\n• плов: 200 г ≈ 480 ккал\n• «сұлы ботқасы» — нет в "
                                      "справочнике, можно добавить вручную\n• «құрт» — нет в справочнике, можно добавить вручную"},
    {"id": "t5", "lang": "ru", "text": "Тренер ответил: Добрый день! Я поговорила с Айгүл Қасымқызы, с понедельника "
                                      "вы занимаетесь у неё по вторникам и четвергам."},
    {"id": "t6", "lang": "ru", "text": "Тренер ответил: Хорошо, Нұрсұлтан, на этой неделе уберём приседания и добавим "
                                      "растяжку после каждой тренировки."},
    {"id": "t7", "lang": "mixed", "text": "Тренер ответил: Жақсы, кардио қосамыз, но сначала неделю лёгкая нагрузка."},
    {"id": "t8", "lang": "mixed", "text": "Тренер ответил: Рахмет за отчёт! Ертең жаттығу жоқ, отдыхайте и пейте больше воды."},
]
TTS_VOICES = {"ru": "ru-RU-SvetlanaNeural", "kk": "kk-KZ-AigulNeural"}


def old_pick_voice(text: str) -> str:
    """TTS_VOICE_RULE=letters, the rule before this change: three Kazakh letters anywhere → the Kazakh voice."""
    from tulpar_ai.lang import KK_LETTERS

    return "kk" if sum(ch in KK_LETTERS for ch in text) >= 3 else "ru"


async def tts_voice_check(audio_dir: Path, pause: float) -> dict:
    from tulpar_ai import tts
    from tulpar_ai.config import get_settings
    from tulpar_ai.lang import is_kazakh, kk_word_share
    from tulpar_ai.pii import mask

    s = get_settings()
    cache = Cache(audio_dir / "tts_check_cache.json")
    rows = []
    for t in TTS_TEXTS:
        spoken = mask(tts.speech_text(t["text"], s.tts_max_chars))
        new = "kk" if is_kazakh(spoken) else "ru"  # TTS_VOICE_RULE=words
        row = {"id": t["id"], "lang": t["lang"], "text": spoken, "old_voice": old_pick_voice(spoken), "new_voice": new,
               "kk_word_share": round(kk_word_share(spoken), 2), "voices": {}}
        for v, voice in TTS_VOICES.items():
            key = f"{t['id']}|{voice}|{hashlib.sha1(spoken.encode()).hexdigest()[:8]}|{s.stt_model}"
            hit = cache.get(key)
            if hit is None:
                t0 = time.perf_counter()
                audio = await synthesize(spoken, voice)
                synth_ms = round((time.perf_counter() - t0) * 1000)
                path = audio_dir / f"tts_{t['id']}_{v}.mp3"
                path.write_bytes(audio)
                heard = await transcribe(path, None if t["lang"] == "mixed" else t["lang"], pause)
                hit = {"synth_ms": synth_ms, "bytes": len(audio), "heard": heard["text"], "stt_error": heard["error"],
                       **error_counts(spoken, heard["text"])}
                cache.put(key, hit)
                await asyncio.sleep(pause)
            row["voices"][v] = {**hit, "cer": round(100 * hit["char_edits"] / max(1, hit["chars"]), 1)}
        better = min(row["voices"], key=lambda v: row["voices"][v]["cer"])
        row |= {"better_voice": better, "old_ok": row["old_voice"] == better, "new_ok": new == better}
        rows.append(row)
    kk_ms = [r["voices"]["kk"]["synth_ms"] for r in rows]
    summary = {"n": len(rows), "old_rule_matches_better_voice": sum(r["old_ok"] for r in rows),
               "new_rule_matches_better_voice": sum(r["new_ok"] for r in rows),
               "kk_voice_synth_p50_ms": statistics.median(kk_ms), "kk_voice_synth_max_ms": max(kk_ms),
               "cer_by_lang": {lg: {v: round(100 * sum(r["voices"][v]["char_edits"] for r in rows if r["lang"] == lg)
                                          / max(1, sum(r["voices"][v]["chars"] for r in rows if r["lang"] == lg)), 1)
                                    for v in TTS_VOICES} for lg in ("kk", "ru", "mixed")}}
    return {"summary": summary, "rows": rows}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts-check", action="store_true", help="only the reply-voice round trip → voice_kk_tts.json")
    ap.add_argument("--golden", default=str(run.GOLDEN / "voice_phrases.jsonl"))
    ap.add_argument("--variants", default="ru,kk,auto")
    ap.add_argument("--audio-dir", default=str(ROOT / "data" / "voice-kk" / "audio"))
    ap.add_argument("--format", choices=["ogg", "mp3"], default="ogg")
    ap.add_argument("--no-router", dest="router", action="store_false")
    ap.add_argument("--repeats", type=int, default=3, help="live router verdicts per text, majority vote")
    ap.add_argument("--reroute", action="store_true", help="ignore cached router verdicts")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=3.0, help="pause between Whisper calls (shared key: ~20 rpm)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--trace", action="store_true")
    return ap


async def main() -> int:
    args = parser().parse_args()
    run.boot_settings(Namespace(trace=args.trace))
    from tulpar_ai.config import get_settings
    from tulpar_ai.prompts import active_version

    s = get_settings()
    if args.tts_check:
        res = await tts_voice_check(Path(args.audio_dir), args.sleep)
        name = "voice_kk_tts" + (f"_{args.tag}" if args.tag else "")
        (run.RESULTS / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
        return 0
    phrases = [json.loads(l) for l in Path(args.golden).read_text(encoding="utf-8").splitlines() if l.strip()]
    phrases = phrases[: args.limit or None]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    audio_dir = Path(args.audio_dir)
    files = await prepare_audio(phrases, audio_dir, args.format)
    stt_cache = Cache(audio_dir / "stt_cache.json")
    route_cache = Cache(audio_dir / "route_cache.json")
    route_tag = f"route.{active_version('route')}|{s.route_models}"

    fresh: set[str] = set()

    async def routed(text: str) -> dict:
        """Majority of `--repeats` live verdicts. Cached per text: an identical transcript in two variants gets the
        same verdicts, and a re-run only tops up what is missing (`--reroute` starts over)."""
        key = f"{route_tag}|{text}"
        got = route_cache.get(key) or []
        if args.reroute and key not in fresh:
            got = []
        fresh.add(key)
        got = [got] if isinstance(got, dict) else list(got)
        if len(got) < args.repeats:
            while len(got) < args.repeats:
                got.append(await route_text(text))
            route_cache.put(key, got)
        return majority(got[: args.repeats])

    text_pred, text_votes = {}, {}
    if args.router:
        for p in phrases:
            r = await routed(p["text"])
            text_pred[p["id"]], text_votes[p["id"]] = r["intent"], r["all"]

    by_variant: dict[str, dict[str, dict]] = {}
    for v in variants:
        by_variant[v] = {}
        for p in phrases:
            path = files[p["id"]]
            key = f"{path.name}|{v}|{s.stt_model}"
            res = stt_cache.get(key)
            if res is None or res.get("error"):
                res = await transcribe(path, VARIANTS[v], args.sleep)
                stt_cache.put(key, res)
                await asyncio.sleep(args.sleep)
            row = {"id": p["id"], "group": p["lang"], "voice": p["voice"], "variant": v, "expected": p["expected_intent"],
                   "ref": p["text"], "text": res["text"], "detected": res["language"] if v == "auto" else None,
                   "ms": res["ms"], "error": res["error"], **error_counts(p["text"], res["text"])}
            if args.router and not res["error"]:
                r = await routed(res["text"])
                row |= {"pred": r["intent"], "all": r["all"], "stable": r["stable"], "reason": r["reason"],
                        "router_model": r["model"], "text_pred": text_pred[p["id"]]}
            by_variant[v][p["id"]] = row

    policies = dict(by_variant)
    if {"ru", "kk"} <= set(variants):
        policies["hint"] = {r["id"]: r for r in derive("hint", by_variant, phrases)}
    if {"ru", "auto"} <= set(variants):
        policies["auto+ru"] = {r["id"]: r for r in derive("auto+ru", by_variant, phrases)}

    summary, labelled = {"n_phrases": len(phrases), "format": args.format, "stt_model": s.stt_model,
                         "route_prompt": active_version("route"), "route_models": s.route_models,
                         "route_repeats": args.repeats,
                         "by_group": {g: sum(p["lang"] == g for p in phrases) for g in GROUPS}}, []
    if args.router:
        text_rows = [{"group": p["lang"], "expected": p["expected_intent"], "pred": text_pred[p["id"]],
                      "stable": len(set(text_votes[p["id"]])) == 1, "text_pred": text_pred[p["id"]], "error": None,
                      "chars": 0, "words": 0, "char_edits": 0, "word_edits": 0, "text": p["text"], "ms": None}
                     for p in phrases]
        summary["text"] = {g: summarize([r for r in text_rows if r["group"] == g]) for g in GROUPS}
        for g in GROUPS:
            labelled.append((f"text (no STT) / {g}", summary["text"][g]))
    summary["policies"] = {}
    for name, rows in policies.items():
        rr = list(rows.values())
        summary["policies"][name] = {"all": summarize(rr), **{g: summarize([r for r in rr if r["group"] == g]) for g in GROUPS}}
        for g in GROUPS:
            labelled.append((f"{name} / {g}", summary["policies"][name][g]))
    payload = {"summary": summary, "rows": [r for v in by_variant.values() for r in v.values()],
               "text_routes": {k: {"pred": text_pred[k], "all": text_votes[k]} for k in text_pred}}
    run.save("voice_kk" + (f"_{args.tag}" if args.tag else ""), payload, "voice", labelled)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
