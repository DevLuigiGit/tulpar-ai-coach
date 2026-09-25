"""Voice replies benchmark: latency and MP3 size of tts.synthesize on real coach answers.

    python evals/tts_bench.py [--source evals/results/qa_jina_answer_v3.json] [--n 10] [--repeats 2] [--audio-dir DIR]
The answers are taken from a saved QA run (first N answered rows, file order), so the input is reproducible and
costs no LLM calls. Calls go one by one through the production path (cleanup → PII mask → edge-tts).
Writes evals/results/tts[_tag].json and appends a table to evals/results/SUMMARY.md, like evals/run.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import run  # noqa: E402

DEFAULT_SOURCE = ROOT / "evals" / "results" / "qa_jina_answer_v3.json"
# edge-tts default output is audio-24khz-48kbitrate-mono-mp3: constant bitrate, so size gives the duration.
BITRATE_BPS = 48_000
run.KEYS["tts"] = ["n", "ok", "errors", "voice", "p50_ms", "p95_ms", "min_ms", "max_ms",
                   "bytes_p50", "bytes_min", "bytes_max", "audio_s_p50", "speech_chars_p50", "truncated"]


def pick_answers(source: Path, n: int) -> list[dict]:
    """First `n` answered rows with non-empty text; a repeated id (from --repeats in QA) counts once."""
    rows = json.loads(source.read_text(encoding="utf-8"))["rows"]
    seen, out = set(), []
    for r in rows:
        text = (r.get("answer") or "").strip()
        if not r.get("answered") or not text or r["id"] in seen:
            continue
        seen.add(r["id"])
        out.append({"id": r["id"], "answer": text})
        if len(out) == n:
            break
    return out


def summarize(rows: list[dict], voice: str) -> dict:
    ok = [r for r in rows if not r.get("error")]
    ms = [r["ms"] for r in ok]
    size = [r["bytes"] for r in ok]
    stat = lambda xs, f: round(f(xs), 1) if xs else None  # noqa: E731
    return {"n": len(rows), "ok": len(ok), "errors": len(rows) - len(ok), "voice": voice,
            "p50_ms": stat(ms, statistics.median), "p95_ms": run.p95(ms) if ms else None,
            "min_ms": stat(ms, min), "max_ms": stat(ms, max),
            "bytes_p50": stat(size, statistics.median), "bytes_min": min(size) if size else None,
            "bytes_max": max(size) if size else None, "kb_p50": stat([b / 1024 for b in size], statistics.median),
            "audio_s_p50": stat([r["audio_s"] for r in ok], statistics.median),
            "speech_chars_p50": stat([r["speech_chars"] for r in rows], statistics.median),
            "truncated": sum(r["truncated"] for r in rows)}


async def bench(answers: list[dict], repeats: int = 1, audio_dir: Path | None = None) -> dict:
    from tulpar_ai import tts
    from tulpar_ai.config import get_settings

    s = get_settings()
    rows = []
    for rep in range(repeats):
        for a in answers:
            spoken = tts.speech_text(a["answer"], s.tts_max_chars)
            row = {"id": a["id"], "rep": rep, "answer_chars": len(a["answer"]), "speech_chars": len(spoken),
                   "truncated": spoken.endswith(tts.TRUNCATED_TAIL), "voice": tts.pick_voice(spoken)}
            t0 = time.perf_counter()
            try:
                audio = await tts.synthesize(a["answer"])
            except Exception as e:  # a failed call is a data point, not the end of the run
                row.update(ms=round((time.perf_counter() - t0) * 1000, 1), bytes=0, audio_s=0.0,
                           error=f"{type(e).__name__}: {e}"[:200])
            else:
                row.update(ms=round((time.perf_counter() - t0) * 1000, 1), bytes=len(audio),
                           audio_s=round(len(audio) * 8 / BITRATE_BPS, 2), error=None)
                if audio_dir is not None:
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    (audio_dir / f"{a['id']}_{rep}.mp3").write_bytes(audio)
            rows.append(row)
    voice = rows[0]["voice"] if rows else s.tts_voice
    return {"summary": {**summarize(rows, voice), "rate": s.tts_rate, "max_chars": s.tts_max_chars,
                        "format": "mp3 24kHz 48kbit/s mono", "repeats": repeats},
            "rows": rows}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--audio-dir")
    ap.add_argument("--tag", default="")
    ap.add_argument("--trace", action="store_true")
    return ap


async def main() -> int:
    args = parser().parse_args()
    run.boot_settings(Namespace(trace=args.trace))
    source = Path(args.source)
    answers = pick_answers(source, args.n)
    res = await bench(answers, args.repeats, Path(args.audio_dir) if args.audio_dir else None)
    res["summary"]["source"] = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else source.name
    name = "tts" + (f"_{args.tag}" if args.tag else "")
    run.save(name, res, "tts", [(args.tag or "default", res["summary"])])
    return 0 if res["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
