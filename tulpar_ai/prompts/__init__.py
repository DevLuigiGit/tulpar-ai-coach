"""Prompts are versioned files: `<name>.v<N>.md`. The active version per prompt is set in PROMPT_VERSIONS
(or env PROMPT_<NAME>=vN) so the defense can show v1 → v2 with the eval numbers that motivated it."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

DIR = Path(__file__).resolve().parent
PROMPT_VERSIONS = {"route": "v1", "rewrite": "v1", "answer": "v1", "vision": "v1", "meal_text": "v1", "draft": "v1",
                   "judge_faithfulness": "v1", "judge_correctness": "v1"}


@lru_cache
def prompt(name: str, version: str | None = None) -> str:
    v = version or os.environ.get(f"PROMPT_{name.upper()}") or PROMPT_VERSIONS[name]
    return (DIR / f"{name}.{v}.md").read_text(encoding="utf-8").strip()


def active_version(name: str) -> str:
    return os.environ.get(f"PROMPT_{name.upper()}") or PROMPT_VERSIONS[name]
