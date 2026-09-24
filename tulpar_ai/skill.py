"""The Skill `skills/tulpar-program-builder` is the single source of truth for program rules.

The graph loads SKILL.md + references into the draft prompt and imports the SAME validator script that
Claude runs from the Skill. One artifact, two consumers.
"""

from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

from .config import ROOT

SKILL_DIR = ROOT / "skills" / "tulpar-program-builder"


@lru_cache
def validator():
    spec = importlib.util.spec_from_file_location("tulpar_validate_plan", SKILL_DIR / "scripts" / "validate_plan.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@lru_cache
def catalog() -> dict[str, dict]:
    return validator().load_catalog()


@lru_cache
def instructions() -> str:
    """SKILL.md body (without frontmatter) + the references the draft step needs."""
    body = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    if body.startswith("---"):
        body = body.split("---", 2)[2]
    parts = [body.strip()]
    for ref in ("progression.md", "contraindications.md"):
        p = SKILL_DIR / "references" / ref
        if p.exists():
            parts.append(f"## reference: {ref}\n" + p.read_text(encoding="utf-8").strip())
    schema = SKILL_DIR / "references" / "plan_ops.schema.json"
    if schema.exists():
        parts.append("## reference: plan_ops.schema.json\n" + json.dumps(json.loads(schema.read_text(encoding="utf-8")), ensure_ascii=False))
    return "\n\n".join(parts)
