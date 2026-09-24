"""Masking before anything leaves for an LLM provider or a trace.

Phones and e-mails are caught by pattern. Names cannot be caught by a regex, so the caller passes the
names it knows (client and trainer from the context) and they are replaced by role labels.
"""

from __future__ import annotations

import re

_PHONE = re.compile(r"(?<!\d)(?:\+?7|8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_IIN = re.compile(r"(?<!\d)\d{12}(?!\d)")


def mask(text: str, names: dict[str, str] | None = None) -> str:
    if not text:
        return text
    out = _EMAIL.sub("[email]", text)
    out = _PHONE.sub("[телефон]", out)
    out = _IIN.sub("[ИИН]", out)
    for name, label in (names or {}).items():
        if not name:
            continue
        for part in {name, *name.split()}:
            if len(part) >= 3:
                out = re.sub(rf"\b{re.escape(part)}\b", label, out)
    return out
