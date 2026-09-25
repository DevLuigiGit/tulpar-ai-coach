"""Normalize whitespace while preserving paragraph boundaries."""

import re

# Characters PDF/DOCX extractors leave behind that split or glue words for the embedder and the splitter:
# invisible ones are dropped, typographic ligatures are expanded, exotic spaces and breaks are unified.
_TRANSLATE = str.maketrans({
    **dict.fromkeys(map(ord, "­​‌‍⁠﻿"), None),
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "\f": "\n", "\v": "\n", " ": "\n", " ": "\n\n",
})
_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t   -   　]+")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").translate(_TRANSLATE)
    text = _CONTROL.sub("", text)
    text = _SPACES.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
