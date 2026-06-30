from __future__ import annotations

import re
from typing import Optional, List


def normalize_whitespace(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def safe_lower(text: Optional[str]) -> str:
    return (text or "").lower()


def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out
