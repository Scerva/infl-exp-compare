"""Text normalization applied after scraping, before sentence splitting.

The scrapers in src/data/{fed_speeches,fomc}.py already strip HTML tags,
navigation chrome, footnote markers, and share/social boilerplate at
extraction time (see their docstrings), so this module's job is narrower:
normalize whitespace/unicode artifacts left over from HTML entity
decoding so downstream sentence splitting and NLP models see clean,
consistent text.
"""
from __future__ import annotations

import re
import unicodedata

# Characters that decode from HTML entities (&#160; etc.) or appear in Fed
# text as "smart" typography; normalize to plain ASCII-ish equivalents so
# regex-based sentence splitting and dictionary matching behave predictably.
_SPACE_CHARS = {
    " ": " ",  # non-breaking space
    " ": " ",  # thin space
    "​": "",  # zero-width space
}
_DASH_CHARS = {
    "–": "-",  # en dash
    "—": "--",  # em dash
}
_QUOTE_CHARS = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
}


def normalize_text(text: str) -> str:
    """Unicode-normalize, standardize whitespace/dashes/quotes, collapse
    repeated blank lines. Idempotent: safe to call more than once.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in {**_SPACE_CHARS, **_DASH_CHARS, **_QUOTE_CHARS}.items():
        text = text.replace(src, dst)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def remove_boilerplate_lines(text: str, min_len: int = 3) -> str:
    """Drop lines that are just whitespace/near-empty artifacts left after
    paragraph joins; does NOT attempt topic-based filtering (that is the
    relevance classifier's job, see src/text/relevance.py).
    """
    lines = [ln for ln in text.split("\n") if len(ln.strip()) >= min_len]
    return "\n".join(lines)
