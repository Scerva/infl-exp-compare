"""Lightweight, dependency-free sentence segmentation tuned for Fed prose.

Fed communications are dense with patterns that break naive
"split on period" tokenizers: decimal numbers ("2.5 percent"), title
abbreviations ("Mr. Powell", "Gov. Waller"), and month abbreviations in
dates. Rather than pull in a full NLP tokenizer (spaCy/NLTK) for this one
task, we use a small protect-then-split regex approach: known
abbreviation and decimal-number patterns are temporarily masked, the text
is split on sentence-ending punctuation followed by whitespace and a
capital letter/quote, then the mask is undone.

This is intentionally simple and auditable -- see tests/test_sentence_split.py
for the exact edge cases it is expected to handle correctly, and the
project's human-annotation validation step (spec section 7) is the
backstop for anything a regex tokenizer gets wrong.
"""
from __future__ import annotations

import re

ABBREVIATIONS = [
    "mr", "mrs", "ms", "dr", "gov", "sec", "st", "jr", "sr", "prof",
    "rep", "sen", "gen", "rev", "hon", "capt", "vs",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "u.s", "u.k", "e.g", "i.e", "etc", "inc", "corp", "co", "ph.d", "no", "pp", "fed", "approx",
]

_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
_ABBR_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in ABBREVIATIONS) + r")\.",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])["\')\]]?\s+(?=["\'(\[]?[A-Z0-9])')

_DECIMAL_PLACEHOLDER = " DEC "
_ABBR_PLACEHOLDER = " ABBR "


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences. Returns [] for empty input."""
    if not text or not text.strip():
        return []

    protected = _DECIMAL_RE.sub(_DECIMAL_PLACEHOLDER, text)
    protected = _ABBR_RE.sub(lambda m: m.group(1) + _ABBR_PLACEHOLDER, protected)

    raw_sentences = _SENTENCE_BOUNDARY_RE.split(protected)

    sentences = []
    for s in raw_sentences:
        s = s.replace(_DECIMAL_PLACEHOLDER, ".").replace(_ABBR_PLACEHOLDER, ".")
        s = s.strip()
        if s:
            sentences.append(s)
    return sentences


def split_paragraphs_to_sentences(paragraphs: list[str]) -> list[tuple[int, int, str]]:
    """Split a list of paragraphs into (paragraph_index, sentence_index_within_paragraph, sentence)."""
    out = []
    for p_idx, para in enumerate(paragraphs):
        for s_idx, sent in enumerate(split_sentences(para)):
            out.append((p_idx, s_idx, sent))
    return out
