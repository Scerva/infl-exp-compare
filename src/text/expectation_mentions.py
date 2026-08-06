"""Direct-mention extraction for inflation-expectations measures (spec
section 6).

This is a separate empirical channel from the document-level
regressions: rather than asking "does the Fed's overall tone correlate
with series X," this asks "which specific measures do policymakers
actually name out loud, how often, and what's the local tone around that
specific mention." A document can be broadly hawkish while a single
sentence naming the Michigan survey is neutral or even dovish -- that
distinction is only visible at the mention level.

For every sentence that names a specific expectations measure, we keep a
+/-2-sentence context window and score that window with the Model A
dictionary stance function (src.text.stance) to get a ``local_stance_score``,
plus a small separate "concern" lexicon (independent of hawk/dove
direction -- a sentence can express concern about expectations
de-anchoring in either a hawkish "we must act" framing or a more neutral
"we are watching this closely" framing) to get a ``concern_score``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.text.stance import score_document_stance

# measure_family follows the same vocabulary as config/sources.yaml
# source_family values (household, professional, market, business, model)
# plus "survey_generic"/"horizon_generic" for mentions that don't name a
# specific source.
@dataclass(frozen=True)
class MeasureDef:
    specific_measure: str
    measure_family: str
    patterns: tuple[str, ...]


MEASURE_DEFS: list[MeasureDef] = [
    MeasureDef(
        "Michigan Survey", "household",
        (r"university of michigan", r"michigan survey", r"surveys? of consumers\b",
         r"\bmichigan\b(?:['’]s)? (?:survey|expectations|sentiment|index)"),
    ),
    MeasureDef(
        "NY Fed SCE", "household",
        (r"new york fed(?:eral reserve)?(?:'s)? survey", r"survey of consumer expectations", r"\bsce\b"),
    ),
    MeasureDef(
        "SPF", "professional",
        (r"survey of professional forecasters", r"\bspf\b"),
    ),
    MeasureDef(
        "Livingston Survey", "professional",
        (r"livingston survey", r"\blivingston\b"),
    ),
    MeasureDef(
        "Philadelphia Fed (general)", "professional",
        (r"philadelphia fed",),
    ),
    MeasureDef(
        "TIPS / Breakevens", "market",
        (r"\btips\b", r"treasury inflation-protected securities", r"breakeven\w*",
         r"inflation compensation", r"market-based (?:measures? of )?inflation expectations?"),
    ),
    MeasureDef(
        "5-Year, 5-Year Forward", "market",
        (r"5-year,?\s*5-year forward", r"five-year,?\s*five-year forward", r"\b5y5y\b"),
    ),
    MeasureDef(
        "Survey-Based (generic)", "survey_generic",
        (r"survey-based (?:measures? of )?(?:inflation )?expectations?",),
    ),
    MeasureDef(
        "Professional Forecasters (generic)", "professional",
        (r"professional forecasters",),
    ),
    MeasureDef(
        "Household/Consumer (generic)", "household",
        (r"household (?:inflation )?expectations?", r"consumer (?:inflation )?expectations?"),
    ),
    MeasureDef(
        "Business (generic)", "business",
        (r"business (?:inflation )?expectations?", r"firms['’]?\s*(?:inflation )?expectations"),
    ),
    MeasureDef(
        "Longer-Term (generic horizon)", "horizon_generic",
        (r"longer-term (?:inflation )?expectations?", r"long-run (?:inflation )?expectations?"),
    ),
]

_COMPILED: list[tuple[MeasureDef, list[re.Pattern]]] = [
    (mdef, [re.compile(p, re.IGNORECASE) for p in mdef.patterns]) for mdef in MEASURE_DEFS
]

CONCERN_LEXICON = [
    r"concern\w*", r"worr\w*", r"\brisk\w*", r"elevated", r"persistent\w*", r"stubborn\w*",
    r"sticky", r"uncomfortably", r"too high", r"above target", r"de-?anchor\w*",
    r"watching closely", r"vigilant", r"uncertain\w*",
]
_CONCERN_RE = [re.compile(p, re.IGNORECASE) for p in CONCERN_LEXICON]


def _concern_score(text: str) -> float:
    n_words = max(1, len(text.split()))
    n_hits = sum(len(p.findall(text)) for p in _CONCERN_RE)
    return min(1.0, n_hits / (n_words / 20))  # ~1 hit per 20 words saturates at 1.0


@dataclass
class Mention:
    sentence_index: int
    measure_family: str
    specific_measure: str
    sentence: str
    context: str
    local_stance_score: float = 0.0
    concern_score: float = 0.0


def find_mentions_in_sentences(sentences: list[str], context_window: int = 2) -> list[Mention]:
    """Locate every sentence naming a specific expectations measure, with
    a +/-``context_window`` sentence context and (stance, concern) scores
    computed over that context.
    """
    mentions: list[Mention] = []
    for i, sent in enumerate(sentences):
        for mdef, compiled in _COMPILED:
            if any(p.search(sent) for p in compiled):
                start = max(0, i - context_window)
                end = min(len(sentences), i + context_window + 1)
                context = " ".join(sentences[start:end])
                context_stance, _ = score_document_stance([context])
                mentions.append(
                    Mention(
                        sentence_index=i,
                        measure_family=mdef.measure_family,
                        specific_measure=mdef.specific_measure,
                        sentence=sent,
                        context=context,
                        local_stance_score=context_stance.stance_mean_weighted,
                        concern_score=_concern_score(context),
                    )
                )
    return mentions


@dataclass
class DocumentMentionSummary:
    n_mentions: int
    n_relevant_sentences: int
    mention_share_of_relevant: float
    mentions_by_family: dict[str, int] = field(default_factory=dict)
    mentions_by_measure: dict[str, int] = field(default_factory=dict)


def summarize_mentions(mentions: list[Mention], n_relevant_sentences: int) -> DocumentMentionSummary:
    by_family: dict[str, int] = {}
    by_measure: dict[str, int] = {}
    for m in mentions:
        by_family[m.measure_family] = by_family.get(m.measure_family, 0) + 1
        by_measure[m.specific_measure] = by_measure.get(m.specific_measure, 0) + 1
    share = len(mentions) / n_relevant_sentences if n_relevant_sentences > 0 else 0.0
    return DocumentMentionSummary(
        n_mentions=len(mentions),
        n_relevant_sentences=n_relevant_sentences,
        mention_share_of_relevant=share,
        mentions_by_family=by_family,
        mentions_by_measure=by_measure,
    )
