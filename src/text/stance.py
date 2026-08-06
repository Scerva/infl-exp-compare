"""Monetary-policy stance scoring: hawkish vs. dovish, NOT positive/negative
sentiment (spec section 7).

This module implements MODEL A, the transparent dictionary baseline. It
is deliberately the first and simplest of the three stance approaches the
project spec calls for (dictionary / zero-shot NLI / FinBERT robustness):
every scoring decision here is a literal, inspectable phrase match plus a
negation check, so it can serve as an audit baseline against which the
heavier transformer models (src/models or a future src/text/stance_nli.py)
are validated -- the spec explicitly says the primary method should be
chosen on validation performance, not convenience, and you cannot judge
"is the fancier model actually better" without a legible baseline.

Pipeline (spec section 7 steps 1-6):
  1. sentence segmentation      -> src.text.sentence_split
  2. monetary-policy relevance  -> src.text.relevance (extended here with
                                    a continuous per-sentence relevance
                                    weight, not just the boolean flag)
  3. hawk/dove phrase matching, with a negation window so "not
     appropriate to raise rates" does not count as hawkish
  4. stance_sentence = (hawk_hits - dove_hits) / (hawk_hits + dove_hits),
     in [-1, 1], 0 for sentences with no phrase hits
  5. sentence weight = relevance_score * forward_looking_factor *
     explicit_policy_language_factor
  6. document stance = weighted mean/median/dispersion/proportions across
     sentences, matching every statistic spec section 7 asks for
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.text.relevance import TOPIC_PATTERNS, score_sentence

# --- Phrase dictionaries -----------------------------------------------
# Multi-word phrases and regex fragments, not single loaded words like
# "inflation" or "rates" (spec section 7: "Do not simply count individual
# words"). Each entry is matched case-insensitively with word boundaries
# implied by \b where relevant.

HAWKISH_PHRASES = [
    r"persistent(?:ly)? (?:high )?inflation",
    r"upside (?:inflation )?risks?(?: to inflation)?",
    r"unacceptably high",
    r"restrictive (?:policy|stance)",
    r"(?:additional|further) (?:policy )?tightening",
    r"higher for longer",
    r"not (?:yet )?appropriate to (?:cut|ease|lower)",
    r"insufficient progress",
    r"unanchor\w* expectations?",
    r"further (?:policy )?firming",
    r"(?:raise|increase) the target range",
    r"additional rate increases?",
    r"keep policy restrictive",
    r"elevated inflation",
    r"inflation (?:remains?|is) (?:too )?high",
    r"remain vigilant",
    r"resolute(?:ly)?",
    r"firmly committed to (?:price stability|returning inflation)",
    r"more work to do",
    r"not there yet",
    r"further policy firming",
    r"willing(?:ness)? to tighten",
    r"reluctan\w* to ease",
    r"risk of de-?anchoring",
    r"too high for too long",
]

DOVISH_PHRASES = [
    r"downside (?:employment|growth|economic) risks?",
    r"policy (?:is |remains? )?too restrictive",
    r"appropriate to reduce (?:restraint|restriction)",
    r"disinflation\w*",
    r"confidence (?:that )?inflation (?:is|will) return\w* to target",
    r"policy normalization",
    r"rate cuts?",
    r"eas(?:e|ing) (?:of )?policy",
    r"weaker labor demand",
    r"(?:lower|decrease) the target range",
    r"cut(?:ting)? (?:the )?(?:target range|rates)",
    r"less restrictive",
    r"loosen\w* (?:policy|stance)",
    r"support(?:ing)? (?:the )?labor market",
    r"downside risks? to (?:employment|growth|the economy|the labor market)",
    r"softening labor market",
    r"reduce (?:the degree of )?restriction",
    r"appropriate to (?:begin )?(?:easing|reducing)",
    r"progress (?:on|toward) (?:disinflation|(?:our|the) inflation objective)",
    r"inflation (?:is|has) (?:coming|come) down",
    r"confidence (?:has )?increased",
]

# Negation cues checked in a short window immediately before a matched
# phrase; a negated hit is DISCOUNTED (dropped), not flipped to the
# opposite polarity -- flipping is a stronger, less defensible claim than
# simply not counting an explicitly negated statement.
NEGATION_CUES = [
    "not", "no longer", "isn't", "aren't", "wasn't", "weren't", "won't",
    "wouldn't", "unlikely", "without", "never", "doesn't", "didn't", "don't",
    "far from", "nowhere near",
]
_NEGATION_WINDOW_CHARS = 40

_HAWK_RE = [re.compile(p, re.IGNORECASE) for p in HAWKISH_PHRASES]
_DOVE_RE = [re.compile(p, re.IGNORECASE) for p in DOVISH_PHRASES]

_FORWARD_LOOKING_RE = re.compile(
    r"\b(?:expect|anticipat\w+|likely to|going forward|over the (?:medium|coming) term|"
    r"in (?:the )?coming months|will (?:likely|continue|remain)|is projected to|"
    r"we (?:will|would)|the committee will)\b",
    re.IGNORECASE,
)

N_TOPICS = len(TOPIC_PATTERNS)


def _is_negated(sentence: str, match_start: int) -> bool:
    window_start = max(0, match_start - _NEGATION_WINDOW_CHARS)
    window = sentence[window_start:match_start].lower()
    return any(re.search(rf"\b{re.escape(cue)}\b", window) for cue in NEGATION_CUES)


def _relevance_score(sentence: str) -> float:
    sr = score_sentence(sentence)
    return min(1.0, 0.34 * len(sr.topics))


@dataclass
class SentenceStance:
    sentence: str
    hawk_hits: int
    dove_hits: int
    stance: float  # (hawk_hits - dove_hits) / (hawk_hits + dove_hits), 0 if no hits
    relevance_score: float
    is_forward_looking: bool
    weight: float
    matched_hawk_phrases: list[str] = field(default_factory=list)
    matched_dove_phrases: list[str] = field(default_factory=list)


def score_sentence_stance(sentence: str) -> SentenceStance:
    hawk_hits, dove_hits = 0, 0
    hawk_matched, dove_matched = [], []

    for pat in _HAWK_RE:
        for m in pat.finditer(sentence):
            if not _is_negated(sentence, m.start()):
                hawk_hits += 1
                hawk_matched.append(m.group(0))
    for pat in _DOVE_RE:
        for m in pat.finditer(sentence):
            if not _is_negated(sentence, m.start()):
                dove_hits += 1
                dove_matched.append(m.group(0))

    total = hawk_hits + dove_hits
    stance = (hawk_hits - dove_hits) / total if total > 0 else 0.0

    relevance = _relevance_score(sentence)
    is_fwd = bool(_FORWARD_LOOKING_RE.search(sentence))
    forward_factor = 1.2 if is_fwd else 1.0
    policy_language_factor = 1.0 + 0.1 * min(total, 5)
    weight = relevance * forward_factor * policy_language_factor

    return SentenceStance(
        sentence=sentence,
        hawk_hits=hawk_hits,
        dove_hits=dove_hits,
        stance=stance,
        relevance_score=relevance,
        is_forward_looking=is_fwd,
        weight=weight,
        matched_hawk_phrases=hawk_matched,
        matched_dove_phrases=dove_matched,
    )


@dataclass
class DocumentStance:
    n_sentences: int
    n_relevant_sentences: int
    stance_mean_weighted: float
    stance_median: float
    stance_top_decile_hawkish: float
    stance_top_decile_dovish: float
    proportion_hawkish: float
    proportion_dovish: float
    stance_dispersion: float
    hawk_score: float
    dove_score: float
    neutral_score: float
    stance_uncertainty: float


def score_document_stance(sentences: list[str], relevance_threshold: float = 0.0) -> tuple[DocumentStance, list[SentenceStance]]:
    """Aggregate sentence-level stance into document-level statistics.

    Only sentences with relevance_score > relevance_threshold contribute
    to the weighted document stance (spec: score hawkish/dovish
    conditional on monetary-policy relevance). ``relevance_threshold=0.0``
    means "any sentence hitting at least one topic," matching
    score_sentence's continuous relevance measure.
    """
    sentence_stances = [score_sentence_stance(s) for s in sentences]
    relevant = [ss for ss in sentence_stances if ss.relevance_score > relevance_threshold]

    n = len(sentences)
    n_rel = len(relevant)
    if n_rel == 0:
        empty = DocumentStance(
            n_sentences=n, n_relevant_sentences=0,
            stance_mean_weighted=0.0, stance_median=0.0,
            stance_top_decile_hawkish=0.0, stance_top_decile_dovish=0.0,
            proportion_hawkish=0.0, proportion_dovish=0.0,
            stance_dispersion=0.0, hawk_score=0.0, dove_score=0.0,
            neutral_score=1.0, stance_uncertainty=0.0,
        )
        return empty, sentence_stances

    stances = np.array([ss.stance for ss in relevant])
    weights = np.array([max(ss.weight, 1e-6) for ss in relevant])

    weighted_mean = float(np.average(stances, weights=weights))
    median = float(np.median(stances))
    dispersion = float(np.std(stances))

    sorted_stances = np.sort(stances)
    decile_n = max(1, int(np.ceil(0.1 * len(sorted_stances))))
    top_decile_hawkish = float(np.mean(sorted_stances[-decile_n:]))
    top_decile_dovish = float(np.mean(sorted_stances[:decile_n]))

    prop_hawkish = float(np.mean(stances > 0))
    prop_dovish = float(np.mean(stances < 0))
    prop_neutral = float(np.mean(stances == 0))

    hawk_score = float(np.average(np.clip(stances, 0, None), weights=weights))
    dove_score = float(np.average(np.clip(-stances, 0, None), weights=weights))

    doc_stance = DocumentStance(
        n_sentences=n,
        n_relevant_sentences=n_rel,
        stance_mean_weighted=weighted_mean,
        stance_median=median,
        stance_top_decile_hawkish=top_decile_hawkish,
        stance_top_decile_dovish=top_decile_dovish,
        proportion_hawkish=prop_hawkish,
        proportion_dovish=prop_dovish,
        stance_dispersion=dispersion,
        hawk_score=hawk_score,
        dove_score=dove_score,
        neutral_score=prop_neutral,
        stance_uncertainty=dispersion,
    )
    return doc_stance, sentence_stances
