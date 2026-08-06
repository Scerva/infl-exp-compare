"""Monetary-policy relevance classification (spec section 5).

This stage runs BEFORE hawkish/dovish stance scoring. Many Fed Board
speeches concern bank supervision, payments, community development,
consumer protection, or technology topics that have no bearing on
monetary-policy stance; scoring those documents for hawkishness would be
meaningless (or worse, silently noisy). We therefore classify every
sentence for relevance to the nine topic areas the spec calls out, and
aggregate to a document-level ``monetary_policy_relevance_score`` used to
exclude/down-weight off-topic documents downstream.

Design choice: a transparent, auditable keyword/phrase dictionary rather
than a black-box classifier for this stage. Unlike stance (hawkish vs.
dovish), topical relevance is a much easier/more literal classification
problem -- "does this sentence mention inflation, rates, employment,
etc." is well captured by phrase matching, and keeping this stage
dictionary-based means the more expensive stance models (Model B/C) only
ever have to run on the subset of sentences that already passed a cheap,
inspectable filter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Each topic maps to a list of regex patterns (case-insensitive, word-
# boundary-aware). Multi-word phrases are listed first for readability but
# matching order does not affect the result (a sentence's topic set is a
# union over all patterns).
TOPIC_PATTERNS: dict[str, list[str]] = {
    "inflation": [
        r"\binflation\b", r"\bdisinflation\b", r"\bdeflation\b", r"\bprice stability\b",
        r"\bcore (?:cpi|pce)\b", r"\bconsumer price\w*\b", r"\bpce price\w*\b",
        r"\bprices? (?:rose|rise|rising|increased?|declined?|fell|falling)\b",
    ],
    "inflation_expectations": [
        r"inflation expectations?", r"expected inflation", r"long-run inflation",
        r"inflation compensation", r"break-?even inflation", r"anchored (?:inflation )?expectations?",
        r"unanchor\w*", r"expectations? (?:are|remain|stay(?:ed|s)?) anchored",
    ],
    "labor_market": [
        r"labor market", r"unemployment rate", r"\bpayrolls?\b", r"job (?:gains?|growth|losses|creation)",
        r"labor force participation", r"wage growth", r"average hourly earnings",
        r"labor demand", r"labor supply", r"hiring",
    ],
    "growth_activity": [
        r"economic (?:activity|growth|outlook|conditions)", r"\bgdp\b", r"real output",
        r"business (?:investment|fixed investment)", r"consumer spending", r"household spending",
        r"\brecession\b", r"economic expansion", r"industrial production",
    ],
    "monetary_policy": [
        r"monetary policy", r"the committee", r"\bfomc\b", r"federal open market committee",
        r"policy stance", r"accommodative policy", r"restrictive (?:policy|stance)",
        r"policy normalization", r"stance of (?:monetary )?policy",
    ],
    "interest_rates": [
        r"federal funds rate", r"interest rates?", r"target range", r"rate (?:hike|cut|increase|decrease|pause)",
        r"basis points?", r"policy rate", r"benchmark rate",
    ],
    "financial_conditions": [
        r"financial conditions", r"credit conditions", r"financial markets?", r"financial stability",
        r"asset prices", r"equity (?:markets?|prices)", r"treasury yields?", r"borrowing costs",
    ],
    "balance_sheet": [
        r"balance sheet", r"quantitative easing", r"\bqe\b", r"\bqt\b", r"asset purchases?",
        r"securities holdings", r"reinvestment", r"balance sheet (?:runoff|reduction|normalization)",
    ],
    "dual_mandate": [
        r"dual mandate", r"maximum employment", r"employment and price stability",
        r"statutory mandate", r"longer-run goals",
    ],
}

_COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {
    topic: [re.compile(p, re.IGNORECASE) for p in patterns] for topic, patterns in TOPIC_PATTERNS.items()
}

DEFAULT_RELEVANCE_THRESHOLD = 0.10


@dataclass
class SentenceRelevance:
    sentence: str
    topics: list[str] = field(default_factory=list)

    @property
    def is_relevant(self) -> bool:
        return len(self.topics) > 0


def score_sentence(sentence: str) -> SentenceRelevance:
    topics = [topic for topic, patterns in _COMPILED_PATTERNS.items() if any(p.search(sentence) for p in patterns)]
    return SentenceRelevance(sentence=sentence, topics=topics)


@dataclass
class DocumentRelevance:
    n_sentences: int
    n_relevant_sentences: int
    monetary_policy_relevance_score: float
    topic_sentence_counts: dict[str, int]
    relevant_sentence_indices: list[int]

    @property
    def is_relevant(self) -> bool:
        return self.monetary_policy_relevance_score >= DEFAULT_RELEVANCE_THRESHOLD


def score_document(sentences: list[str]) -> DocumentRelevance:
    """Aggregate sentence-level relevance to a document score.

    ``monetary_policy_relevance_score`` is the share of sentences that hit
    at least one topic pattern -- simple and auditable by design (spec
    section 5: "Create a monetary_policy_relevance_score from 0 to 1").
    """
    if not sentences:
        return DocumentRelevance(0, 0, 0.0, {t: 0 for t in TOPIC_PATTERNS}, [])

    topic_counts = {t: 0 for t in TOPIC_PATTERNS}
    relevant_idx = []
    for i, s in enumerate(sentences):
        sr = score_sentence(s)
        if sr.is_relevant:
            relevant_idx.append(i)
            for t in sr.topics:
                topic_counts[t] += 1

    n = len(sentences)
    n_rel = len(relevant_idx)
    return DocumentRelevance(
        n_sentences=n,
        n_relevant_sentences=n_rel,
        monetary_policy_relevance_score=n_rel / n,
        topic_sentence_counts=topic_counts,
        relevant_sentence_indices=relevant_idx,
    )
