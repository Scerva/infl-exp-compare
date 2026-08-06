#!/usr/bin/env python
"""Stage 3: run the monetary-policy relevance filter, Model A dictionary
stance scorer, and expectation-mention extractor over every downloaded
Fed communication (speeches, FOMC minutes, FOMC statements).

Produces:
  data/processed/documents_scored.parquet   -- one row per document, with
    the full field set spec section 1 requires (monetary_policy_relevance,
    hawk_score, dove_score, neutral_score, stance_score, stance_uncertainty,
    inflation_relevance, expectations_relevance, ...).
  data/processed/expectation_mentions.parquet -- one row per direct
    reference to a specific inflation-expectations measure, with local
    context and local stance/concern scores (spec section 6).

Usage: python scripts/03_build_stance_scores.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text.clean import normalize_text
from src.text.expectation_mentions import find_mentions_in_sentences, summarize_mentions
from src.text.relevance import DEFAULT_RELEVANCE_THRESHOLD, score_document
from src.text.sentence_split import split_sentences
from src.text.stance import score_document_stance
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DOCUMENT_SOURCES = ["fed_speeches", "fomc_minutes", "fomc_statements"]

DOCUMENT_COLUMNS = [
    "document_id", "document_type", "event_date", "publication_date", "meeting_date",
    "speaker", "speaker_role", "title", "venue", "source_url",
    "raw_text", "clean_text",
    "monetary_policy_relevance", "inflation_relevance", "expectations_relevance",
    "hawk_score", "dove_score", "neutral_score", "stance_score", "stance_uncertainty",
    "n_sentences", "n_relevant_sentences",
]


def _process_document(doc_id: str, doc_type: str, text: str) -> dict:
    normalized = normalize_text(text or "")
    sentences = split_sentences(normalized)

    doc_rel = score_document(sentences)
    doc_stance, _ = score_document_stance(sentences)

    n = max(doc_rel.n_sentences, 1)
    inflation_relevance = doc_rel.topic_sentence_counts.get("inflation", 0) / n
    expectations_relevance = doc_rel.topic_sentence_counts.get("inflation_expectations", 0) / n

    return {
        "monetary_policy_relevance": doc_rel.monetary_policy_relevance_score,
        "inflation_relevance": inflation_relevance,
        "expectations_relevance": expectations_relevance,
        "hawk_score": doc_stance.hawk_score,
        "dove_score": doc_stance.dove_score,
        "neutral_score": doc_stance.neutral_score,
        "stance_score": doc_stance.stance_mean_weighted,
        "stance_uncertainty": doc_stance.stance_uncertainty,
        "n_sentences": doc_rel.n_sentences,
        "n_relevant_sentences": doc_rel.n_relevant_sentences,
    }, sentences, doc_rel


def main() -> int:
    frames = []
    for name in DOCUMENT_SOURCES:
        path = PROCESSED_DIR / f"{name}.parquet"
        if not path.exists():
            logger.error("%s not found; run scripts/02_download_fed_text.py first.", path)
            return 1
        frames.append(pd.read_parquet(path))
    raw_docs = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("Loaded %d raw documents across %s", len(raw_docs), DOCUMENT_SOURCES)

    doc_rows = []
    mention_rows = []
    n_excluded = 0

    for _, doc in raw_docs.iterrows():
        scores, sentences, doc_rel = _process_document(doc["document_id"], doc["document_type"], doc.get("clean_text", ""))
        if scores["monetary_policy_relevance"] < DEFAULT_RELEVANCE_THRESHOLD:
            n_excluded += 1

        row = {c: doc.get(c) for c in ["document_id", "document_type", "event_date", "publication_date",
                                         "meeting_date", "speaker", "speaker_role", "title", "venue",
                                         "source_url", "raw_text", "clean_text"]}
        row.update(scores)
        doc_rows.append(row)

        mentions = find_mentions_in_sentences(sentences)
        for m in mentions:
            mention_rows.append(
                {
                    "document_id": doc["document_id"],
                    "date": doc["event_date"],
                    "speaker": doc.get("speaker"),
                    "document_type": doc["document_type"],
                    "sentence_index": m.sentence_index,
                    "measure_family": m.measure_family,
                    "specific_measure": m.specific_measure,
                    "sentence": m.sentence,
                    "context": m.context,
                    "local_stance_score": m.local_stance_score,
                    "concern_score": m.concern_score,
                }
            )

    documents_df = pd.DataFrame(doc_rows)
    for c in DOCUMENT_COLUMNS:
        if c not in documents_df.columns:
            documents_df[c] = pd.NA
    documents_df = documents_df[DOCUMENT_COLUMNS]

    mentions_df = pd.DataFrame(mention_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    documents_df.to_parquet(PROCESSED_DIR / "documents_scored.parquet", index=False)
    mentions_df.to_parquet(PROCESSED_DIR / "expectation_mentions.parquet", index=False)

    logger.info("Wrote documents_scored.parquet: %d rows", len(documents_df))
    logger.info(
        "  %d/%d documents (%.1f%%) below monetary-policy relevance threshold (%.2f) -- excluded from stance regressions downstream",
        n_excluded, len(documents_df), 100 * n_excluded / max(len(documents_df), 1), DEFAULT_RELEVANCE_THRESHOLD,
    )
    logger.info("Wrote expectation_mentions.parquet: %d rows across %d documents", len(mentions_df), mentions_df["document_id"].nunique() if len(mentions_df) else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
