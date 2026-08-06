#!/usr/bin/env python
"""Run Model B (zero-shot NLI) and Model C (FinBERT) over the human-
annotation sample and compare against Model A (dictionary), which is
already scored for every sentence in that sample.

This does NOT run the transformer models over the full corpus -- see
src/text/stance_nli.py's module docstring for why (CPU-only, ~1.6GB
model, full corpus would take hours). Running over the same 400-sentence
stratified sample that will eventually get human labels is what actually
matters for the spec's model-selection requirement: once
data/annotation/stance_annotation_sample.csv has human_label filled in,
src.text.annotation.score_annotations can be run against each model's
predictions to decide which one is primary.

Usage: python scripts/run_nli_benchmark.py
Writes: data/processed/stance_model_comparison.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text.stance_finbert import score_sentences_finbert
from src.text.stance_nli import score_sentences_nli
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_PATH = PROJECT_ROOT / "data" / "annotation" / "stance_annotation_sample.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "stance_model_comparison.parquet"


def main() -> int:
    if not ANNOTATION_PATH.exists():
        logger.error("%s not found; run scripts/build_annotation_sample.py first.", ANNOTATION_PATH)
        return 1
    sample = pd.read_csv(ANNOTATION_PATH)
    sentences = sample["sentence"].tolist()
    logger.info("Scoring %d sentences with Model B (zero-shot NLI)...", len(sentences))
    nli_results = score_sentences_nli(sentences)

    logger.info("Scoring %d sentences with Model C (FinBERT)...", len(sentences))
    finbert_results = score_sentences_finbert(sentences)

    out = sample.copy()
    out["model_a_stance"] = sample["model_stance_score"]  # already computed at sample-build time
    out["model_b_p_hawkish"] = [r["p_hawkish"] for r in nli_results]
    out["model_b_p_dovish"] = [r["p_dovish"] for r in nli_results]
    out["model_b_stance"] = [r["stance"] for r in nli_results]
    out["model_c_p_positive"] = [r["p_positive"] for r in finbert_results]
    out["model_c_p_negative"] = [r["p_negative"] for r in finbert_results]
    out["model_c_finbert_tone"] = [r["finbert_tone"] for r in finbert_results]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %s (%d rows)", OUT_PATH, len(out))

    corr_ab = out["model_a_stance"].corr(out["model_b_stance"])
    corr_ac = out["model_a_stance"].corr(out["model_c_finbert_tone"])
    corr_bc = out["model_b_stance"].corr(out["model_c_finbert_tone"])
    logger.info("Pairwise correlations (dictionary A vs NLI B vs FinBERT C, over annotation sample):")
    logger.info("  corr(A, B) = %.3f", corr_ab)
    logger.info("  corr(A, C) = %.3f  (expected weak: C is financial tone, not stance)", corr_ac)
    logger.info("  corr(B, C) = %.3f  (expected weak, same reason)", corr_bc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
