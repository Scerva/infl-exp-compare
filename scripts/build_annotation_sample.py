#!/usr/bin/env python
"""Build the stratified human-annotation sample (spec section 7).

Writes data/annotation/stance_annotation_sample.csv with human_label and
human_confidence left blank for manual annotation. Run this after
scripts/03_build_stance_scores.py.

Usage: python scripts/build_annotation_sample.py [--n-target 400]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text.annotation import build_annotation_sample
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotation"


def main(n_target: int, seed: int) -> int:
    path = PROCESSED_DIR / "documents_scored.parquet"
    if not path.exists():
        logger.error("%s not found; run scripts/03_build_stance_scores.py first.", path)
        return 1
    documents_df = pd.read_parquet(path)

    sample = build_annotation_sample(documents_df, n_target=n_target, seed=seed)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANNOTATION_DIR / "stance_annotation_sample.csv"
    sample.to_csv(out_path, index=False)

    logger.info("Wrote %d sentences to %s", len(sample), out_path)
    logger.info("Stratum coverage:\n%s", sample.groupby(["decade", "document_type", "model_predicted_label"]).size())
    logger.info("Speaker coverage:\n%s", sample["speaker"].value_counts().head(15))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-target", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sys.exit(main(args.n_target, args.seed))
