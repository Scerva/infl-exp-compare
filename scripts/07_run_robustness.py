#!/usr/bin/env python
"""Stage 7: robustness suite (spec section 23).

Produces TABLE 8 -- Robustness Results: for the top-ranked measures from
Table 7, refits the baseline specification across regime splits
(pre-2020 / pandemic / post-2022 tightening / excluding pandemic),
document-type splits (speech-only / FOMC-only), a change-vs-level
specification, and an alternative (Newey-West) inference method, and
reports whether the coefficient sign survives each variant.

Also updates Table 7 with the resulting robustness_pass_rate column.

Usage: python scripts/07_run_robustness.py [--top-n 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.registry import MACRO_CONTROL_COLUMNS, MEASURES_BY_ID
from src.models.robustness import run_robustness_checks, summarize_robustness_pass_rates
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"


def main(top_n: int) -> int:
    panel = pd.read_parquet(PROCESSED_DIR / "model_panel.parquet")

    importance_path = TABLES_DIR / "table7_importance_ranking.csv"
    if not importance_path.exists():
        logger.error("%s not found; run scripts/06_run_models.py first.", importance_path)
        return 1
    importance = pd.read_csv(importance_path)
    top_measure_ids = importance.head(top_n)["measure_id"].tolist()
    measures = [MEASURES_BY_ID[mid] for mid in top_measure_ids if mid in MEASURES_BY_ID]
    logger.info("Running robustness checks for top %d measures: %s", len(measures), [m.measure_id for m in measures])

    robustness_df = run_robustness_checks(panel, measures, control_cols=MACRO_CONTROL_COLUMNS)
    robustness_df.to_csv(TABLES_DIR / "table8_robustness.csv", index=False)
    logger.info("Table 8 (robustness): %d rows -> %s", len(robustness_df), TABLES_DIR / "table8_robustness.csv")

    pass_rates = summarize_robustness_pass_rates(robustness_df)
    logger.info("Robustness pass rates (share of variants matching baseline sign):\n%s", pass_rates.to_string())

    importance["robustness_pass_rate"] = importance["measure_id"].map(pass_rates)
    importance.to_csv(importance_path, index=False)
    logger.info("Updated %s with robustness_pass_rate.", importance_path)

    logger.info("Stage 7 completed successfully.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()
    sys.exit(main(args.top_n))
