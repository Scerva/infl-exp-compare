#!/usr/bin/env python
"""Stage 2: download Fed Board speeches and FOMC minutes/statements,
persist to data/processed/ as parquet.

Usage: python scripts/02_download_fed_text.py [--start-year 1996] [--end-year <current>] [--force-refresh]

Note: a full run (1996-present) makes on the order of 1,500-2,000 HTTP
requests (one per speech/minutes/statement page) and will take a while
even with caching on repeat runs. Use --start-year/--end-year to scope a
smaller run during development.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fed_speeches import LEGACY_MIN_YEAR, get_speeches
from src.data.fomc import HISTORICAL_MIN_YEAR, get_minutes, get_statements
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _write(df: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("wrote %s (%d rows) -> %s", name, len(df), out_path)


def main(start_year: int, end_year: int, force_refresh: bool = False) -> int:
    failures: list[str] = []

    try:
        speeches = get_speeches(max(start_year, LEGACY_MIN_YEAR), end_year, force_refresh=force_refresh)
        _write(speeches, "fed_speeches")
    except Exception:
        logger.exception("Fed speeches download failed")
        failures.append("FED_SPEECHES")

    try:
        minutes = get_minutes(max(start_year, HISTORICAL_MIN_YEAR), end_year, force_refresh=force_refresh)
        _write(minutes, "fomc_minutes")
    except Exception:
        logger.exception("FOMC minutes download failed")
        failures.append("FOMC_MINUTES")

    try:
        statements = get_statements(max(start_year, HISTORICAL_MIN_YEAR), end_year, force_refresh=force_refresh)
        _write(statements, "fomc_statements")
    except Exception:
        logger.exception("FOMC statements download failed")
        failures.append("FOMC_STATEMENTS")

    if failures:
        logger.error("Stage 2 completed with %d failed source(s): %s", len(failures), failures)
        return 1
    logger.info("Stage 2 completed successfully.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1996)
    parser.add_argument("--end-year", type=int, default=dt.date.today().year)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()
    sys.exit(main(args.start_year, args.end_year, force_refresh=args.force_refresh))
