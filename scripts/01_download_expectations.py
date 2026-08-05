#!/usr/bin/env python
"""Stage 1: download every inflation-expectations source plus the macro
control panel, validate each, and persist to data/processed/ as parquet.

Usage: python scripts/01_download_expectations.py [--force-refresh]

This script is intentionally sequential and verbose (one series at a
time, logged) rather than parallelized: several sources (Philly Fed,
NY Fed) share rate-limited hosts, and failures need to be attributable to
a specific series for the "never silently continue on a broken source"
requirement in the project spec (section 30).

The Atlanta Fed BIE module is explicitly OPTIONAL (spec section 3H): a
failure there is logged and skipped rather than aborting the whole run.
Every other source is required -- a failure aborts with a nonzero exit
code so broken pipelines are never silently reported as successful.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.atlanta_bie import get_atlanta_bie
from src.data.atsix import get_atsix, get_atsix_selected_horizons
from src.data.fred import get_fred_series
from src.data.livingston import get_livingston_series
from src.data.nyfed_sce import get_sce_inflation_expectations
from src.data.philly_spf import get_spf_series
from src.features.macro_controls import get_macro_panel
from src.utils.config import sources_by_provider
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _write(df: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("wrote %s (%d rows) -> %s", name, len(df), out_path)


def main(force_refresh: bool = False) -> int:
    failures: list[str] = []

    # --- FRED-sourced series (market-based + Cleveland model + Michigan) ---
    fred_sources = sources_by_provider("FRED")
    fred_frames = []
    for s in fred_sources:
        try:
            fred_frames.append(get_fred_series(s.series_id, start_date="2000-01-01", force_refresh=force_refresh))
        except Exception:
            logger.exception("FRED series %s failed", s.series_id)
            failures.append(f"FRED:{s.series_id}")
    if fred_frames:
        _write(pd.concat(fred_frames, ignore_index=True), "fred_series_long")

    # --- NY Fed SCE ---
    try:
        _write(get_sce_inflation_expectations(force_refresh=force_refresh), "nyfed_sce")
    except Exception:
        logger.exception("NY Fed SCE failed")
        failures.append("NYFED_SCE")

    # --- Philly Fed SPF ---
    spf_frames = []
    for var in ["CPI10", "CPI5YR", "PCE10", "PCE5YR", "CPI1YR", "PCE1YR"]:
        try:
            spf_frames.append(get_spf_series(var, "median", force_refresh=force_refresh))
        except Exception:
            logger.exception("SPF series %s failed", var)
            failures.append(f"SPF:{var}")
    if spf_frames:
        _write(pd.concat(spf_frames, ignore_index=True), "philly_spf")

    # --- Livingston ---
    try:
        _write(get_livingston_series(force_refresh=force_refresh), "livingston")
    except Exception:
        logger.exception("Livingston failed")
        failures.append("LIVINGSTON")

    # --- ATSIX ---
    try:
        _write(get_atsix(force_refresh=force_refresh), "atsix_full_curve")
        _write(get_atsix_selected_horizons(force_refresh=force_refresh), "atsix_selected_horizons")
    except Exception:
        logger.exception("ATSIX failed")
        failures.append("ATSIX")

    # --- Atlanta Fed BIE (OPTIONAL) ---
    try:
        _write(get_atlanta_bie(force_refresh=force_refresh), "atlanta_bie")
    except Exception:
        logger.warning("Atlanta Fed BIE failed (OPTIONAL source, continuing): %s", sys.exc_info()[1])

    # --- Macro control panel ---
    try:
        _write(get_macro_panel(start_date="2000-01-01", force_refresh=force_refresh), "macro_controls")
    except Exception:
        logger.exception("Macro control panel failed")
        failures.append("MACRO_CONTROLS")

    if failures:
        logger.error("Stage 1 completed with %d failed REQUIRED source(s): %s", len(failures), failures)
        return 1
    logger.info("Stage 1 completed successfully: all required sources downloaded and validated.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-refresh", action="store_true", help="Bypass local cache and re-download everything.")
    args = parser.parse_args()
    sys.exit(main(force_refresh=args.force_refresh))
