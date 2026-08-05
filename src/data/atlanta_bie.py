"""Atlanta Fed Business Inflation Expectations (BIE) downloader -- OPTIONAL.

Firms' expected year-ahead unit-cost growth (spec section 3H). This is the
one series in the catalog explicitly marked optional: the Atlanta Fed's
de-identified microdata requires a separate researcher request, out of
scope here, but the aggregate monthly headline series IS published as a
plain downloadable workbook, which this module retrieves.

Because this is optional, callers (e.g. scripts/01_download_expectations.py)
should catch failures from this module and continue the pipeline rather
than treating them as fatal -- this module itself does not silently
degrade or fabricate data; it raises loudly like every other downloader if
something is actually wrong.

COLUMN DISCOVERY: the workbook's "BIE Survey results" sheet has a two-row
merged-cell header (question text on row 1, sub-statistic labels -- e.g.
"Mean", "Median", "Diffusion Index" -- on row 2). We locate the "Question
4" (year-ahead unit-cost expectations) block by text search and then find
its "Mean" sub-column within that block, rather than hard-coding a column
index that would silently break if a question were inserted/removed.

RELEASE TIMING: the BIE page states its own publication rule explicitly:
survey collection runs Monday-Friday "typically at mid-month," and
"Survey release dates are scheduled for the following Wednesday," with
stated exceptions around FOMC blackout periods. The workbook's date column
is the Friday the survey period closed; we therefore compute
release_date = the next Wednesday after that date. This was verified
against the live page's "(July data) Updated: July 22, 2026" note, which
matches a 2026-07-17 (Friday) -> 2026-07-22 (Wednesday) pair present in
the same download. FOMC-blackout exceptions are not individually modeled,
so every row is flagged release_date_estimated=True; a verified-date
override file works the same way as for NY Fed SCE (see
config/atlanta_bie_release_dates_verified.csv).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.cache import FetchError, fetch
from src.utils.dates import to_ts
from src.utils.logging import get_logger
from src.utils.xlsx_repair import repair_xlsx_in_place

logger = get_logger(__name__)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

BIE_PAGE = "https://www.atlantafed.org/research/inflationproject/bie"
_XLSX_HREF_RE = re.compile(r'href="([^"]*/bie/data/bie\.xlsx[^"]*)"', re.IGNORECASE)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERIFIED_RELEASE_DATES_PATH = PROJECT_ROOT / "config" / "atlanta_bie_release_dates_verified.csv"

RESULTS_SHEET = "BIE Survey results"
TARGET_QUESTION_MARKER = "Question 4"
TARGET_STAT_LABEL = "mean"


def discover_workbook_url(force_refresh: bool = False) -> str:
    resp = fetch(
        BIE_PAGE,
        cache_key="bie_page",
        subdir="atlanta_bie",
        headers=BROWSER_HEADERS,
        ext="html",
        force_refresh=force_refresh,
        max_age_seconds=7 * 24 * 3600,
    )
    m = _XLSX_HREF_RE.search(resp.text())
    if not m:
        raise FetchError(f"Could not find bie.xlsx link on {BIE_PAGE}.")
    href = m.group(1).replace("&amp;", "&")
    return href if href.startswith("http") else "https://www.atlantafed.org" + href


def _find_year_ahead_mean_column(raw: pd.DataFrame) -> int:
    """Locate the 'Mean' sub-column within the Question-4 (year-ahead unit
    cost expectations) header block, by text search rather than a fixed
    index.
    """
    question_row = raw.iloc[1]
    stat_row = raw.iloc[2]

    q4_cols = [j for j, v in enumerate(question_row) if isinstance(v, str) and TARGET_QUESTION_MARKER in v]
    if not q4_cols:
        raise FetchError(f"Could not locate a {TARGET_QUESTION_MARKER!r} header in {RESULTS_SHEET} row 1.")
    start_col = q4_cols[0]

    later_headers = [
        j for j, v in enumerate(question_row) if j > start_col and isinstance(v, str) and v.strip()
    ]
    end_col = min(later_headers) if later_headers else len(question_row)

    for j in range(start_col, end_col):
        cell = stat_row[j]
        if isinstance(cell, str) and cell.strip().lower() == TARGET_STAT_LABEL:
            return j
    raise FetchError(
        f"Could not find a {TARGET_STAT_LABEL!r} sub-column within the {TARGET_QUESTION_MARKER!r} "
        f"block (columns {start_col}:{end_col}) of {RESULTS_SHEET}."
    )


def _next_wednesday_after(d: pd.Timestamp) -> pd.Timestamp:
    days_ahead = (2 - d.weekday()) % 7  # Wednesday == 2
    if days_ahead == 0:
        days_ahead = 7
    return d + pd.Timedelta(days=days_ahead)


def _load_verified_overrides() -> Optional[pd.DataFrame]:
    if not VERIFIED_RELEASE_DATES_PATH.exists():
        return None
    df = pd.read_csv(VERIFIED_RELEASE_DATES_PATH, parse_dates=["survey_period_end", "release_date"])
    return df if not df.empty else None


def get_atlanta_bie(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch the Atlanta Fed BIE year-ahead unit-cost mean expectation.

    Returns columns: series_id, survey_period_end, release_date,
    release_date_estimated, value (as a decimal fraction, e.g. 0.023 = 2.3%).
    """
    url = discover_workbook_url(force_refresh=force_refresh)
    resp = fetch(
        url,
        cache_key="bie",
        subdir="atlanta_bie",
        headers=BROWSER_HEADERS,
        ext="xlsx",
        force_refresh=force_refresh,
        max_age_seconds=7 * 24 * 3600,
    )
    repair_xlsx_in_place(resp.raw_path)
    xl = pd.ExcelFile(resp.raw_path)
    if RESULTS_SHEET not in xl.sheet_names:
        raise FetchError(f"Expected sheet {RESULTS_SHEET!r} in {url}, got {xl.sheet_names}")
    raw = xl.parse(RESULTS_SHEET, header=None)

    mean_col = _find_year_ahead_mean_column(raw)
    data = raw.iloc[3:][[0, mean_col]].copy()
    data.columns = ["survey_period_end", "value"]
    data["survey_period_end"] = pd.to_datetime(data["survey_period_end"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["survey_period_end", "value"]).reset_index(drop=True)
    if data.empty:
        raise FetchError(f"No usable observations parsed from {url} (column {mean_col}).")

    overrides = _load_verified_overrides()
    release_dates, is_estimated = [], []
    for d in data["survey_period_end"]:
        if overrides is not None:
            match = overrides.loc[overrides["survey_period_end"] == d]
            if not match.empty:
                release_dates.append(to_ts(match.iloc[0]["release_date"]))
                is_estimated.append(False)
                continue
        release_dates.append(_next_wednesday_after(d))
        is_estimated.append(True)
    data["release_date"] = release_dates
    data["release_date_estimated"] = is_estimated
    data["series_id"] = "ATL_BIE_YOY"

    out = data[
        ["series_id", "survey_period_end", "release_date", "release_date_estimated", "value"]
    ].sort_values("survey_period_end").reset_index(drop=True)
    logger.info(
        "Atlanta Fed BIE: %d obs, %s to %s",
        len(out),
        out["survey_period_end"].min().date(),
        out["survey_period_end"].max().date(),
    )
    return out
