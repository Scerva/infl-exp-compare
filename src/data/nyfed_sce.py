"""New York Fed Survey of Consumer Expectations (SCE) downloader.

Retrieves the official public chart-data workbook (an .xlsx maintained by
the NY Fed Center for Microeconomic Data) and extracts the 1-year, 3-year
and 5-year median inflation-expectation series.

Two design points required by the project spec:

1. The download URL is discovered dynamically by scraping the SCE landing
   page for its own link to the chart-data workbook (never hard-coded),
   since the New York Fed has changed this path before.
2. Every row is timestamped with BOTH ``reference_month`` (the survey
   period the number describes) and ``release_date`` (when the NY Fed
   actually published it). Only ``release_date`` may be used for
   information-set alignment with Fed communications.

RELEASE-DATE CAVEAT (read before using in event alignment):
The NY Fed does not publish a machine-readable release-date field
alongside the chart-data workbook, and the interactive SCE page is a
client-rendered SPA whose press-release archive is not reachable via
plain HTTP GET (no headless browser is used in this project - see spec
section 29). Release dates are therefore RECONSTRUCTED using a documented
rule: "the second Tuesday of the month following the reference month",
calibrated against the one directly observable data point in this
environment (as of the run date, the most recent reference month with
data present versus absent). This is accurate to within about a week in
most cases but can be off around holidays or schedule changes. Every
reconstructed row is flagged ``release_date_estimated=True``. If a
verified-date override file exists at
``config/nyfed_sce_release_dates_verified.csv`` (columns:
reference_month,release_date), those dates take precedence and are
flagged ``release_date_estimated=False``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import openpyxl
import pandas as pd

from src.utils.cache import FetchError, fetch
from src.utils.dates import to_ts
from src.utils.logging import get_logger

logger = get_logger(__name__)

SCE_LANDING_PAGE = "https://www.newyorkfed.org/microeconomics/sce"
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERIFIED_RELEASE_DATES_PATH = PROJECT_ROOT / "config" / "nyfed_sce_release_dates_verified.csv"

_XLSX_LINK_RE = re.compile(r'href="(/medialibrary/[^"]*frbny-sce-data\.xlsx)"', re.IGNORECASE)

INFLATION_SHEET = "Inflation expectations"
FIVE_YEAR_SHEET = "Five-year ahead Infl Exp"

# Column-name substrings we look for on the header row of each sheet, so we
# are robust to the sheet layout changing rather than assuming a fixed
# column index.
INFLATION_SHEET_COLUMNS = {
    "median_1yr": "Median one-year ahead expected inflation rate",
    "median_3yr": "Median three-year ahead expected inflation rate",
    "p25_1yr": "25th Percentile one-year ahead expected inflation rate",
    "p25_3yr": "25th Percentile three-year ahead expected inflation rate",
    "p75_1yr": "75th Percentile one-year ahead expected inflation rate",
    "p75_3yr": "75th Percentile three-year ahead expected inflation rate",
}
FIVE_YEAR_SHEET_COLUMNS = {
    "median_5yr": "Median five-year ahead expected inflation rate",
    "p25_5yr": "25th Percentile five-year ahead expected inflation rate",
    "p75_5yr": "75th Percentile five-year ahead expected inflation rate",
}


def discover_workbook_url(force_refresh: bool = False) -> str:
    """Scrape the SCE landing page for the current chart-data workbook URL."""
    resp = fetch(
        SCE_LANDING_PAGE,
        cache_key="sce_landing_page",
        subdir="nyfed",
        headers=BROWSER_HEADERS,
        ext="html",
        force_refresh=force_refresh,
        max_age_seconds=7 * 24 * 3600,
    )
    html = resp.text()
    m = _XLSX_LINK_RE.search(html)
    if not m:
        raise FetchError(
            "Could not find the SCE chart-data workbook link on "
            f"{SCE_LANDING_PAGE}. The page structure may have changed; "
            "inspect it manually and update src/data/nyfed_sce.py."
        )
    return "https://www.newyorkfed.org" + m.group(1)


def _find_header_row(rows: list[tuple], required_substring: str) -> int:
    for i, row in enumerate(rows):
        if any(isinstance(c, str) and required_substring in c for c in row):
            return i
    raise FetchError(f"Could not locate header row containing {required_substring!r}.")


def _parse_sheet(ws, column_map: dict[str, str]) -> pd.DataFrame:
    rows = list(ws.iter_rows(values_only=True))
    header_idx = _find_header_row(rows, next(iter(column_map.values())))
    header = rows[header_idx]
    col_positions: dict[str, int] = {}
    for out_name, needle in column_map.items():
        matches = [j for j, c in enumerate(header) if isinstance(c, str) and needle in c]
        if not matches:
            raise FetchError(f"Column {needle!r} not found in header row {header}.")
        col_positions[out_name] = matches[0]

    records = []
    for row in rows[header_idx + 1 :]:
        if row[0] is None:
            continue
        try:
            ref_month = int(row[0])
        except (TypeError, ValueError):
            continue
        if not (190001 <= ref_month <= 210012):
            continue
        rec = {"reference_month_int": ref_month}
        for out_name, pos in col_positions.items():
            rec[out_name] = row[pos] if pos < len(row) else None
        records.append(rec)

    if not records:
        raise FetchError("No data rows parsed from SCE sheet.")
    return pd.DataFrame(records)


def _reconstruct_release_date(reference_month: pd.Timestamp) -> pd.Timestamp:
    """Second Tuesday of the month following the reference month.

    See module docstring RELEASE-DATE CAVEAT for the evidence and
    limitations behind this rule.
    """
    following = (reference_month + pd.offsets.MonthBegin(1)).replace(day=1)
    first_of_month = pd.Timestamp(year=following.year, month=following.month, day=1)
    first_tuesday_offset = (1 - first_of_month.weekday()) % 7  # Tuesday == 1
    first_tuesday = first_of_month + pd.Timedelta(days=first_tuesday_offset)
    second_tuesday = first_tuesday + pd.Timedelta(days=7)
    return second_tuesday


def _load_verified_overrides() -> Optional[pd.DataFrame]:
    if not VERIFIED_RELEASE_DATES_PATH.exists():
        return None
    df = pd.read_csv(VERIFIED_RELEASE_DATES_PATH, parse_dates=["reference_month", "release_date"])
    if df.empty:
        return None
    return df


def get_sce_inflation_expectations(force_refresh: bool = False) -> pd.DataFrame:
    """Return the SCE 1y/3y/5y median inflation-expectations panel.

    Columns: reference_month, release_date, release_date_estimated,
    median_1yr, median_3yr, median_5yr, p25_1yr, p75_1yr, p25_3yr, p75_3yr,
    p25_5yr, p75_5yr.
    """
    url = discover_workbook_url(force_refresh=force_refresh)
    resp = fetch(
        url,
        cache_key="frbny-sce-data",
        subdir="nyfed",
        headers=BROWSER_HEADERS,
        ext="xlsx",
        force_refresh=force_refresh,
        max_age_seconds=7 * 24 * 3600,
    )
    wb = openpyxl.load_workbook(resp.raw_path, read_only=True, data_only=True)

    if INFLATION_SHEET not in wb.sheetnames or FIVE_YEAR_SHEET not in wb.sheetnames:
        raise FetchError(
            f"Expected sheets {INFLATION_SHEET!r} and {FIVE_YEAR_SHEET!r} not found. "
            f"Available sheets: {wb.sheetnames}. Workbook layout may have changed."
        )

    infl_df = _parse_sheet(wb[INFLATION_SHEET], INFLATION_SHEET_COLUMNS)
    five_yr_df = _parse_sheet(wb[FIVE_YEAR_SHEET], FIVE_YEAR_SHEET_COLUMNS)

    df = infl_df.merge(five_yr_df, on="reference_month_int", how="left")
    df["reference_month"] = pd.to_datetime(df["reference_month_int"].astype(str), format="%Y%m")
    df = df.drop(columns=["reference_month_int"]).sort_values("reference_month").reset_index(drop=True)

    n_before = len(df)
    df = df.drop_duplicates(subset="reference_month", keep="last")
    if len(df) != n_before:
        logger.warning("nyfed_sce: dropped %d duplicate reference_month rows", n_before - len(df))

    for col in ["median_1yr", "median_3yr", "median_5yr"]:
        if col in df.columns:
            bad = df[col].dropna()
            bad = bad[(bad < -5) | (bad > 30)]
            if len(bad):
                logger.warning("nyfed_sce: %d implausible values (<-5 or >30) in %s", len(bad), col)

    overrides = _load_verified_overrides()
    release_dates = []
    is_estimated = []
    for ref_month in df["reference_month"]:
        if overrides is not None:
            match = overrides.loc[overrides["reference_month"] == ref_month]
            if not match.empty:
                release_dates.append(to_ts(match.iloc[0]["release_date"]))
                is_estimated.append(False)
                continue
        release_dates.append(_reconstruct_release_date(ref_month))
        is_estimated.append(True)
    df["release_date"] = release_dates
    df["release_date_estimated"] = is_estimated

    logger.info(
        "nyfed_sce: %d monthly obs, %s to %s (%s)",
        len(df),
        df["reference_month"].min().date(),
        df["reference_month"].max().date(),
        "cache" if resp.from_cache else "downloaded",
    )
    cols = [
        "reference_month",
        "release_date",
        "release_date_estimated",
        "median_1yr",
        "median_3yr",
        "median_5yr",
        "p25_1yr",
        "p75_1yr",
        "p25_3yr",
        "p75_3yr",
        "p25_5yr",
        "p75_5yr",
    ]
    return df[[c for c in cols if c in df.columns]]
