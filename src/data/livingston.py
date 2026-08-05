"""Philadelphia Fed Livingston Survey downloader.

Oldest continuous US professional-forecaster survey (June 1946 onward for
several variables). We use it only for the long-run 10-year CPI inflation
expectation (``CPI_10Y`` column of the official ``medians.xlsx`` workbook)
as a low-frequency (semiannual, June/December) robustness/cross-check
series -- see spec section 3F, which explicitly says this should not
dominate high-frequency regressions given its frequency.

Both download URLs (data workbook and release-dates workbook) are
discovered dynamically from the official Livingston pages rather than
hard-coded, since Philadelphia Fed file hashes change on republication.

RELEASE-DATE HANDLING:
The official ``livingston-release-dates.xlsx`` has two columns per survey:
an "Exact" release date (known for recent decades) and a "Not After"
upper-bound date (known back to the survey's 1946 inception, per the
Philadelphia Fed's own historical reconstruction). We use "Exact" where
available (``release_date_estimated=False``); otherwise we use "Not
After" as a conservative (look-ahead-safe) release date, flagged
``release_date_estimated=True`` since it is a bound, not the true date.

KNOWN UPSTREAM QUIRK: the "Survey" column in that workbook mis-parses
two-digit years for 1946-1999 surveys (e.g. "6/1/46" is read back as
2046-06-01 instead of 1946-06-01) due to Excel/pandas' default century
pivot. We detect and correct any parsed survey date more than one year in
the future.
"""
from __future__ import annotations

import re

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

LIVINGSTON_MAIN_PAGE = "https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/livingston-survey"
LIVINGSTON_HIST_PAGE = "https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/livingston-historical-data"

_RELEASE_DATES_HREF_RE = re.compile(r'href="([^"]*livingston-release-dates\.xlsx[^"]*)"', re.IGNORECASE)
_MEDIANS_HREF_RE = re.compile(r'href="([^"]*/medians\.xlsx[^"]*)"', re.IGNORECASE)


def _resolve(href: str) -> str:
    href = href.replace("&amp;", "&")
    return href if href.startswith("http") else "https://www.philadelphiafed.org" + href


def discover_release_dates_url(force_refresh: bool = False) -> str:
    resp = fetch(
        LIVINGSTON_MAIN_PAGE,
        cache_key="livingston_main_page",
        subdir="livingston",
        headers=BROWSER_HEADERS,
        ext="html",
        force_refresh=force_refresh,
        max_age_seconds=7 * 24 * 3600,
    )
    m = _RELEASE_DATES_HREF_RE.search(resp.text())
    if not m:
        raise FetchError(f"Could not find livingston-release-dates.xlsx link on {LIVINGSTON_MAIN_PAGE}.")
    return _resolve(m.group(1))


def discover_medians_url(force_refresh: bool = False) -> str:
    resp = fetch(
        LIVINGSTON_HIST_PAGE,
        cache_key="livingston_historical_page",
        subdir="livingston",
        headers=BROWSER_HEADERS,
        ext="html",
        force_refresh=force_refresh,
        max_age_seconds=7 * 24 * 3600,
    )
    m = _MEDIANS_HREF_RE.search(resp.text())
    if not m:
        raise FetchError(f"Could not find medians.xlsx link on {LIVINGSTON_HIST_PAGE}.")
    return _resolve(m.group(1))


def _fix_two_digit_year_overflow(dates: pd.Series) -> pd.Series:
    cutoff = pd.Timestamp.now() + pd.Timedelta(days=365)
    fixed = dates.copy()
    overflow_mask = fixed.notna() & (fixed > cutoff)
    fixed.loc[overflow_mask] = fixed.loc[overflow_mask].apply(lambda d: d.replace(year=d.year - 100))
    if overflow_mask.any():
        logger.info("livingston: corrected %d two-digit-year-overflow survey dates", overflow_mask.sum())
    return fixed


def get_livingston_release_dates(force_refresh: bool = False) -> pd.DataFrame:
    url = discover_release_dates_url(force_refresh=force_refresh)
    resp = fetch(
        url,
        cache_key="livingston_release_dates",
        subdir="livingston",
        headers=BROWSER_HEADERS,
        ext="xlsx",
        force_refresh=force_refresh,
        max_age_seconds=7 * 24 * 3600,
    )
    repair_xlsx_in_place(resp.raw_path)
    xl = pd.ExcelFile(resp.raw_path)
    if "Dates" not in xl.sheet_names:
        raise FetchError(f"Expected 'Dates' sheet in {url}, got {xl.sheet_names}")
    df = xl.parse("Dates")
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        if c == "Survey":
            rename[c] = "survey_date"
        elif "Release Date" in c:
            rename[c] = "exact_release_date"
        elif c.startswith("Unnamed"):
            rename[c] = "not_after_date"
    df = df.rename(columns=rename)
    df = df.dropna(subset=["survey_date"]).reset_index(drop=True)

    df["survey_date"] = pd.to_datetime(df["survey_date"])
    df["survey_date"] = _fix_two_digit_year_overflow(df["survey_date"])
    df["exact_release_date"] = pd.to_datetime(df.get("exact_release_date"), errors="coerce")
    df["not_after_date"] = pd.to_datetime(df.get("not_after_date"), errors="coerce")

    df["release_date"] = df["exact_release_date"].fillna(df["not_after_date"])
    df["release_date_estimated"] = df["exact_release_date"].isna()
    return df[["survey_date", "release_date", "release_date_estimated"]]


def get_livingston_series(variable: str = "CPI_10Y", force_refresh: bool = False) -> pd.DataFrame:
    """Fetch the Livingston Survey median long-run CPI expectation.

    Returns columns: series_id, reference_date, release_date,
    release_date_estimated, value.
    """
    medians_url = discover_medians_url(force_refresh=force_refresh)
    resp = fetch(
        medians_url,
        cache_key="livingston_medians",
        subdir="livingston",
        headers=BROWSER_HEADERS,
        ext="xlsx",
        force_refresh=force_refresh,
        max_age_seconds=7 * 24 * 3600,
    )
    repair_xlsx_in_place(resp.raw_path)
    xl = pd.ExcelFile(resp.raw_path)
    if "CPI" not in xl.sheet_names:
        raise FetchError(f"Expected 'CPI' sheet in {medians_url}, got {xl.sheet_names}")
    cpi_df = xl.parse("CPI")
    cpi_df.columns = [str(c).strip() for c in cpi_df.columns]
    if variable not in cpi_df.columns:
        raise FetchError(f"Column {variable!r} not found in Livingston CPI sheet: {list(cpi_df.columns)}")

    df = cpi_df[["Date", variable]].rename(columns={"Date": "reference_date", variable: "value"})
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    if df.empty:
        raise FetchError(f"No non-missing observations for Livingston {variable}.")

    release_dates = get_livingston_release_dates(force_refresh=force_refresh)
    merged = df.merge(release_dates, left_on="reference_date", right_on="survey_date", how="left")

    missing = merged["release_date"].isna()
    if missing.any():
        # Documented fallback: Livingston is historically released mid-month
        # in the month following the June/December survey date.
        merged.loc[missing, "release_date"] = merged.loc[missing, "reference_date"] + pd.DateOffset(months=1, days=14)
        merged.loc[missing, "release_date_estimated"] = True
        logger.info("livingston: %d/%d obs used fallback (no release-date record found)", missing.sum(), len(merged))

    merged["release_date"] = merged["release_date"].apply(to_ts)
    merged["series_id"] = f"LIV_{variable}"

    out = merged[["series_id", "reference_date", "release_date", "release_date_estimated", "value"]]
    out = out.sort_values("reference_date").reset_index(drop=True)
    logger.info(
        "Livingston %s: %d obs, %s to %s",
        variable,
        len(out),
        out["reference_date"].min().date(),
        out["reference_date"].max().date(),
    )
    return out
