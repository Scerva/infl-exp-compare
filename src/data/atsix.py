"""Philadelphia Fed Aruoba Term Structure of Inflation Expectations (ATSIX).

ATSIX combines the SPF and Blue Chip Economic Indicators/Financial
Forecasts surveys via a factor model (Aruoba 2016) into a smooth monthly
term structure of inflation expectations spanning roughly 3 to 120
months. It therefore belongs to the model/professional family (spec
section 3G), not an independent primitive survey -- most of its
information content is a smoothed/interpolated combination of series we
also pull directly (SPF) or via proxy (Blue Chip is not separately
ingested in this project).

The download URL is discovered dynamically from the official ATSIX page.

RELEASE TIMING: the ATSIX page states explicitly (as of the page content
retrieved for this project): "The ATSIX is updated around the 20th of the
month after the source data are released." We use that literal,
officially-stated cadence to construct release_date = the 20th of the
calendar month following each observation's reference month. This is
conservative (errs toward a later, safer release date, avoiding
look-ahead bias) but is still a general rule rather than a per-observation
verified timestamp, so every row is flagged release_date_estimated=True.
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

ATSIX_PAGE = "https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/atsix"
_XLSX_HREF_RE = re.compile(r'href="([^"]*ATSIX_Vintages\.xlsx[^"]*)"', re.IGNORECASE)

# Selected horizons of primary interest (spec section 3G); the full grid
# (3..120 months) is retained too for slope calculations elsewhere.
SELECTED_HORIZONS_MONTHS = [12, 24, 36, 60, 120]


def discover_workbook_url(force_refresh: bool = False) -> str:
    resp = fetch(
        ATSIX_PAGE,
        cache_key="atsix_page",
        subdir="atsix",
        headers=BROWSER_HEADERS,
        ext="html",
        force_refresh=force_refresh,
        max_age_seconds=7 * 24 * 3600,
    )
    m = _XLSX_HREF_RE.search(resp.text())
    if not m:
        raise FetchError(f"Could not find ATSIX_Vintages.xlsx link on {ATSIX_PAGE}.")
    href = m.group(1).replace("&amp;", "&")
    return href if href.startswith("http") else "https://www.philadelphiafed.org" + href


def _estimate_release_date(reference_month: pd.Timestamp) -> pd.Timestamp:
    following = reference_month + pd.DateOffset(months=1)
    return pd.Timestamp(year=following.year, month=following.month, day=20)


def get_atsix(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch the full ATSIX term structure (all months x all horizons),
    long-format.

    Returns columns: reference_month, release_date, release_date_estimated,
    horizon_months, value.
    """
    url = discover_workbook_url(force_refresh=force_refresh)
    resp = fetch(
        url,
        cache_key="atsix_vintages",
        subdir="atsix",
        headers=BROWSER_HEADERS,
        ext="xlsx",
        force_refresh=force_refresh,
        max_age_seconds=7 * 24 * 3600,
    )
    repair_xlsx_in_place(resp.raw_path)
    xl = pd.ExcelFile(resp.raw_path)
    if "InfExp" not in xl.sheet_names:
        raise FetchError(f"Expected 'InfExp' sheet in {url}, got {xl.sheet_names}")
    wide = xl.parse("InfExp")
    wide.columns = [str(c).strip() for c in wide.columns]
    if "_date_" not in wide.columns:
        raise FetchError(f"Expected '_date_' column in ATSIX InfExp sheet, got {list(wide.columns)}")

    horizon_cols = [c for c in wide.columns if re.match(r"^infexp\d+$", c)]
    if not horizon_cols:
        raise FetchError(f"No infexpN horizon columns found in {url}: {list(wide.columns)}")

    long_df = wide.melt(id_vars=["_date_"], value_vars=horizon_cols, var_name="horizon_col", value_name="value")
    long_df["horizon_months"] = long_df["horizon_col"].str.replace("infexp", "", regex=False).astype(int)
    long_df["reference_month"] = pd.to_datetime(long_df["_date_"])
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])

    long_df["release_date"] = long_df["reference_month"].apply(_estimate_release_date).apply(to_ts)
    long_df["release_date_estimated"] = True

    out = long_df[["reference_month", "release_date", "release_date_estimated", "horizon_months", "value"]]
    out = out.sort_values(["reference_month", "horizon_months"]).reset_index(drop=True)
    logger.info(
        "ATSIX: %d obs (%d months x %d horizons), %s to %s",
        len(out),
        out["reference_month"].nunique(),
        out["horizon_months"].nunique(),
        out["reference_month"].min().date(),
        out["reference_month"].max().date(),
    )
    return out


def get_atsix_selected_horizons(force_refresh: bool = False) -> pd.DataFrame:
    """Wide panel with just SELECTED_HORIZONS_MONTHS, plus computed slopes."""
    long_df = get_atsix(force_refresh=force_refresh)
    sel = long_df[long_df["horizon_months"].isin(SELECTED_HORIZONS_MONTHS)]
    wide = sel.pivot_table(
        index=["reference_month", "release_date", "release_date_estimated"],
        columns="horizon_months",
        values="value",
    ).reset_index()
    wide.columns = [
        f"atsix_{c}m" if isinstance(c, int) else c for c in wide.columns
    ]
    if "atsix_120m" in wide.columns and "atsix_12m" in wide.columns:
        wide["atsix_slope_10y_1y"] = wide["atsix_120m"] - wide["atsix_12m"]
    if "atsix_60m" in wide.columns and "atsix_12m" in wide.columns:
        wide["atsix_slope_5y_1y"] = wide["atsix_60m"] - wide["atsix_12m"]
    if "atsix_120m" in wide.columns and "atsix_60m" in wide.columns:
        wide["atsix_slope_10y_5y"] = wide["atsix_120m"] - wide["atsix_60m"]
    return wide
