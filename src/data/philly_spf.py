"""Philadelphia Fed Survey of Professional Forecasters (SPF) downloader.

Two official, machine-readable sources are combined:

1. Per-variable data workbooks, discovered dynamically from each series'
   own landing page (e.g. philadelphiafed.org/surveys-and-data/cpi10)
   rather than a hard-coded file hash, per spec section 3E. Statistic
   files follow the observed naming convention
   ``{Statistic}_{VARIABLE}_Level.xlsx`` (Median_CPI10_Level.xlsx,
   Mean_CPI10_Level.xlsx, ...); we still discover the exact link+hash from
   the page rather than assuming the convention holds indefinitely.

2. The official ``spf-release-dates.txt`` (true deadline + news-release
   date for every survey since 1990:Q2, explicitly stated by the
   Philadelphia Fed as unknown/unreliable before that). This gives EXACT
   release dates -- no reconstruction needed for 1990:Q2 onward. Earlier
   quarters (relevant only for CPI10 back to 1991/1992, so rarely
   triggered) fall back to a documented mid-quarter estimate, flagged
   ``release_date_estimated=True``.

NOTE ON A KNOWN UPSTREAM DATA BUG: the SPF workbooks are exported via SAS
and several have a malformed ``docProps/core.xml`` timestamp
(``T 2:35:20`` instead of ``T02:35:20``) that crashes openpyxl before it
reads any worksheet data. See src/utils/xlsx_repair.py, which is applied
to every downloaded workbook here.
"""
from __future__ import annotations

import re
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

SERIES_PAGE_TMPL = "https://www.philadelphiafed.org/surveys-and-data/{slug}"
RELEASE_DATES_URL = (
    "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
    "survey-of-professional-forecasters/spf-release-dates.txt"
)

# Catalog series_id -> (page slug, workbook file-variable code, value column).
# The long-horizon files (CPI10/CPI5YR/PCE10/PCE5YR) hold one value column
# per row (a single point forecast). The near-term files (CPI/PCE) instead
# hold one column per horizon (CPI1..CPI6 = current..5-quarters-ahead
# annualized quarterly rate; CPIA/B/C = current/next/next+1 calendar-year
# average). We proxy "one-year-ahead" with the *B column (next calendar
# year average), since a single quarter-ahead annualized rate is far
# noisier and not what "one-year-ahead inflation expectations" usually
# means in the literature. value_col=None means "auto-detect the sole
# non-YEAR/QUARTER column".
SPF_VARIABLE_CONFIG = {
    "CPI10": {"slug": "cpi10", "file_variable": "CPI10", "value_col": None},
    "CPI5YR": {"slug": "cpi5yr", "file_variable": "CPI5YR", "value_col": None},
    "PCE10": {"slug": "pce10", "file_variable": "PCE10", "value_col": None},
    "PCE5YR": {"slug": "pce5yr", "file_variable": "PCE5YR", "value_col": None},
    "CPI1YR": {"slug": "cpi-spf", "file_variable": "CPI", "value_col": "CPIB"},
    "PCE1YR": {"slug": "pce", "file_variable": "PCE", "value_col": "PCEB"},
}
# Backward-compatible alias used by discover_workbook_url callers/tests.
VARIABLE_SLUGS = {k: v["slug"] for k, v in SPF_VARIABLE_CONFIG.items()}

_XLSX_HREF_RE_TMPL = r'href="([^"]*{stat}_{var}_Level\.xlsx[^"]*)"'

_RELEASE_LINE_RE = re.compile(
    r"^\s*(?:(?P<year>\d{4})\s+)?Q(?P<quarter>\d)\s+"
    r"(?P<deadline>\d{1,2}/\d{1,2}/\d{2,4})\*{0,3}\s+"
    r"(?P<release>\d{1,2}/\d{1,2}/\d{2,4})\*{0,3}"
)


def discover_workbook_url(variable: str, statistic: str, force_refresh: bool = False) -> str:
    if variable not in SPF_VARIABLE_CONFIG:
        raise ValueError(f"Unknown SPF variable {variable!r}; known: {list(SPF_VARIABLE_CONFIG)}")
    cfg = SPF_VARIABLE_CONFIG[variable]
    slug, file_variable = cfg["slug"], cfg["file_variable"]
    page_url = SERIES_PAGE_TMPL.format(slug=slug)
    resp = fetch(
        page_url,
        cache_key=f"page_{slug}",
        subdir="philly_spf",
        headers=BROWSER_HEADERS,
        ext="html",
        force_refresh=force_refresh,
        max_age_seconds=7 * 24 * 3600,
    )
    html = resp.text()
    stat_prefix = {"median": "Median", "mean": "Mean"}[statistic.lower()]
    pattern = re.compile(_XLSX_HREF_RE_TMPL.format(stat=stat_prefix, var=file_variable), re.IGNORECASE)
    m = pattern.search(html)
    if not m:
        raise FetchError(
            f"Could not find {stat_prefix}_{variable}_Level.xlsx link on {page_url}. "
            "Page structure may have changed."
        )
    href = m.group(1).replace("&amp;", "&")
    if href.startswith("http"):
        return href
    return "https://www.philadelphiafed.org" + href


def _parse_release_dates_txt(text: str) -> pd.DataFrame:
    records = []
    current_year: Optional[int] = None
    for line in text.splitlines():
        m = _RELEASE_LINE_RE.match(line)
        if not m:
            continue
        if m.group("year"):
            current_year = int(m.group("year"))
        if current_year is None:
            continue
        quarter = int(m.group("quarter"))
        release = pd.to_datetime(m.group("release"), format="%m/%d/%y", errors="coerce")
        if pd.isna(release):
            release = pd.to_datetime(m.group("release"), errors="coerce")
        if pd.isna(release):
            continue
        records.append({"year": current_year, "quarter": quarter, "release_date": release})
    if not records:
        raise FetchError("Parsed zero rows from SPF release-dates file; format may have changed.")
    return pd.DataFrame(records)


def get_spf_release_dates(force_refresh: bool = False) -> pd.DataFrame:
    resp = fetch(
        RELEASE_DATES_URL,
        cache_key="spf_release_dates",
        subdir="philly_spf",
        headers=BROWSER_HEADERS,
        ext="txt",
        force_refresh=force_refresh,
        max_age_seconds=7 * 24 * 3600,
    )
    return _parse_release_dates_txt(resp.text())


def _quarter_start(year: int, quarter: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=3 * (quarter - 1) + 1, day=1)


def _estimate_release_date(year: int, quarter: int) -> pd.Timestamp:
    """Fallback for quarters before the official release-dates file begins
    (pre-1990:Q2): Philly Fed's modern pattern is release in the second
    half of the survey quarter's second month; we use the 20th of that
    month as a documented, clearly-flagged estimate.
    """
    month = 3 * (quarter - 1) + 2
    return pd.Timestamp(year=year, month=month, day=20)


def get_spf_series(variable: str, statistic: str = "median", force_refresh: bool = False) -> pd.DataFrame:
    """Fetch one SPF variable/statistic series with exact (where available)
    release-date alignment.

    Returns columns: series_id, statistic, reference_quarter, release_date,
    release_date_estimated, value.
    """
    url = discover_workbook_url(variable, statistic, force_refresh=force_refresh)
    resp = fetch(
        url,
        cache_key=f"{statistic}_{variable}_Level",
        subdir="philly_spf",
        headers=BROWSER_HEADERS,
        ext="xlsx",
        force_refresh=force_refresh,
        max_age_seconds=7 * 24 * 3600,
    )
    repair_xlsx_in_place(resp.raw_path)

    xl = pd.ExcelFile(resp.raw_path)
    sheet = next((s for s in xl.sheet_names if "level" in s.lower()), xl.sheet_names[0])
    df = xl.parse(sheet)

    df.columns = [str(c).strip().upper() for c in df.columns]
    if "YEAR" not in df.columns or "QUARTER" not in df.columns:
        raise FetchError(f"Expected YEAR/QUARTER columns in {url}, got {list(df.columns)}")
    configured_value_col = SPF_VARIABLE_CONFIG[variable]["value_col"]
    if configured_value_col is not None:
        if configured_value_col not in df.columns:
            raise FetchError(
                f"Configured value column {configured_value_col!r} not found in {url}: {list(df.columns)}"
            )
        value_col = configured_value_col
    else:
        other_cols = [c for c in df.columns if c not in ("YEAR", "QUARTER")]
        if len(other_cols) != 1:
            raise FetchError(
                f"Expected exactly one value column (excl. YEAR/QUARTER) in {url}, "
                f"got {other_cols}. Set an explicit value_col in SPF_VARIABLE_CONFIG."
            )
        value_col = other_cols[0]

    df = df[["YEAR", "QUARTER", value_col]].rename(columns={value_col: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    if df.empty:
        raise FetchError(f"No non-missing observations parsed for {variable}/{statistic} from {url}")

    release_dates = get_spf_release_dates(force_refresh=force_refresh)
    merged = df.merge(release_dates, left_on=["YEAR", "QUARTER"], right_on=["year", "quarter"], how="left")
    merged["release_date_estimated"] = merged["release_date"].isna()
    missing_mask = merged["release_date"].isna()
    if missing_mask.any():
        merged.loc[missing_mask, "release_date"] = merged.loc[missing_mask].apply(
            lambda r: _estimate_release_date(int(r["YEAR"]), int(r["QUARTER"])), axis=1
        )
        logger.info(
            "%s/%s: %d/%d quarters used estimated release dates (pre-1990:Q2 or gap in official file)",
            variable,
            statistic,
            missing_mask.sum(),
            len(merged),
        )

    merged["reference_quarter"] = merged.apply(lambda r: _quarter_start(int(r["YEAR"]), int(r["QUARTER"])), axis=1)
    merged["series_id"] = variable
    merged["statistic"] = statistic.lower()
    merged["release_date"] = merged["release_date"].apply(to_ts)

    out = merged[
        ["series_id", "statistic", "reference_quarter", "release_date", "release_date_estimated", "value"]
    ].sort_values("reference_quarter").reset_index(drop=True)

    logger.info(
        "SPF %s/%s: %d obs, %s to %s",
        variable,
        statistic,
        len(out),
        out["reference_quarter"].min().date(),
        out["reference_quarter"].max().date(),
    )
    return out
