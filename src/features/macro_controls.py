"""Macro-economic control panel (spec section 11).

Pulls the baseline macro series a Fed communication's tone might react to
contemporaneously, so that a positive univariate association between an
inflation-expectations measure and hawkish stance is not mechanically just
"the Fed reacted to actual inflation, which is correlated with
expectations" -- see section 24 (reverse causality/endogeneity) and
section 13 (baseline model must control for the macro environment).

RELEASE-TIMING NOTE: FRED's default (non-ALFRED) query returns each
observation under its *reference* date, not the date that vintage was
first published, and pulling true first-release ("real-time") vintages
would require a full ALFRED vintage-crawl per series. As a documented,
transparent middle ground consistent with how every other module in this
project handles the same trade-off (see nyfed_sce.py, atsix.py,
atlanta_bie.py), we approximate each series' publication lag using its
well-known, official release-calendar convention (e.g. BLS CPI ~13 days
after month-end, BLS Employment Situation on the first Friday of the
following month, BEA GDP advance estimate ~25 days after quarter-end).
Every row is flagged ``release_date_estimated=True`` for exactly this
reason. Daily market series (yields, spot prices) need no such estimate:
FRED publishes those same-day/next-business-day, which we encode directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.data.fred import get_fred_series
from src.utils.dates import to_ts
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MacroSeriesDef:
    fred_id: str
    frequency: str  # daily | weekly | monthly | quarterly
    lag_rule: str  # same_day | next_business_day | first_friday | fixed_day_of_month
    months_after: int = 1
    release_day_of_month: Optional[int] = None


MACRO_SERIES: dict[str, MacroSeriesDef] = {
    "CPI_HEADLINE": MacroSeriesDef("CPIAUCSL", "monthly", "fixed_day_of_month", 1, 13),
    "CPI_CORE": MacroSeriesDef("CPILFESL", "monthly", "fixed_day_of_month", 1, 13),
    "PCE_HEADLINE": MacroSeriesDef("PCEPI", "monthly", "fixed_day_of_month", 1, 28),
    "PCE_CORE": MacroSeriesDef("PCEPILFE", "monthly", "fixed_day_of_month", 1, 28),
    "UNRATE": MacroSeriesDef("UNRATE", "monthly", "first_friday", 1),
    "PAYROLLS": MacroSeriesDef("PAYEMS", "monthly", "first_friday", 1),
    "AVG_HOURLY_EARNINGS": MacroSeriesDef("CES0500000003", "monthly", "first_friday", 1),
    "REAL_GDP": MacroSeriesDef("GDPC1", "quarterly", "fixed_day_of_month", 3, 26),
    "INDPRO": MacroSeriesDef("INDPRO", "monthly", "fixed_day_of_month", 1, 16),
    "OIL_WTI": MacroSeriesDef("DCOILWTICO", "daily", "next_business_day"),
    "GASOLINE": MacroSeriesDef("GASREGW", "weekly", "next_business_day"),
    "TREASURY_2Y": MacroSeriesDef("DGS2", "daily", "same_day"),
    "TREASURY_10Y": MacroSeriesDef("DGS10", "daily", "same_day"),
    "FED_FUNDS_EFFECTIVE": MacroSeriesDef("DFF", "daily", "next_business_day"),
    "FED_TARGET_UPPER": MacroSeriesDef("DFEDTARU", "daily", "same_day"),
    "FED_TARGET_LOWER": MacroSeriesDef("DFEDTARL", "daily", "same_day"),
    "FIN_CONDITIONS_NFCI": MacroSeriesDef("NFCI", "weekly", "next_business_day"),
    "SP500": MacroSeriesDef("SP500", "daily", "same_day"),
}


def _first_friday_on_or_after(d: pd.Timestamp) -> pd.Timestamp:
    days_ahead = (4 - d.weekday()) % 7  # Friday == 4
    return d + pd.Timedelta(days=days_ahead)


def _compute_release_date(reference_date: pd.Timestamp, series_def: MacroSeriesDef) -> tuple[pd.Timestamp, bool]:
    if series_def.lag_rule == "same_day":
        return reference_date, False
    if series_def.lag_rule == "next_business_day":
        d = reference_date + pd.Timedelta(days=1)
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        return d, False
    if series_def.lag_rule == "first_friday":
        target_month = reference_date + pd.DateOffset(months=series_def.months_after)
        month_start = pd.Timestamp(year=target_month.year, month=target_month.month, day=1)
        return _first_friday_on_or_after(month_start), True
    if series_def.lag_rule == "fixed_day_of_month":
        target_month = reference_date + pd.DateOffset(months=series_def.months_after)
        return pd.Timestamp(year=target_month.year, month=target_month.month, day=series_def.release_day_of_month), True
    raise ValueError(f"Unknown lag_rule {series_def.lag_rule!r}")


def get_macro_series(name: str, start_date: Optional[str] = None, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch one macro control series with an estimated release_date.

    Returns columns: series_id, reference_date, release_date,
    release_date_estimated, value.
    """
    if name not in MACRO_SERIES:
        raise KeyError(f"Unknown macro series {name!r}; known: {list(MACRO_SERIES)}")
    series_def = MACRO_SERIES[name]
    raw = get_fred_series(series_def.fred_id, start_date=start_date, force_refresh=force_refresh)
    raw = raw.dropna(subset=["value"]).rename(columns={"date": "reference_date"})

    release_dates, is_estimated = [], []
    for d in raw["reference_date"]:
        rd, est = _compute_release_date(d, series_def)
        release_dates.append(to_ts(rd))
        is_estimated.append(est)
    raw["release_date"] = release_dates
    raw["release_date_estimated"] = is_estimated
    raw["series_id"] = name

    out = raw[["series_id", "reference_date", "release_date", "release_date_estimated", "value"]]
    logger.info(
        "macro %s (%s): %d obs, %s to %s",
        name,
        series_def.fred_id,
        len(out),
        out["reference_date"].min().date(),
        out["reference_date"].max().date(),
    )
    return out.reset_index(drop=True)


def get_macro_panel(names: Optional[list[str]] = None, start_date: Optional[str] = None, force_refresh: bool = False) -> pd.DataFrame:
    """Long-format panel stacking every requested macro series."""
    names = names or list(MACRO_SERIES)
    frames = [get_macro_series(n, start_date=start_date, force_refresh=force_refresh) for n in names]
    return pd.concat(frames, ignore_index=True)
