"""Transform primitives for inflation-expectations series (spec section 10).

Because CPI, PCE, "prices in general," and inflation compensation are not
identical concepts (see config/sources.yaml's inflation_concept and
target_anchor fields), the default input for any cross-source comparison
should be a STANDARDIZED transform (z-score, standardized change), not
the raw level -- section 10 is explicit about this. This module builds
every transform generically off a (date, value) series plus a frequency
label, so the same code handles a daily market series, a monthly survey,
a quarterly SPF release, and a semiannual Livingston observation without
one-off logic per source.

Window sizes below (roughly one trailing year of observations at each
series' native frequency) are the "trailing 1-year" the spec asks for
explicitly for market series (section 3A) and generalized to every other
frequency for consistency.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

FREQUENCY_CONFIG: dict[str, dict] = {
    "daily": {"zscore_window": 252, "min_periods_fraction": 0.5, "change_periods": {"1d": 1, "5d": 5, "20d": 20}},
    "weekly": {"zscore_window": 52, "min_periods_fraction": 0.5, "change_periods": {"1w": 1, "4w": 4}},
    "monthly": {"zscore_window": 12, "min_periods_fraction": 0.5, "change_periods": {"1m": 1, "3m": 3, "12m": 12}},
    "quarterly": {"zscore_window": 4, "min_periods_fraction": 0.5, "change_periods": {"1q": 1, "4q": 4}},
    "semiannual": {"zscore_window": 2, "min_periods_fraction": 1.0, "change_periods": {"1p": 1}},
}


def _rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    def _pct_rank_last(x: np.ndarray) -> float:
        if len(x) < 2:
            return np.nan
        return float((x < x[-1]).sum() + 0.5 * (x == x[-1]).sum()) / len(x)

    return series.rolling(window, min_periods=min_periods).apply(_pct_rank_last, raw=True)


def transform_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    frequency: str,
    target_anchor: Optional[float] = None,
) -> pd.DataFrame:
    """Add level/change/standardized/deviation/zscore/percentile columns.

    Returns a new DataFrame (input sorted by date_col, deduplicated on
    date_col keeping the last observation) with these added columns,
    all prefixed with ``{value_col}_``:
      level, change_<horizon>... , zscore, dev_from_trailing_mean,
      pctile, change_zscore, and (if target_anchor given) dev_from_target.
    """
    cfg = FREQUENCY_CONFIG.get(frequency, FREQUENCY_CONFIG["monthly"])
    window = cfg["zscore_window"]
    min_periods = max(2, int(np.ceil(window * cfg["min_periods_fraction"])))

    out = df.sort_values(date_col).drop_duplicates(subset=date_col, keep="last").reset_index(drop=True)
    v = out[value_col]

    out[f"{value_col}_level"] = v

    change_cols = []
    for label, periods in cfg["change_periods"].items():
        col = f"{value_col}_change_{label}"
        out[col] = v.diff(periods)
        change_cols.append(col)

    roll_mean = v.rolling(window, min_periods=min_periods).mean()
    roll_std = v.rolling(window, min_periods=min_periods).std()
    out[f"{value_col}_zscore"] = (v - roll_mean) / roll_std.replace(0, np.nan)
    out[f"{value_col}_dev_from_trailing_mean"] = v - roll_mean
    out[f"{value_col}_pctile"] = _rolling_percentile(v, window, min_periods)

    if change_cols:
        primary_change = out[change_cols[0]]
        chg_roll_std = primary_change.rolling(window, min_periods=min_periods).std()
        out[f"{value_col}_change_zscore"] = primary_change / chg_roll_std.replace(0, np.nan)

    if target_anchor is not None:
        out[f"{value_col}_dev_from_target"] = v - target_anchor

    return out


def compute_slope(df: pd.DataFrame, short_col: str, long_col: str, slope_name: str) -> pd.DataFrame:
    """Forward-vs-spot / term-structure slope: long-horizon minus
    short-horizon expectation (spec section 3A/3G, e.g. 10y-1y, 5y-1y).
    """
    out = df.copy()
    out[slope_name] = out[long_col] - out[short_col]
    return out


def anchoring_gap(df: pd.DataFrame, value_col: str, target_anchor: float, out_col: Optional[str] = None) -> pd.DataFrame:
    """abs(long_run_expectation - target_consistent_level) (spec section 10).
    Kept separate from dev_from_target (which is signed) since "how far
    from anchored" is naturally an absolute-value question.
    """
    out = df.copy()
    out_col = out_col or f"{value_col}_anchoring_gap"
    out[out_col] = (out[value_col] - target_anchor).abs()
    return out
