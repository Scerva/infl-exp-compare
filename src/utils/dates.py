"""Date/timing utilities underpinning the information-set-alignment framework.

The single most important correctness property in this project (see spec
section 9) is: for a Fed communication at timestamp ``t``, every explanatory
variable must reflect information *actually publicly available strictly
before* ``t``. The helpers here are the load-bearing primitives for that:
``asof_available`` and ``asof_merge_strict`` never look at an observation's
reference period, only its ``release_date``/``available_from`` timestamp.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

US_BUSINESS_DAY = BDay()


def to_ts(x) -> pd.Timestamp:
    """Coerce to a tz-naive pandas Timestamp (midnight if date-only)."""
    ts = pd.Timestamp(x)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def business_days_before(anchor, n: int) -> pd.Timestamp:
    return to_ts(anchor) - n * US_BUSINESS_DAY


def business_days_after(anchor, n: int) -> pd.Timestamp:
    return to_ts(anchor) + n * US_BUSINESS_DAY


def prior_close_date(event_ts) -> pd.Timestamp:
    """Most recent market-close date strictly before an intraday event.

    Used for daily market series (breakevens, Treasury yields) when the Fed
    communication itself could move that day's closing price: we must use
    the close *before* the communication, not the same-day close.
    """
    event_ts = to_ts(event_ts)
    d = event_ts.normalize()
    # If the event has no time component (date-only), treat "prior close" as
    # the close on the previous calendar day's most recent business day,
    # since we cannot tell if the communication happened before/after that
    # day's close. This is the conservative (look-ahead-safe) choice.
    if event_ts == d:
        candidate = d - pd.Timedelta(days=1)
    else:
        candidate = d
    while candidate.weekday() >= 5:
        candidate -= pd.Timedelta(days=1)
    return candidate


def asof_available(
    panel: pd.DataFrame,
    value_col: str,
    availability_col: str,
    as_of: pd.Timestamp,
    strict: bool = True,
) -> Optional[float]:
    """Latest ``value_col`` whose ``availability_col`` is (strictly) <= as_of.

    ``panel`` need not be sorted; this filters + takes the max availability.
    Returns None (not NaN silently) when nothing qualifies, so callers can
    distinguish "no info yet" from "value is legitimately NaN".
    """
    if panel.empty:
        return None
    as_of = to_ts(as_of)
    avail = pd.to_datetime(panel[availability_col])
    mask = (avail < as_of) if strict else (avail <= as_of)
    eligible = panel.loc[mask]
    if eligible.empty:
        return None
    idx = pd.to_datetime(eligible[availability_col]).idxmax()
    return eligible.loc[idx, value_col]


def asof_merge_strict(
    events: pd.DataFrame,
    event_time_col: str,
    series: pd.DataFrame,
    availability_col: str,
    value_cols: list[str],
    prefix: str,
    strict: bool = True,
) -> pd.DataFrame:
    """Vectorized as-of merge: attach the latest known ``value_cols`` from
    ``series`` as of each event, using ``pandas.merge_asof`` with
    ``availability_col`` as the series timestamp (never the reference date).

    Both frames are sorted internally; original ``events`` row order is
    restored on return. Column names in the output are ``f"{prefix}_{col}"``.
    """
    left = events[[event_time_col]].copy()
    left["_evt_ts"] = pd.to_datetime(left[event_time_col])
    left["_orig_order"] = np.arange(len(left))
    left = left.sort_values("_evt_ts")

    right = series[[availability_col] + value_cols].copy()
    right["_avail_ts"] = pd.to_datetime(right[availability_col])
    right = right.dropna(subset=["_avail_ts"]).sort_values("_avail_ts")
    right = right.drop_duplicates(subset="_avail_ts", keep="last")

    merged = pd.merge_asof(
        left,
        right,
        left_on="_evt_ts",
        right_on="_avail_ts",
        direction="backward",
        allow_exact_matches=not strict,
    )
    merged = merged.sort_values("_orig_order")
    rename = {c: f"{prefix}_{c}" for c in value_cols}
    rename[availability_col] = f"{prefix}_{availability_col}"
    merged = merged.rename(columns=rename)
    out_cols = ["_orig_order"] + [f"{prefix}_{c}" for c in value_cols] + [f"{prefix}_{availability_col}"]
    result = merged[out_cols].reset_index(drop=True)
    events_out = events.reset_index(drop=True).copy()
    for c in out_cols[1:]:
        events_out[c] = result[c].values
    return events_out
