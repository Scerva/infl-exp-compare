"""Information-set-aligned panel construction (spec section 9) -- the
single most important correctness requirement in this project.

For every Fed communication event, every explanatory variable must
reflect information actually publicly available STRICTLY BEFORE that
event, never the observation's reference period. This module is a thin,
config-driven orchestration layer over src.utils.dates.asof_merge_strict
(already unit-tested there): it repeatedly asof-merges each expectations/
macro series into the event panel using that series' ``availability_col``
(release_date, never reference_date/reference_month/reference_quarter).

WHY THIS AUTOMATICALLY IMPLEMENTS "PRIOR CLOSE" FOR DAILY MARKET SERIES
WITHOUT SPECIAL-CASE CODE: every event timestamp in this project (speech/
statement/minutes dates) and every daily market series' date are stored
as date-only values (midnight-anchored Timestamps; we do not have
intraday release times for Fed communications). ``asof_merge_strict``
with ``strict=True`` excludes exact-timestamp matches, so a market
observation dated the SAME calendar day as the event is never used --
the merge falls through to the most recent STRICTLY EARLIER date, which
is exactly "yesterday's close" for a same-day speech, satisfying spec
section 3A's requirement ("use the most recent market close BEFORE the
communication") without bespoke logic. This is deliberately the
conservative choice: it can occasionally under-use same-day information
released before an afternoon speech, but it can never look ahead. Every
alignment call in this module defaults to ``strict=True`` for exactly
this reason -- do not change it to ``strict=False`` without re-reading
this docstring.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.dates import asof_merge_strict
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SeriesAlignmentSpec:
    """One expectations/macro series ready to be asof-merged into the
    event panel.

    ``availability_col`` MUST be a release-date-like column (release_date,
    minutes_release_date, ...), never a reference-period column
    (reference_month, reference_quarter, meeting_date). Every downloader
    in src/data and every transform in src/features/expectations.py and
    src/features/macro_controls.py already produces exactly this
    distinction -- see their availability_col / release_date fields.
    """

    name: str
    df: pd.DataFrame
    availability_col: str
    value_cols: list[str]


def align_series_to_events(
    events: pd.DataFrame,
    event_time_col: str,
    series_specs: list[SeriesAlignmentSpec],
    strict: bool = True,
) -> pd.DataFrame:
    """Asof-merge every series in ``series_specs`` into ``events``.

    Returns a new DataFrame with ``events``' original columns plus, for
    each spec, columns named ``{spec.name}_{value_col}`` and
    ``{spec.name}_{availability_col}`` (so the actual release date used
    for each row is always inspectable/auditable, not hidden).
    """
    panel = events.copy()
    for spec in series_specs:
        if spec.df.empty:
            logger.warning("Series %s has an empty DataFrame; skipping (all merged columns will be NaN).", spec.name)
            for vc in spec.value_cols:
                panel[f"{spec.name}_{vc}"] = pd.NA
            continue
        panel = asof_merge_strict(
            panel,
            event_time_col=event_time_col,
            series=spec.df,
            availability_col=spec.availability_col,
            value_cols=spec.value_cols,
            prefix=spec.name,
            strict=strict,
        )
        release_col = f"{spec.name}_{spec.availability_col}"
        if release_col in panel.columns:
            days_stale = (pd.to_datetime(panel[event_time_col]) - pd.to_datetime(panel[release_col])).dt.days
            panel[f"{spec.name}_days_since_release"] = days_stale
    return panel


def coverage_report(panel: pd.DataFrame, series_specs: list[SeriesAlignmentSpec]) -> pd.DataFrame:
    """Diagnostic: for each series, what share of event rows got a
    non-null value, and the average/median staleness in days. Use this
    to catch a mis-specified availability_col (e.g. one that produces
    100% missingness because every release postdates every event) before
    it silently degrades a regression's sample size.
    """
    rows = []
    for spec in series_specs:
        value_col = f"{spec.name}_{spec.value_cols[0]}"
        stale_col = f"{spec.name}_days_since_release"
        if value_col not in panel.columns:
            continue
        non_null = panel[value_col].notna()
        rows.append(
            {
                "series": spec.name,
                "n_events": len(panel),
                "n_available": int(non_null.sum()),
                "coverage_pct": round(100 * non_null.mean(), 1),
                "median_days_stale": panel.loc[non_null, stale_col].median() if stale_col in panel.columns and non_null.any() else None,
                "max_days_stale": panel.loc[non_null, stale_col].max() if stale_col in panel.columns and non_null.any() else None,
            }
        )
    return pd.DataFrame(rows)
