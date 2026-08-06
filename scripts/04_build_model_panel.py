#!/usr/bin/env python
"""Stage 4: assemble the information-set-aligned model panel.

Combines the scored Fed-communication event panel (documents_scored.parquet)
with every inflation-expectations series (transformed: level, change,
z-score, deviation-from-target) and the macro control panel, using
strict asof-merges on RELEASE dates (never reference dates) so no
regressor in the final panel reflects information unavailable at the
time of the communication (spec section 9).

Also computes the speaker-relative stance features from spec section 8
(delta_stance vs. the same speaker's previous monetary-policy-relevant
communication; stance vs. that speaker's trailing rolling mean), since a
policymaker's own baseline hawkishness/dovishness is a confound for the
absolute stance level.

Usage: python scripts/04_build_model_panel.py
Writes: data/processed/model_panel.parquet, output/diagnostics/series_coverage.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.align_information_set import SeriesAlignmentSpec, align_series_to_events, coverage_report
from src.features.expectations import transform_series
from src.utils.config import get_source
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "diagnostics"

# FRED series without a native release-date column (fred.py returns the
# CURRENT revised vintage's realtime_start, not the original release
# date -- see module note below). We assign a release date from each
# series' catalog release_timing_method plus a documented, conservative
# per-family lag, exactly like every other "no official release calendar
# available" case in this project (nyfed_sce.py, atsix.py, ...).
FRED_RELEASE_RULES = {
    # same-day market series: release_date = observation date itself.
    "T5YIE": ("same_day", 0),
    "T10YIE": ("same_day", 0),
    "T5YIFR": ("same_day", 0),
    # Cleveland Fed model estimate: conservatively assume not available
    # until mid-month of the SAME reference month (errs toward a later,
    # safer release date rather than an earlier one).
    "EXPINF1YR": ("mid_month_same", 15),
    "EXPINF5YR": ("mid_month_same", 15),
    "EXPINF10YR": ("mid_month_same", 15),
    "EXPINF30YR": ("mid_month_same", 15),
    # Michigan preliminary estimate: released mid-month for the same
    # reference month.
    "MICH": ("mid_month_same", 14),
}


def _fred_release_date(series_id: str, reference_dates: pd.Series) -> pd.Series:
    rule, day = FRED_RELEASE_RULES[series_id]
    if rule == "same_day":
        return reference_dates
    if rule == "mid_month_same":
        return reference_dates.apply(lambda d: pd.Timestamp(year=d.year, month=d.month, day=day))
    raise ValueError(f"Unknown release rule {rule!r} for {series_id}")


def _build_fred_specs() -> list[SeriesAlignmentSpec]:
    path = PROCESSED_DIR / "fred_series_long.parquet"
    long_df = pd.read_parquet(path)
    specs = []
    for series_id in FRED_RELEASE_RULES:
        sub = long_df[long_df["series_id"] == series_id][["date", "value"]].dropna()
        if sub.empty:
            logger.warning("No data for FRED series %s; skipping.", series_id)
            continue
        frequency = "daily" if series_id.startswith("T5") or series_id.startswith("T10") else "monthly"
        src = get_source(series_id)
        transformed = transform_series(sub, "date", "value", frequency, target_anchor=src.target_anchor)
        transformed["release_date"] = _fred_release_date(series_id, transformed["date"])
        specs.append(
            SeriesAlignmentSpec(
                name=series_id,
                df=transformed,
                availability_col="release_date",
                value_cols=[c for c in transformed.columns if c.startswith("value_")],
            )
        )
    return specs


def _build_nyfed_sce_specs() -> list[SeriesAlignmentSpec]:
    df = pd.read_parquet(PROCESSED_DIR / "nyfed_sce.parquet")
    specs = []
    for horizon_col in ["median_1yr", "median_3yr", "median_5yr"]:
        sub = df[["reference_month", "release_date", horizon_col]].rename(columns={horizon_col: "value"}).dropna(subset=["value"])
        if sub.empty:
            continue
        transformed = transform_series(sub, "reference_month", "value", "monthly", target_anchor=None)
        transformed["release_date"] = sub.sort_values("reference_month")["release_date"].values[: len(transformed)]
        name = f"SCE_{horizon_col}"
        specs.append(
            SeriesAlignmentSpec(
                name=name,
                df=transformed,
                availability_col="release_date",
                value_cols=[c for c in transformed.columns if c.startswith("value_")],
            )
        )
    return specs


def _build_spf_specs() -> list[SeriesAlignmentSpec]:
    df = pd.read_parquet(PROCESSED_DIR / "philly_spf.parquet")
    specs = []
    for series_id in df["series_id"].unique():
        sub = df[df["series_id"] == series_id][["reference_quarter", "release_date", "value"]].dropna(subset=["value"])
        if sub.empty:
            continue
        try:
            src = get_source(series_id)
            anchor = src.target_anchor
        except KeyError:
            anchor = None
        transformed = transform_series(sub, "reference_quarter", "value", "quarterly", target_anchor=anchor)
        transformed["release_date"] = sub.sort_values("reference_quarter")["release_date"].values[: len(transformed)]
        specs.append(
            SeriesAlignmentSpec(
                name=f"SPF_{series_id}",
                df=transformed,
                availability_col="release_date",
                value_cols=[c for c in transformed.columns if c.startswith("value_")],
            )
        )
    return specs


def _build_livingston_spec() -> list[SeriesAlignmentSpec]:
    df = pd.read_parquet(PROCESSED_DIR / "livingston.parquet")
    if df.empty:
        return []
    sub = df[["reference_date", "release_date", "value"]].dropna(subset=["value"])
    transformed = transform_series(sub, "reference_date", "value", "semiannual", target_anchor=2.0)
    transformed["release_date"] = sub.sort_values("reference_date")["release_date"].values[: len(transformed)]
    return [
        SeriesAlignmentSpec(
            name="LIVINGSTON_CPI10Y",
            df=transformed,
            availability_col="release_date",
            value_cols=[c for c in transformed.columns if c.startswith("value_")],
        )
    ]


def _build_atsix_specs() -> list[SeriesAlignmentSpec]:
    path = PROCESSED_DIR / "atsix_selected_horizons.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    specs = []
    for col in ["atsix_12m", "atsix_24m", "atsix_36m", "atsix_60m", "atsix_120m",
                "atsix_slope_10y_1y", "atsix_slope_5y_1y", "atsix_slope_10y_5y"]:
        if col not in df.columns:
            continue
        sub = df[["reference_month", "release_date", col]].rename(columns={col: "value"}).dropna(subset=["value"])
        if sub.empty:
            continue
        anchor = 2.0 if "slope" not in col else None
        transformed = transform_series(sub, "reference_month", "value", "monthly", target_anchor=anchor)
        transformed["release_date"] = sub.sort_values("reference_month")["release_date"].values[: len(transformed)]
        specs.append(
            SeriesAlignmentSpec(
                name=f"ATSIX_{col.replace('atsix_', '')}",
                df=transformed,
                availability_col="release_date",
                value_cols=[c for c in transformed.columns if c.startswith("value_")],
            )
        )
    return specs


def _build_atlanta_bie_spec() -> list[SeriesAlignmentSpec]:
    path = PROCESSED_DIR / "atlanta_bie.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    if df.empty:
        return []
    sub = df[["survey_period_end", "release_date", "value"]].dropna(subset=["value"])
    transformed = transform_series(sub, "survey_period_end", "value", "monthly", target_anchor=None)
    transformed["release_date"] = sub.sort_values("survey_period_end")["release_date"].values[: len(transformed)]
    return [
        SeriesAlignmentSpec(
            name="ATL_BIE",
            df=transformed,
            availability_col="release_date",
            value_cols=[c for c in transformed.columns if c.startswith("value_")],
        )
    ]


def _build_macro_specs() -> list[SeriesAlignmentSpec]:
    path = PROCESSED_DIR / "macro_controls.parquet"
    long_df = pd.read_parquet(path)
    specs = []
    for series_id in long_df["series_id"].unique():
        sub = long_df[long_df["series_id"] == series_id][["reference_date", "release_date", "value"]].dropna(subset=["value"])
        if sub.empty:
            continue
        sub = sub.sort_values("reference_date").rename(columns={"reference_date": "date"})
        specs.append(
            SeriesAlignmentSpec(
                name=f"MACRO_{series_id}",
                df=sub,
                availability_col="release_date",
                value_cols=["value"],
            )
        )
    return specs


def _add_speaker_relative_features(panel: pd.DataFrame) -> pd.DataFrame:
    """spec section 8: delta_stance vs. same speaker's previous MP-relevant
    communication, plus stance vs. that speaker's trailing rolling mean.
    FOMC-collective documents (speaker is null) are compared against the
    previous document of the SAME document_type instead.
    """
    panel = panel.sort_values("event_date").reset_index(drop=True)
    panel["stance_group"] = panel["speaker"].fillna(panel["document_type"])

    delta_stance = []
    speaker_rolling_mean = []
    stance_vs_speaker_rolling_mean = []
    group_history: dict[str, list[float]] = {}

    for _, row in panel.iterrows():
        group = row["stance_group"]
        history = group_history.setdefault(group, [])
        if history:
            delta_stance.append(row["stance_score"] - history[-1])
            rolling_mean = sum(history) / len(history)
            speaker_rolling_mean.append(rolling_mean)
            stance_vs_speaker_rolling_mean.append(row["stance_score"] - rolling_mean)
        else:
            delta_stance.append(pd.NA)
            speaker_rolling_mean.append(pd.NA)
            stance_vs_speaker_rolling_mean.append(pd.NA)
        history.append(row["stance_score"])

    panel["delta_stance"] = delta_stance
    panel["speaker_rolling_mean_stance"] = speaker_rolling_mean
    panel["stance_vs_speaker_rolling_mean"] = stance_vs_speaker_rolling_mean
    return panel


def main() -> int:
    events = pd.read_parquet(PROCESSED_DIR / "documents_scored.parquet")
    events = events.sort_values("event_date").reset_index(drop=True)
    logger.info("Loaded %d scored documents as the base event panel", len(events))

    all_specs: list[SeriesAlignmentSpec] = []
    all_specs += _build_fred_specs()
    all_specs += _build_nyfed_sce_specs()
    all_specs += _build_spf_specs()
    all_specs += _build_livingston_spec()
    all_specs += _build_atsix_specs()
    all_specs += _build_atlanta_bie_spec()
    all_specs += _build_macro_specs()
    logger.info("Built %d series alignment specs", len(all_specs))

    panel = align_series_to_events(events, "event_date", all_specs, strict=True)
    panel = _add_speaker_relative_features(panel)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PROCESSED_DIR / "model_panel.parquet", index=False)
    logger.info("Wrote model_panel.parquet: %d rows x %d columns", *panel.shape)

    report = coverage_report(panel, all_specs)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(DIAGNOSTICS_DIR / "series_coverage.csv", index=False)
    logger.info("Wrote series coverage report (%d series) -> %s", len(report), DIAGNOSTICS_DIR / "series_coverage.csv")
    low_coverage = report[report["coverage_pct"] < 20]
    if not low_coverage.empty:
        logger.warning("Series with <20%% event coverage (check availability_col / sample window):\n%s", low_coverage.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
