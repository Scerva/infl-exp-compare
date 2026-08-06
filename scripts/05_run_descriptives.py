#!/usr/bin/env python
"""Stage 5: descriptive tables and figures (spec section 12).

Produces:
  TABLE 1 -- Inflation Expectations Data Sources
  TABLE 2 -- Fed Communication Corpus Summary
  TABLE 3 -- Correlations Across Inflation Expectations
  FIGURE 1 -- Major Inflation Expectations Measures
  FIGURE 2 -- Standardized Expectations by Source Family
  FIGURE 3 -- Fed Hawkish/Dovish Stance Through Time
  FIGURE 4 -- Expectation Measure Mention Frequency
  FIGURE 8 -- Consumer vs Market vs Professional Divergence
  FIGURE 9 -- Speaker Heterogeneity

Usage: python scripts/05_run_descriptives.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.expectations import transform_series
from src.utils.config import load_source_catalog
from src.utils.logging import get_logger
from src.visualization.expectations import plot_correlation_heatmap, plot_series_by_family
from src.visualization.stance import (
    plot_family_divergence,
    plot_mention_frequency,
    plot_speaker_stance,
    plot_stance_over_time,
)

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"


def table1_source_catalog() -> pd.DataFrame:
    catalog = load_source_catalog()
    rows = [
        {
            "series_id": s.series_id,
            "series_name": s.series_name,
            "source_family": s.source_family,
            "provider": s.provider,
            "frequency": s.frequency,
            "horizon_months": s.horizon_months,
            "inflation_concept": s.inflation_concept,
            "availability_start": s.availability_start,
            "target_anchor": s.target_anchor,
        }
        for s in catalog
    ]
    return pd.DataFrame(rows)


def table2_corpus_summary(documents_df: pd.DataFrame) -> pd.DataFrame:
    df = documents_df.copy()
    df["year"] = pd.to_datetime(df["event_date"]).dt.year
    by_type_year = df.groupby(["document_type", "year"]).size().unstack(fill_value=0)
    by_type = df.groupby("document_type").agg(
        n_documents=("document_id", "count"),
        n_relevant=("monetary_policy_relevance", lambda s: (s >= 0.10).sum()),
        mean_relevance=("monetary_policy_relevance", "mean"),
        mean_stance=("stance_score", "mean"),
    )
    by_type_year.to_csv(TABLES_DIR / "table2b_corpus_by_type_year.csv")
    return by_type


def _load_expectation_series_for_correlation() -> dict[str, pd.DataFrame]:
    """Monthly-resampled level series for the primary cross-source
    comparison set (one representative long-run measure per family plus
    the household/professional/market family members), keyed by display
    name -> DataFrame with columns [date, value].
    """
    out = {}

    fred = pd.read_parquet(PROCESSED_DIR / "fred_series_long.parquet")
    for sid, label in [("T5YIE", "5y Breakeven (mkt)"), ("T10YIE", "10y Breakeven (mkt)"),
                        ("T5YIFR", "5y5y Fwd (mkt)"), ("EXPINF10YR", "Cleveland Fed 10y (model)"),
                        ("MICH", "Michigan 1y (household)")]:
        sub = fred[fred["series_id"] == sid][["date", "value"]].dropna()
        out[label] = sub

    sce = pd.read_parquet(PROCESSED_DIR / "nyfed_sce.parquet")
    out["NY Fed SCE 3y (household)"] = sce[["reference_month", "median_3yr"]].rename(
        columns={"reference_month": "date", "median_3yr": "value"}
    ).dropna()

    spf = pd.read_parquet(PROCESSED_DIR / "philly_spf.parquet")
    for sid, label in [("CPI10", "SPF CPI 10y (professional)"), ("PCE10", "SPF PCE 10y (professional)")]:
        sub = spf[spf["series_id"] == sid][["reference_quarter", "value"]].rename(
            columns={"reference_quarter": "date"}
        ).dropna()
        out[label] = sub

    livingston = pd.read_parquet(PROCESSED_DIR / "livingston.parquet")
    out["Livingston CPI 10y (professional)"] = livingston[["reference_date", "value"]].rename(
        columns={"reference_date": "date"}
    ).dropna()

    atsix = pd.read_parquet(PROCESSED_DIR / "atsix_selected_horizons.parquet")
    out["ATSIX 10y (model)"] = atsix[["reference_month", "atsix_120m"]].rename(
        columns={"reference_month": "date", "atsix_120m": "value"}
    ).dropna()

    return out


def table3_correlations(series_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    monthly = {}
    for label, df in series_dict.items():
        s = df.set_index("date")["value"].resample("MS").mean()
        monthly[label] = s
    wide = pd.DataFrame(monthly)
    return wide.corr()


def figure1_expectations_by_family(series_dict: dict[str, pd.DataFrame]) -> None:
    families = {
        "5y Breakeven (mkt)": "market", "10y Breakeven (mkt)": "market", "5y5y Fwd (mkt)": "market",
        "Cleveland Fed 10y (model)": "model", "ATSIX 10y (model)": "model",
        "Michigan 1y (household)": "household", "NY Fed SCE 3y (household)": "household",
        "SPF CPI 10y (professional)": "professional", "SPF PCE 10y (professional)": "professional",
        "Livingston CPI 10y (professional)": "professional",
    }
    series_for_plot = {
        label: (df, "date", "value", families.get(label, "other")) for label, df in series_dict.items()
    }
    plot_series_by_family(
        series_for_plot,
        title="Major Inflation Expectations Measures (levels, percent)",
        ylabel="Percent",
        out_path=FIGURES_DIR / "figure1_expectations_levels.png",
    )


def figure2_standardized_expectations(series_dict: dict[str, pd.DataFrame]) -> None:
    standardized = {}
    for label, df in series_dict.items():
        freq = "daily" if "mkt" in label else ("quarterly" if ("SPF" in label or "Livingston" in label) else "monthly")
        t = transform_series(df, "date", "value", freq)
        standardized[label] = t[["date", f"value_zscore"]].rename(columns={"value_zscore": "value"})
    families = {
        "5y Breakeven (mkt)": "market", "10y Breakeven (mkt)": "market", "5y5y Fwd (mkt)": "market",
        "Cleveland Fed 10y (model)": "model", "ATSIX 10y (model)": "model",
        "Michigan 1y (household)": "household", "NY Fed SCE 3y (household)": "household",
        "SPF CPI 10y (professional)": "professional", "SPF PCE 10y (professional)": "professional",
        "Livingston CPI 10y (professional)": "professional",
    }
    series_for_plot = {
        label: (df, "date", "value", families.get(label, "other")) for label, df in standardized.items()
    }
    plot_series_by_family(
        series_for_plot,
        title="Standardized Inflation Expectations by Source Family (rolling z-score)",
        ylabel="Z-score",
        out_path=FIGURES_DIR / "figure2_expectations_standardized.png",
        standardized=True,
    )


def figure8_family_divergence(series_dict: dict[str, pd.DataFrame]) -> None:
    monthly_z = {}
    family_map = {
        "5y Breakeven (mkt)": "market", "10y Breakeven (mkt)": "market", "5y5y Fwd (mkt)": "market",
        "Michigan 1y (household)": "household", "NY Fed SCE 3y (household)": "household",
        "SPF CPI 10y (professional)": "professional", "SPF PCE 10y (professional)": "professional",
        "Livingston CPI 10y (professional)": "professional",
    }
    for label, df in series_dict.items():
        if label not in family_map:
            continue
        freq = "daily" if "mkt" in label else ("quarterly" if ("SPF" in label or "Livingston" in label) else "monthly")
        t = transform_series(df, "date", "value", freq)
        s = t.set_index("date")["value_zscore"].resample("MS").mean()
        monthly_z.setdefault(family_map[label], []).append(s)

    family_avg = {fam: pd.concat(series_list, axis=1, sort=True).mean(axis=1) for fam, series_list in monthly_z.items()}
    panel = pd.DataFrame(family_avg).dropna(how="all")
    panel["household_minus_market"] = panel.get("household") - panel.get("market")
    panel["professional_minus_market"] = panel.get("professional") - panel.get("market")
    panel = panel.reset_index().rename(columns={"index": "date"})
    plot_family_divergence(panel, "date", FIGURES_DIR / "figure8_family_divergence.png")


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    documents_df = pd.read_parquet(PROCESSED_DIR / "documents_scored.parquet")
    mentions_df = pd.read_parquet(PROCESSED_DIR / "expectation_mentions.parquet")

    t1 = table1_source_catalog()
    t1.to_csv(TABLES_DIR / "table1_source_catalog.csv", index=False)
    logger.info("Table 1 (source catalog): %d rows -> %s", len(t1), TABLES_DIR / "table1_source_catalog.csv")

    t2 = table2_corpus_summary(documents_df)
    t2.to_csv(TABLES_DIR / "table2_corpus_summary.csv")
    logger.info("Table 2 (corpus summary):\n%s", t2.to_string())

    series_dict = _load_expectation_series_for_correlation()
    t3 = table3_correlations(series_dict)
    t3.to_csv(TABLES_DIR / "table3_expectations_correlations.csv")
    logger.info("Table 3 (correlations): %dx%d matrix -> %s", *t3.shape, TABLES_DIR / "table3_expectations_correlations.csv")

    figure1_expectations_by_family(series_dict)
    logger.info("Figure 1 written.")
    figure2_standardized_expectations(series_dict)
    logger.info("Figure 2 written.")
    plot_correlation_heatmap(t3, "Correlation Matrix: Inflation Expectations Measures", FIGURES_DIR / "table3_correlation_heatmap.png")

    plot_stance_over_time(documents_df, FIGURES_DIR / "figure3_stance_over_time.png")
    logger.info("Figure 3 written.")

    plot_mention_frequency(mentions_df, FIGURES_DIR / "figure4_mention_frequency.png")
    logger.info("Figure 4 written.")

    figure8_family_divergence(series_dict)
    logger.info("Figure 8 written.")

    plot_speaker_stance(documents_df, FIGURES_DIR / "figure9_speaker_heterogeneity.png")
    logger.info("Figure 9 written.")

    logger.info("Stage 5 completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
