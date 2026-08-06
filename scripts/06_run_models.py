#!/usr/bin/env python
"""Stage 6: the core econometric horse race (spec sections 13-19, 22).

Produces:
  TABLE 4 -- Univariate Reaction Models
  TABLE 5 -- Multivariate Reaction Models (OLS / Ridge / Elastic Net)
  TABLE 6 -- Source-Family Models
  TABLE 7 -- Out-of-Sample Predictive Importance (summary matrix)
  FIGURE 5 -- Standardized Regression Coefficients
  FIGURE 6 -- Incremental Out-of-Sample R^2 by Measure
  FIGURE 7 -- Release Event Responses
  output/regressions/hierarchical_speaker_model.csv
  output/regressions/local_projections_<measure>.csv

Usage: python scripts/06_run_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baseline import build_regression_sample, run_univariate_horse_race
from src.models.event_study import release_event_study_all_measures
from src.models.hierarchical import fit_hierarchical_speaker_model
from src.models.importance import build_summary_matrix, compute_permutation_importance
from src.models.local_projections import local_projection_coefficients
from src.models.registry import MACRO_CONTROL_COLUMNS, PRIMARY_MEASURES
from src.models.regularized import leave_one_out_incremental_r2, pca_factor_scores, run_multivariate_models
from src.utils.logging import get_logger
from src.visualization.style import CATEGORICAL_PALETTE, GRID_COLOR, NEUTRAL_MARK, apply_paper_style

logger = get_logger(__name__)

# Maps each registry measure_id to the (possibly shared, possibly none)
# text-mention label it corresponds to in expectation_mentions.parquet's
# "specific_measure" column (src.text.expectation_mentions.MEASURE_DEFS).
# A "TIPS / Breakevens" mention in a speech doesn't distinguish which of
# T5YIE/T10YIE/T5YIFR the speaker meant, so those three legitimately
# share one mention count; ATSIX/Cleveland-Fed-model measures have no
# corresponding text pattern (nobody says "ATSIX" out loud) and are left
# unmapped -- their mention_count is honestly 0, not a missing value.
MEASURE_ID_TO_MENTION_LABELS = {
    "T5YIE": ["TIPS / Breakevens"], "T10YIE": ["TIPS / Breakevens"],
    "T5YIFR": ["TIPS / Breakevens", "5-Year, 5-Year Forward"],
    "MICH": ["Michigan Survey"],
    "SCE_1yr": ["NY Fed SCE"], "SCE_3yr": ["NY Fed SCE"], "SCE_5yr": ["NY Fed SCE"],
    "SPF_CPI1YR": ["SPF"], "SPF_CPI5YR": ["SPF"], "SPF_CPI10": ["SPF"],
    "SPF_PCE1YR": ["SPF"], "SPF_PCE5YR": ["SPF"], "SPF_PCE10": ["SPF"],
    "LIVINGSTON": ["Livingston Survey"],
    "ATL_BIE": ["Business (generic)"],
}


def _mention_counts_by_measure_id(mentions_df: pd.DataFrame) -> pd.Series:
    raw_counts = mentions_df["specific_measure"].value_counts()
    out = {}
    for measure_id, labels in MEASURE_ID_TO_MENTION_LABELS.items():
        out[measure_id] = sum(raw_counts.get(label, 0) for label in labels)
    return pd.Series(out)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"
REGRESSIONS_DIR = PROJECT_ROOT / "output" / "regressions"

DV_COL = "delta_stance"


def _well_covered_measures(panel: pd.DataFrame, min_coverage: float = 0.30) -> list:
    """Restrict the multivariate/family analysis to measures with enough
    non-missing coverage in the regression sample to be usable together
    (many measures individually have >50% coverage, per
    output/diagnostics/series_coverage.csv, but jointly requiring all 21
    non-missing at once would gut the sample -- see Table 5's notes).
    """
    sample = build_regression_sample(panel, dv_col=DV_COL)
    keep = []
    for m in PRIMARY_MEASURES:
        if m.level_zscore_col in sample.columns:
            coverage = sample[m.level_zscore_col].notna().mean()
            if coverage >= min_coverage:
                keep.append(m)
    return keep


def table4_univariate(panel: pd.DataFrame) -> pd.DataFrame:
    out = run_univariate_horse_race(panel, dv_col=DV_COL, control_cols=MACRO_CONTROL_COLUMNS)
    out.to_csv(TABLES_DIR / "table4_univariate_models.csv", index=False)
    logger.info("Table 4 (univariate): %d measures", len(out))
    return out


def table5_multivariate(panel: pd.DataFrame, measures: list) -> tuple[pd.DataFrame, dict]:
    sample = build_regression_sample(panel, dv_col=DV_COL, extra_required_cols=MACRO_CONTROL_COLUMNS)
    reg_cols = [m.level_zscore_col for m in measures]
    sub = sample.dropna(subset=reg_cols + MACRO_CONTROL_COLUMNS)
    logger.info("Table 5 (multivariate): %d measures, %d obs after requiring all present", len(reg_cols), len(sub))

    results = run_multivariate_models(sub, DV_COL, reg_cols + MACRO_CONTROL_COLUMNS, n_splits=5)
    rows = []
    for method, r in results.items():
        for term, coef in r.full_sample_coef.items():
            rows.append({"method": method, "term": term, "coef": coef, "mean_oos_r2": r.mean_oos_r2, "n_obs": r.n_obs})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "table5_multivariate_models.csv", index=False)
    return out, results


def table6_source_family(panel: pd.DataFrame, measures: list) -> pd.DataFrame:
    sample = build_regression_sample(panel, dv_col=DV_COL, extra_required_cols=MACRO_CONTROL_COLUMNS)
    families = sorted(set(m.source_family for m in measures))
    rows = []
    for family in families:
        family_measures = [m for m in measures if m.source_family == family]
        cols = [m.level_zscore_col for m in family_measures]
        sub = sample.dropna(subset=cols + MACRO_CONTROL_COLUMNS)
        if len(sub) < 50 or len(cols) < 1:
            logger.warning("Skipping family %s: insufficient data (%d obs, %d measures)", family, len(sub), len(cols))
            continue
        if len(cols) == 1:
            factor = sub[cols[0]]
            explained_var = 1.0
        else:
            factor_scores, evr = pca_factor_scores(sub[cols], n_components=1)
            factor = factor_scores["factor_1"]
            explained_var = float(evr[0])
        sub = sub.loc[factor.index].copy()
        sub["family_factor"] = factor.values
        from src.models.baseline import fit_baseline_model

        try:
            result = fit_baseline_model(sub, DV_COL, ["family_factor"], MACRO_CONTROL_COLUMNS, compute_oos=True)
        except Exception as exc:
            logger.error("Family model failed for %s: %s", family, exc)
            continue
        coef = result.coef_row("family_factor")
        rows.append(
            {
                "source_family": family,
                "n_measures_in_family": len(cols),
                "measures": ", ".join(m.measure_id for m in family_measures),
                "pca_explained_variance": explained_var,
                "coef": coef["coef"],
                "se": coef["se"],
                "t": coef["t"],
                "pvalue": coef["pvalue"],
                "adj_r2": result.adj_r2,
                "oos_r2": result.oos_r2,
                "n_obs": result.n_obs,
            }
        )
    out = pd.DataFrame(rows).sort_values("t", key=lambda s: s.abs(), ascending=False)
    out.to_csv(TABLES_DIR / "table6_source_family_models.csv", index=False)
    logger.info("Table 6 (source family): %d families", len(out))

    grouping = {m.level_zscore_col: m.source_family for m in measures}
    reg_cols = [m.level_zscore_col for m in measures]
    sub_all = sample.dropna(subset=reg_cols)
    if len(sub_all) >= 50:
        loo_family = leave_one_out_incremental_r2(sub_all, DV_COL, reg_cols, grouping=grouping, n_splits=5)
        loo_family.to_csv(TABLES_DIR / "table6b_family_incremental_r2.csv", index=False)
    return out


def table7_importance(panel: pd.DataFrame, univariate_df: pd.DataFrame, measures: list) -> pd.DataFrame:
    sample = build_regression_sample(panel, dv_col=DV_COL, extra_required_cols=MACRO_CONTROL_COLUMNS)
    reg_cols = [m.level_zscore_col for m in measures]
    sub = sample.dropna(subset=reg_cols)
    logger.info("Table 7 (importance): %d measures, %d obs", len(reg_cols), len(sub))

    # grouping maps each zscore column directly to its measure_id (not to
    # itself) so leave_one_out_incremental_r2's "removed" column already
    # contains values that match univariate_df["measure_id"] -- build_summary_matrix
    # renames "removed" -> "measure_id" and merges on it directly.
    grouping = {m.level_zscore_col: m.measure_id for m in measures}
    loo = leave_one_out_incremental_r2(sub, DV_COL, reg_cols, grouping=grouping, n_splits=5)

    perm = compute_permutation_importance(sub, DV_COL, reg_cols, n_repeats=15)

    mentions_path = PROCESSED_DIR / "expectation_mentions.parquet"
    mention_counts = None
    if mentions_path.exists():
        mdf = pd.read_parquet(mentions_path)
        mention_counts = _mention_counts_by_measure_id(mdf)

    event_path = TABLES_DIR / "_event_study_raw.csv"
    event_df = pd.read_csv(event_path) if event_path.exists() else None

    summary = build_summary_matrix(
        univariate_df, incremental_r2_df=loo, permutation_df=perm,
        mention_counts=mention_counts, event_study_df=event_df,
    )
    summary.to_csv(TABLES_DIR / "table7_importance_ranking.csv", index=False)
    logger.info("Table 7 (importance ranking) top 5:\n%s", summary.head(5)[["measure_id", "display_name", "convergence_rank_score"]].to_string())
    return summary


def figure5_coefficients(univariate_df: pd.DataFrame) -> None:
    apply_paper_style()
    df = univariate_df.sort_values("coef")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(df))))
    colors = [CATEGORICAL_PALETTE[0] if p < 0.05 else NEUTRAL_MARK for p in df["pvalue"]]
    ax.barh(range(len(df)), df["coef"], xerr=1.96 * df["se"], color=colors, height=0.6,
            ecolor=CATEGORICAL_PALETTE[1])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["display_name"], fontsize=7)
    ax.axvline(0, color=GRID_COLOR, linewidth=1.0)
    ax.set_xlabel("Coefficient (change in stance per 1 s.d. move), 95% CI; dark = p<0.05")
    ax.set_title("Univariate Standardized Regression Coefficients")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure5_standardized_coefficients.png")
    plt.close(fig)


def figure6_incremental_oos_r2(table7: pd.DataFrame) -> None:
    apply_paper_style()
    df = table7.dropna(subset=["incremental_oos_r2"]).sort_values("incremental_oos_r2")
    if df.empty:
        logger.warning("No incremental_oos_r2 values available; skipping Figure 6.")
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(df))))
    ax.barh(range(len(df)), df["incremental_oos_r2"], color=CATEGORICAL_PALETTE[0], height=0.6)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["display_name"], fontsize=7)
    ax.axvline(0, color=GRID_COLOR, linewidth=1.0)
    ax.set_xlabel("Incremental out-of-sample R^2 (full Ridge model minus model without this measure)")
    ax.set_title("Incremental Out-of-Sample Predictive Contribution by Measure")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure6_incremental_oos_r2.png")
    plt.close(fig)


def figure7_event_responses(event_df: pd.DataFrame) -> None:
    if event_df is None or event_df.empty:
        logger.warning("No event-study results; skipping Figure 7.")
        return
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, measure_id in enumerate(event_df["measure_id"].unique()):
        sub = event_df[event_df["measure_id"] == measure_id].sort_values("window_days")
        color = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
        ax.errorbar(sub["window_days"], sub["coef"], yerr=1.96 * sub["se"], label=measure_id,
                    color=color, marker="o", markersize=4, linewidth=1.2, capsize=3)
    ax.axhline(0, color=GRID_COLOR, linewidth=1.0)
    ax.set_xlabel("Window (business days after release)")
    ax.set_ylabel("Coefficient: stance ~ release surprise")
    ax.set_title("Release-Window Stance Association by Measure")
    ax.legend(loc="best", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure7_release_event_responses.png")
    plt.close(fig)


def run_hierarchical_and_local_projections(panel: pd.DataFrame, measures: list) -> None:
    sample = build_regression_sample(panel, dv_col=DV_COL, extra_required_cols=MACRO_CONTROL_COLUMNS)
    top_measure = measures[0]
    try:
        h_result = fit_hierarchical_speaker_model(
            sample, DV_COL, top_measure.level_zscore_col, control_cols=MACRO_CONTROL_COLUMNS[:3], min_speaker_docs=10
        )
        h_result.speaker_effects.to_csv(REGRESSIONS_DIR / "hierarchical_speaker_model.csv", index=False)
        logger.info(
            "Hierarchical model (%s): committee avg coef=%.4f, %d speakers, converged=%s",
            top_measure.measure_id, h_result.committee_avg_coef, h_result.n_groups, h_result.converged,
        )
    except Exception as exc:
        logger.error("Hierarchical model failed: %s", exc)

    for m in measures[:5]:
        try:
            lp = local_projection_coefficients(panel, m.change_zscore_col, control_cols=MACRO_CONTROL_COLUMNS[:3])
            if not lp.empty:
                lp.to_csv(REGRESSIONS_DIR / f"local_projections_{m.measure_id}.csv", index=False)
        except Exception as exc:
            logger.error("Local projection failed for %s: %s", m.measure_id, exc)


def run_event_studies(panel: pd.DataFrame, measures: list) -> pd.DataFrame:
    # The panel's column prefix (from SeriesAlignmentSpec.name in
    # scripts/04_build_model_panel.py) doesn't always equal the
    # registry's shorter measure_id (e.g. "SCE_median_1yr" vs "SCE_1yr",
    # "LIVINGSTON_CPI10Y" vs "LIVINGSTON") -- derive it from
    # level_zscore_col instead of assuming a match, then relabel results
    # back to the registry's measure_id for consistency with every other
    # table in this script.
    panel_prefix_to_measure_id = {m.level_zscore_col.replace("_value_zscore", ""): m.measure_id for m in measures}
    out = release_event_study_all_measures(panel, list(panel_prefix_to_measure_id.keys()), control_cols=MACRO_CONTROL_COLUMNS[:3])
    if not out.empty:
        out["measure_id"] = out["measure_id"].map(panel_prefix_to_measure_id)
        out.to_csv(TABLES_DIR / "_event_study_raw.csv", index=False)
    return out


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REGRESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PROCESSED_DIR / "model_panel.parquet")
    logger.info("Loaded model panel: %d rows x %d cols", *panel.shape)

    measures = _well_covered_measures(panel)
    logger.info("%d/%d measures have >=30%% coverage in the regression sample: %s", len(measures), len(PRIMARY_MEASURES), [m.measure_id for m in measures])

    univariate_df = table4_univariate(panel)
    figure5_coefficients(univariate_df)

    _, mv_results = table5_multivariate(panel, measures)

    table6_source_family(panel, measures)

    event_df = run_event_studies(panel, measures)
    figure7_event_responses(event_df)

    table7 = table7_importance(panel, univariate_df, measures)
    figure6_incremental_oos_r2(table7)

    run_hierarchical_and_local_projections(panel, measures)

    logger.info("Stage 6 completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
