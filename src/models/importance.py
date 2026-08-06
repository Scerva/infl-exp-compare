"""Importance / ranking methodology (spec section 22).

Deliberately does NOT collapse everything into one arbitrary weighted
"overall score." Instead builds a summary MATRIX -- rows are candidate
expectations measures, columns are the different empirical tests this
project runs -- so a reader can see where the evidence converges (a
measure that ranks highly on coefficient significance, incremental OOS
R^2, permutation importance, AND mention frequency is a much stronger
claim than a measure that only wins on one axis). Section 22 is explicit
that convergence across tests, not a single index, is the answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from src.models.registry import PRIMARY_MEASURES
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_permutation_importance(
    panel_sample: pd.DataFrame,
    dv_col: str,
    regressor_cols: list[str],
    n_repeats: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """In-sample permutation importance from a Ridge fit across ALL
    measures simultaneously (so importance reflects each measure's
    marginal contribution net of its correlation with the others, unlike
    the univariate horse race). Documented as in-sample: an out-of-sample
    version would require refitting per fold, which
    src.models.regularized.leave_one_out_incremental_r2 already covers
    from a different angle (OOS R^2 delta rather than permutation score).
    """
    df = panel_sample.dropna(subset=[dv_col] + regressor_cols)
    X = df[regressor_cols].values
    y = df[dv_col].values
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)

    model = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(X_s, y)
    result = permutation_importance(model, X_s, y, n_repeats=n_repeats, random_state=random_state, scoring="r2")

    return pd.DataFrame(
        {
            "regressor": regressor_cols,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False)


def build_summary_matrix(
    univariate_df: pd.DataFrame,
    incremental_r2_df: pd.DataFrame | None = None,
    permutation_df: pd.DataFrame | None = None,
    mention_counts: pd.Series | None = None,
    event_study_df: pd.DataFrame | None = None,
    robustness_pass_rates: pd.Series | None = None,
) -> pd.DataFrame:
    """Assemble the section-22 summary matrix: one row per measure, one
    column per empirical test. Any input left as None simply produces
    NaN in that column rather than failing -- callers can build this up
    incrementally as later stages (event studies, robustness) complete.
    """
    measures_meta = {m.measure_id: m for m in PRIMARY_MEASURES}

    base = univariate_df[["measure_id", "display_name", "source_family", "horizon_bucket", "n_obs", "coef", "t", "pvalue", "oos_r2"]].copy()
    base = base.rename(columns={"coef": "univariate_beta", "t": "univariate_t", "pvalue": "univariate_pvalue", "n_obs": "N"})

    if incremental_r2_df is not None and not incremental_r2_df.empty:
        inc = incremental_r2_df.rename(columns={"removed": "measure_id"})[["measure_id", "incremental_oos_r2"]]
        base = base.merge(inc, on="measure_id", how="left")
    else:
        base["incremental_oos_r2"] = np.nan

    if permutation_df is not None and not permutation_df.empty:
        # permutation_df is indexed by regressor column name (e.g. "SPF_CPI5YR_value_zscore");
        # map back to measure_id via the registry's level_zscore_col.
        col_to_measure = {m.level_zscore_col: m.measure_id for m in PRIMARY_MEASURES}
        perm = permutation_df.copy()
        perm["measure_id"] = perm["regressor"].map(col_to_measure)
        perm = perm.dropna(subset=["measure_id"])[["measure_id", "permutation_importance_mean"]]
        base = base.merge(perm, on="measure_id", how="left")
    else:
        base["permutation_importance_mean"] = np.nan

    if mention_counts is not None:
        # mention_counts must be indexed by measure_id (the caller's
        # responsibility to aggregate raw text-mention labels into
        # measure_id -- see scripts/06_run_models.py's
        # MEASURE_ID_TO_MENTION_LABELS for why this can't be a naive
        # display_name string match: a "TIPS / Breakevens" mention in
        # text doesn't distinguish which of T5YIE/T10YIE/T5YIFR was
        # meant, so several measure_ids legitimately share one mention
        # count).
        base["mention_count"] = base["measure_id"].map(mention_counts).fillna(0)
        total_mentions = mention_counts.sum() if mention_counts.sum() > 0 else np.nan
        base["mention_share"] = base["mention_count"] / total_mentions
    else:
        base["mention_count"] = np.nan
        base["mention_share"] = np.nan

    if event_study_df is not None and not event_study_df.empty:
        event_summary = (
            event_study_df.groupby("measure_id")["t"].apply(lambda s: s.abs().max()).rename("event_response_max_abs_t")
        )
        base = base.merge(event_summary, on="measure_id", how="left")
    else:
        base["event_response_max_abs_t"] = np.nan

    if robustness_pass_rates is not None:
        base["robustness_pass_rate"] = base["measure_id"].map(robustness_pass_rates)
    else:
        base["robustness_pass_rate"] = np.nan

    # Simple, fully transparent rank-based convergence score: average
    # percentile rank across the available numeric evidence columns.
    # NOT a claimed "overall importance" weight -- purely a sort key to
    # surface convergence, with every underlying column still reported.
    rank_cols = ["univariate_t", "incremental_oos_r2", "permutation_importance_mean", "mention_share", "event_response_max_abs_t"]
    pct_ranks = base[rank_cols].abs().rank(pct=True)
    base["convergence_rank_score"] = pct_ranks.mean(axis=1, skipna=True)

    return base.sort_values("convergence_rank_score", ascending=False).reset_index(drop=True)
