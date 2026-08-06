"""Baseline reaction-function model and the univariate horse race
(spec sections 13-14).

Core specification:

    delta_stance(i,t) = a + b * ExpectationMeasure(t-) + controls(t-)
                        + SpeakerFE + DocumentTypeFE + TimeFE + e(i,t)

All continuous regressors are pre-standardized (z-scored) upstream in
src/features/expectations.py, so b is directly interpretable as "the
change in Fed stance associated with a one-standard-deviation move in
the expectation measure" (spec section 13). Standard errors are
clustered by speaker (FOMC-collective documents cluster on document
type) since observations from the same policymaker are not independent.

The univariate horse race (section 14) runs this exact specification
once per candidate measure with IDENTICAL controls and DV construction,
varying only the measure under test -- so coefficients across measures
are comparable and the eventual ranking isn't an artifact of
inconsistent specifications.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.models.registry import MACRO_CONTROL_COLUMNS, PRIMARY_MEASURES, MeasureRegistryEntry
from src.models.time_series_cv import expanding_window_splits, oos_r2
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_RELEVANCE_THRESHOLD = 0.10


def build_regression_sample(
    panel: pd.DataFrame,
    dv_col: str = "delta_stance",
    min_relevance: float = DEFAULT_RELEVANCE_THRESHOLD,
    extra_required_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Filter the model panel to monetary-policy-relevant documents with
    a non-missing dependent variable (drops the first document per
    speaker/type, which by construction has no delta_stance).
    """
    sample = panel[panel["monetary_policy_relevance"] >= min_relevance].copy()
    required = [dv_col] + (extra_required_cols or [])
    sample = sample.dropna(subset=[c for c in required if c in sample.columns])
    sample["event_date"] = pd.to_datetime(sample["event_date"])
    sample["year"] = sample["event_date"].dt.year
    sample["stance_group"] = sample["speaker"].fillna(sample["document_type"])
    return sample.sort_values("event_date").reset_index(drop=True)


def _clean_fe_name(col: str) -> str:
    return "Q(" + repr(col) + ")"


def _build_formula(dv_col: str, regressor_cols: list[str], control_cols: list[str],
                    speaker_fe: bool, doctype_fe: bool, year_fe: bool) -> str:
    terms = list(regressor_cols) + list(control_cols)
    if speaker_fe:
        terms.append("C(stance_group)")
    if doctype_fe:
        terms.append("C(document_type)")
    if year_fe:
        terms.append("C(year)")
    return f"{dv_col} ~ " + " + ".join(terms)


@dataclass
class ModelResult:
    formula: str
    n_obs: int
    r2: float
    adj_r2: float
    aic: float
    bic: float
    coefs: pd.DataFrame
    sample_start: pd.Timestamp
    sample_end: pd.Timestamp
    oos_r2: float = np.nan
    raw_result: object = field(default=None, repr=False)

    def coef_row(self, term: str) -> pd.Series | None:
        matches = self.coefs[self.coefs["term"] == term]
        return matches.iloc[0] if len(matches) else None


def _compute_oos_r2_ols(df: pd.DataFrame, formula: str, n_splits: int = 5) -> float:
    """Expanding-window OOS R^2 for an OLS formula.

    Builds the design matrix ONCE over the full (chronologically sorted)
    sample via patsy, then slices rows per fold -- rather than re-running
    the formula on each training subset. The latter would re-derive
    category dummies (C(year), C(stance_group), ...) from whatever levels
    happen to appear in that fold, so a category present only in the test
    fold (a new speaker, a new year) silently becomes an "unseen column"
    at predict time and statsmodels returns NaN for that row. With year
    and speaker fixed effects and an expanding window, that is not an
    edge case -- it happens on almost every fold. Building the matrix
    once against the FULL sample's categories fixes the column set up
    front, so every fold's test rows are always predictable (their
    dummy columns exist, just zero-filled if that level never appears in
    the training slice).
    """
    import patsy

    y_full, X_full = patsy.dmatrices(formula, data=df, return_type="dataframe")
    y_full = y_full.iloc[:, 0]
    n = len(df)
    scores = []
    for train_idx, test_idx in expanding_window_splits(n, n_splits=n_splits):
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]
        # Drop any column that is entirely zero in the training slice
        # (a category that never appears yet) to keep the fit well-posed.
        nonzero_cols = X_train.columns[(X_train != 0).any(axis=0)]
        X_train, X_test = X_train[nonzero_cols], X_test[nonzero_cols]
        try:
            beta, *_ = np.linalg.lstsq(X_train.values, y_train.values, rcond=None)
            y_pred = X_test.values @ beta
        except Exception:
            continue
        scores.append(oos_r2(y_test.values, y_pred, y_train_mean=y_train.mean()))
    return float(np.nanmean(scores)) if scores else np.nan


def fit_baseline_model(
    sample: pd.DataFrame,
    dv_col: str,
    regressor_cols: list[str],
    control_cols: list[str] | None = None,
    speaker_fe: bool = True,
    doctype_fe: bool = True,
    year_fe: bool = True,
    cluster_col: str = "stance_group",
    compute_oos: bool = True,
) -> ModelResult:
    control_cols = control_cols or []
    needed = [dv_col] + regressor_cols + control_cols
    df = sample.dropna(subset=[c for c in needed if c in sample.columns]).copy()
    if len(df) < 20:
        raise ValueError(f"Too few observations ({len(df)}) to fit a model for {regressor_cols}.")

    # build_regression_sample() normally creates 'stance_group' and
    # 'year', but callers sometimes pass a subset filtered directly from
    # the raw panel (e.g. event-study window subsets) that never went
    # through it. Create them here defensively so speaker_fe/year_fe/
    # cluster_col never reference an undefined column.
    if speaker_fe and "stance_group" not in df.columns:
        df["stance_group"] = df.get("speaker", pd.Series(index=df.index)).fillna(df.get("document_type", "unknown"))
    if cluster_col == "stance_group" and "stance_group" not in df.columns:
        df["stance_group"] = df.get("speaker", pd.Series(index=df.index)).fillna(df.get("document_type", "unknown"))
    if year_fe and "year" not in df.columns:
        df["year"] = pd.to_datetime(df["event_date"]).dt.year

    formula = _build_formula(dv_col, regressor_cols, control_cols, speaker_fe, doctype_fe, year_fe)
    model = smf.ols(formula, data=df)
    if cluster_col in df.columns and df[cluster_col].nunique() > 1:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
    else:
        result = model.fit(cov_type="HAC", cov_kwds={"maxlags": 4})

    coefs = pd.DataFrame(
        {
            "term": result.params.index,
            "coef": result.params.values,
            "se": result.bse.values,
            "t": result.tvalues.values,
            "pvalue": result.pvalues.values,
            "ci_low": result.conf_int()[0].values,
            "ci_high": result.conf_int()[1].values,
        }
    )

    # OOS R^2 deliberately uses a LEANER formula (regressors + macro
    # controls, no speaker/year fixed effects) even though the reported
    # coefficients above come from the full FE specification. A fixed
    # effect for a year/speaker unseen in the training fold has no
    # estimated value ex-ante -- it is not something a real forecaster
    # could have used -- and an expanding window's early folds have very
    # few observations relative to a full FE dummy set, which blows up
    # out-of-sample variance (verified empirically: the full-FE OOS R^2
    # on this project's real panel came out around -10 to -40, an
    # overfitting artifact, not a genuine finding). The lean formula is
    # what the incremental-R^2 / importance ranking in section 15/22
    # actually needs: how much predictive content does this measure add
    # on top of the macro environment, forecastable in real time.
    lean_formula = _build_formula(dv_col, regressor_cols, control_cols, False, False, False)
    oos = _compute_oos_r2_ols(df, lean_formula) if compute_oos else np.nan

    return ModelResult(
        formula=formula,
        n_obs=int(result.nobs),
        r2=result.rsquared,
        adj_r2=result.rsquared_adj,
        aic=result.aic,
        bic=result.bic,
        coefs=coefs,
        sample_start=df["event_date"].min(),
        sample_end=df["event_date"].max(),
        oos_r2=oos,
        raw_result=result,
    )


def run_univariate_horse_race(
    panel: pd.DataFrame,
    dv_col: str = "delta_stance",
    measures: list[MeasureRegistryEntry] | None = None,
    use_change: bool = False,
    control_cols: list[str] | None = None,
    min_relevance: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> pd.DataFrame:
    """Section 14: identical controls, one measure at a time. Returns a
    tidy summary table (one row per measure), NOT sorted by significance
    -- ranking is a separate, deliberate step (src/models/importance.py).
    """
    measures = measures or PRIMARY_MEASURES
    control_cols = control_cols if control_cols is not None else MACRO_CONTROL_COLUMNS
    rows = []
    for m in measures:
        reg_col = m.change_zscore_col if use_change else m.level_zscore_col
        sample = build_regression_sample(panel, dv_col=dv_col, min_relevance=min_relevance, extra_required_cols=[reg_col])
        if len(sample) < 30:
            logger.warning("Skipping %s: only %d usable observations.", m.measure_id, len(sample))
            continue
        try:
            result = fit_baseline_model(sample, dv_col, [reg_col], control_cols)
        except Exception as exc:
            # logger.error (not .exception): see baseline OOS-fit comment
            # -- full traceback formatting crashes on this Python/patsy
            # combination when a PatsyError is in the chain.
            logger.error("Failed to fit univariate model for %s: %s", m.measure_id, exc)
            continue
        coef_row = result.coef_row(reg_col)
        if coef_row is None:
            continue
        rows.append(
            {
                "measure_id": m.measure_id,
                "display_name": m.display_name,
                "source_family": m.source_family,
                "horizon_bucket": m.horizon_bucket,
                "regressor": reg_col,
                "coef": coef_row["coef"],
                "se": coef_row["se"],
                "t": coef_row["t"],
                "pvalue": coef_row["pvalue"],
                "adj_r2": result.adj_r2,
                "oos_r2": result.oos_r2,
                "aic": result.aic,
                "bic": result.bic,
                "n_obs": result.n_obs,
                "sample_start": result.sample_start,
                "sample_end": result.sample_end,
            }
        )
    return pd.DataFrame(rows)
