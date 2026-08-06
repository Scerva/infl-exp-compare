"""Multivariate horse race (spec section 15): all expectation measures
together, since they are highly correlated (see Table 3) and a
one-at-a-time univariate coefficient can be misleading when its signal
is shared with other measures.

Implements OLS-all-together, Ridge, and Elastic Net (time-series-CV-tuned
alpha), plus the diagnostics section 15 explicitly asks for
(correlation/VIF/condition number) before trusting any of them. Every
out-of-sample score uses expanding-window CV (src.models.time_series_cv),
never a random split.

"Incremental R^2 when a source or family is removed" (section 15's other
headline metric) is implemented as leave-one-measure-out and
leave-one-family-out OOS R^2 deltas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

from src.models.registry import MeasureRegistryEntry
from src.models.time_series_cv import expanding_window_splits, oos_r2
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MultivariateDiagnostics:
    correlation_matrix: pd.DataFrame
    vif: pd.DataFrame
    condition_number: float
    n_obs: int
    n_regressors: int


def compute_diagnostics(X: pd.DataFrame) -> MultivariateDiagnostics:
    """Correlation matrix, VIF per regressor, and the design matrix's
    condition number -- spec section 15's required multicollinearity
    checks, run BEFORE trusting any multivariate coefficient.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X_clean = X.dropna()
    corr = X_clean.corr()

    vif_rows = []
    X_with_const = np.column_stack([np.ones(len(X_clean)), X_clean.values])
    for i, col in enumerate(X_clean.columns):
        try:
            vif = variance_inflation_factor(X_with_const, i + 1)
        except Exception:
            vif = np.nan
        vif_rows.append({"regressor": col, "vif": vif})
    vif_df = pd.DataFrame(vif_rows)

    _, singular_values, _ = np.linalg.svd(X_clean.values - X_clean.values.mean(axis=0))
    condition_number = float(singular_values.max() / singular_values.min()) if singular_values.min() > 0 else np.inf

    return MultivariateDiagnostics(
        correlation_matrix=corr, vif=vif_df, condition_number=condition_number,
        n_obs=len(X_clean), n_regressors=X_clean.shape[1],
    )


def _fit_and_score_expanding(estimator_factory, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> tuple[list, list]:
    """Returns (fold_oos_r2_scores, fold_coefficients)."""
    scores, coefs = [], []
    for train_idx, test_idx in expanding_window_splits(len(y), n_splits=n_splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
        est = estimator_factory()
        est.fit(X_train_s, y_train)
        pred = est.predict(X_test_s)
        scores.append(oos_r2(y_test, pred, y_train_mean=y_train.mean()))
        coefs.append(getattr(est, "coef_", None))
    return scores, coefs


@dataclass
class MultivariateModelResult:
    method: str
    regressors: list[str]
    mean_oos_r2: float
    fold_oos_r2: list[float]
    full_sample_coef: pd.Series
    n_obs: int


def run_multivariate_models(
    panel_sample: pd.DataFrame,
    dv_col: str,
    regressor_cols: list[str],
    n_splits: int = 5,
) -> dict[str, MultivariateModelResult]:
    """Fits OLS, Ridge (CV-tuned), and Elastic Net (CV-tuned) with all
    regressors simultaneously. Returns one MultivariateModelResult per
    method, keyed 'ols' / 'ridge' / 'elasticnet'.
    """
    df = panel_sample.dropna(subset=[dv_col] + regressor_cols).sort_values("event_date")
    y = df[dv_col].values
    X = df[regressor_cols].values
    n = len(df)

    results = {}

    ols_scores, ols_coefs = _fit_and_score_expanding(lambda: LinearRegression(), X, y, n_splits)
    ols_full = LinearRegression().fit(StandardScaler().fit_transform(X), y)
    results["ols"] = MultivariateModelResult(
        "ols", regressor_cols, float(np.nanmean(ols_scores)), ols_scores,
        pd.Series(ols_full.coef_, index=regressor_cols), n,
    )

    alphas = np.logspace(-2, 3, 30)
    ridge_scores, _ = _fit_and_score_expanding(lambda: RidgeCV(alphas=alphas), X, y, n_splits)
    ridge_full = RidgeCV(alphas=alphas).fit(StandardScaler().fit_transform(X), y)
    results["ridge"] = MultivariateModelResult(
        "ridge", regressor_cols, float(np.nanmean(ridge_scores)), ridge_scores,
        pd.Series(ridge_full.coef_, index=regressor_cols), n,
    )

    enet_scores, _ = _fit_and_score_expanding(
        lambda: ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-4, 1, 20), max_iter=5000), X, y, n_splits
    )
    enet_full = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-4, 1, 20), max_iter=5000).fit(
        StandardScaler().fit_transform(X), y
    )
    results["elasticnet"] = MultivariateModelResult(
        "elasticnet", regressor_cols, float(np.nanmean(enet_scores)), enet_scores,
        pd.Series(enet_full.coef_, index=regressor_cols), n,
    )

    return results


def leave_one_out_incremental_r2(
    panel_sample: pd.DataFrame,
    dv_col: str,
    regressor_cols: list[str],
    grouping: dict[str, str] | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    """For each measure (or, if ``grouping`` maps measure->family, each
    family), compute OOS R^2 of the full Ridge model MINUS OOS R^2 with
    that measure/family removed. A large positive drop means that
    measure/family was carrying real incremental predictive content the
    others could not replace (spec section 15's headline "incremental
    out-of-sample R^2" metric, and section 16's family-removal version).
    """
    df = panel_sample.dropna(subset=[dv_col] + regressor_cols).sort_values("event_date")
    y = df[dv_col].values

    def _fit_ridge_oos(cols: list[str]) -> float:
        if not cols:
            return np.nan
        X = df[cols].values
        scores, _ = _fit_and_score_expanding(lambda: RidgeCV(alphas=np.logspace(-2, 3, 30)), X, y, n_splits)
        return float(np.nanmean(scores))

    full_r2 = _fit_ridge_oos(regressor_cols)

    groups = grouping or {c: c for c in regressor_cols}
    unique_groups = sorted(set(groups.get(c, c) for c in regressor_cols))

    rows = []
    for grp in unique_groups:
        remaining = [c for c in regressor_cols if groups.get(c, c) != grp]
        removed_r2 = _fit_ridge_oos(remaining)
        rows.append(
            {
                "removed": grp,
                "full_model_oos_r2": full_r2,
                "without_group_oos_r2": removed_r2,
                "incremental_oos_r2": full_r2 - removed_r2,
            }
        )
    return pd.DataFrame(rows).sort_values("incremental_oos_r2", ascending=False)


def pca_factor_scores(X: pd.DataFrame, n_components: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    """Transparent baseline factor model (spec section 3I / 16): PCA on
    standardized inputs. Returns (factor_scores_df, explained_variance_ratio).
    """
    from sklearn.decomposition import PCA

    X_clean = X.dropna()
    scaler = StandardScaler().fit(X_clean)
    X_s = scaler.transform(X_clean)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_s)
    cols = [f"factor_{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, index=X_clean.index, columns=cols)
    return scores_df, pca.explained_variance_ratio_
