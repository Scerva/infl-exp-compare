import numpy as np
import pandas as pd
import pytest

from src.models.regularized import compute_diagnostics, leave_one_out_incremental_r2, pca_factor_scores, run_multivariate_models


def _synthetic_sample(n=300, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="W")
    a = rng.normal(0, 1, n)
    b = 0.8 * a + rng.normal(0, 0.3, n)  # correlated with a
    c = rng.normal(0, 1, n)  # independent, no signal
    y = 0.1 * a + rng.normal(0, 0.05, n)  # only a matters
    return pd.DataFrame({"event_date": dates, "dv": y, "a_zscore": a, "b_zscore": b, "c_zscore": c})


def test_compute_diagnostics_returns_reasonable_condition_number():
    df = _synthetic_sample()
    diag = compute_diagnostics(df[["a_zscore", "b_zscore", "c_zscore"]])
    assert diag.n_obs == len(df)
    assert diag.condition_number > 1
    assert set(diag.vif["regressor"]) == {"a_zscore", "b_zscore", "c_zscore"}


def test_compute_diagnostics_correlated_regressors_get_higher_vif_than_independent():
    df = _synthetic_sample()
    diag = compute_diagnostics(df[["a_zscore", "b_zscore", "c_zscore"]])
    vif_a = diag.vif.set_index("regressor").loc["a_zscore", "vif"]
    vif_c = diag.vif.set_index("regressor").loc["c_zscore", "vif"]
    assert vif_a > vif_c


def test_run_multivariate_models_returns_all_three_methods():
    df = _synthetic_sample()
    results = run_multivariate_models(df, "dv", ["a_zscore", "b_zscore", "c_zscore"], n_splits=4)
    assert set(results.keys()) == {"ols", "ridge", "elasticnet"}
    for r in results.values():
        assert r.n_obs == len(df)
        assert len(r.full_sample_coef) == 3


def test_leave_one_out_incremental_r2_identifies_informative_regressor():
    df = _synthetic_sample(n=400)
    loo = leave_one_out_incremental_r2(df, "dv", ["a_zscore", "b_zscore", "c_zscore"], n_splits=4)
    assert set(loo["removed"]) == {"a_zscore", "b_zscore", "c_zscore"}
    # removing 'a' (the true signal) should hurt more than removing 'c' (pure noise)
    inc_a = loo.set_index("removed").loc["a_zscore", "incremental_oos_r2"]
    inc_c = loo.set_index("removed").loc["c_zscore", "incremental_oos_r2"]
    assert inc_a >= inc_c


def test_leave_one_out_incremental_r2_supports_family_grouping():
    df = _synthetic_sample(n=400)
    grouping = {"a_zscore": "family1", "b_zscore": "family1", "c_zscore": "family2"}
    loo = leave_one_out_incremental_r2(df, "dv", ["a_zscore", "b_zscore", "c_zscore"], grouping=grouping, n_splits=4)
    assert set(loo["removed"]) == {"family1", "family2"}


def test_pca_factor_scores_shape_and_variance_ratio():
    df = _synthetic_sample()
    scores, evr = pca_factor_scores(df[["a_zscore", "b_zscore", "c_zscore"]], n_components=2)
    assert scores.shape == (len(df), 2)
    assert len(evr) == 2
    assert 0 <= evr[0] <= 1
    assert evr[0] >= evr[1]  # first component explains most variance
