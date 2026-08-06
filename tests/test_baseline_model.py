import numpy as np
import pandas as pd
import pytest

from src.models.baseline import build_regression_sample, fit_baseline_model, run_univariate_horse_race
from src.models.registry import MeasureRegistryEntry


def _synthetic_panel(n=200, seed=0, true_beta=0.05):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="W")
    x = rng.normal(0, 1, n)
    control = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.02, n)
    delta_stance = true_beta * x + 0.01 * control + noise
    speakers = rng.choice(["Powell", "Waller", "Cook", "Bowman"], size=n)
    doctypes = rng.choice(["speech", "fomc_statement"], size=n, p=[0.8, 0.2])
    return pd.DataFrame(
        {
            "event_date": dates,
            "document_type": doctypes,
            "speaker": speakers,
            "monetary_policy_relevance": 0.5,
            "delta_stance": delta_stance,
            "x_zscore": x,
            "control_value": control,
        }
    )


def test_build_regression_sample_filters_by_relevance():
    panel = _synthetic_panel(n=50)
    panel.loc[:10, "monetary_policy_relevance"] = 0.01
    sample = build_regression_sample(panel, dv_col="delta_stance", min_relevance=0.10)
    assert (sample["monetary_policy_relevance"] >= 0.10).all()
    assert len(sample) == 39


def test_build_regression_sample_drops_missing_dv():
    panel = _synthetic_panel(n=50)
    panel.loc[0:5, "delta_stance"] = np.nan
    sample = build_regression_sample(panel, dv_col="delta_stance")
    assert sample["delta_stance"].notna().all()


def test_fit_baseline_model_recovers_sign_of_true_coefficient():
    panel = _synthetic_panel(n=300, true_beta=0.1)
    sample = build_regression_sample(panel, dv_col="delta_stance")
    result = fit_baseline_model(
        sample, "delta_stance", ["x_zscore"], control_cols=["control_value"],
        speaker_fe=True, doctype_fe=True, year_fe=True,
    )
    coef = result.coef_row("x_zscore")
    assert coef is not None
    assert coef["coef"] > 0  # true_beta=0.1, positive
    assert coef["pvalue"] < 0.05


def test_fit_baseline_model_raises_on_too_few_observations():
    panel = _synthetic_panel(n=5)
    sample = build_regression_sample(panel, dv_col="delta_stance")
    with pytest.raises(ValueError):
        fit_baseline_model(sample, "delta_stance", ["x_zscore"])


def test_fit_baseline_model_oos_r2_is_finite_number():
    panel = _synthetic_panel(n=300, true_beta=0.15)
    sample = build_regression_sample(panel, dv_col="delta_stance")
    result = fit_baseline_model(sample, "delta_stance", ["x_zscore"], control_cols=["control_value"], compute_oos=True)
    assert np.isfinite(result.oos_r2)


def test_run_univariate_horse_race_returns_one_row_per_measure():
    panel = _synthetic_panel(n=300, true_beta=0.1)
    panel["y_zscore"] = panel["x_zscore"] * -1  # second synthetic measure
    panel["y_value_change_zscore"] = panel["y_zscore"]
    panel["x_value_change_zscore"] = panel["x_zscore"]
    measures = [
        MeasureRegistryEntry("X", "Measure X", "market", "near", "x_zscore", "x_value_change_zscore"),
        MeasureRegistryEntry("Y", "Measure Y", "household", "near", "y_zscore", "y_value_change_zscore"),
    ]
    results = run_univariate_horse_race(panel, dv_col="delta_stance", measures=measures, control_cols=["control_value"])
    assert len(results) == 2
    assert set(results["measure_id"]) == {"X", "Y"}


def test_run_univariate_horse_race_skips_measure_with_insufficient_data():
    panel = _synthetic_panel(n=300, true_beta=0.1)
    panel["sparse_zscore"] = np.nan
    panel.loc[:5, "sparse_zscore"] = 1.0  # only 6 non-null values
    panel["sparse_value_change_zscore"] = panel["sparse_zscore"]
    measures = [MeasureRegistryEntry("SPARSE", "Sparse", "market", "near", "sparse_zscore", "sparse_value_change_zscore")]
    results = run_univariate_horse_race(panel, dv_col="delta_stance", measures=measures, control_cols=["control_value"])
    assert len(results) == 0
