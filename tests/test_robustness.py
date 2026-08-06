import numpy as np
import pandas as pd

from src.models.registry import MeasureRegistryEntry
from src.models.robustness import run_robustness_checks, summarize_robustness_pass_rates


def _synthetic_panel(n=600, seed=9, true_beta=0.1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="W")
    x = rng.normal(0, 1, n)
    x_change = rng.normal(0, 1, n)
    control = rng.normal(0, 1, n)
    delta_stance = true_beta * x + 0.01 * control + rng.normal(0, 0.02, n)
    return pd.DataFrame(
        {
            "event_date": dates,
            "document_type": rng.choice(["speech", "fomc_statement", "fomc_minutes"], size=n, p=[0.7, 0.15, 0.15]),
            "speaker": rng.choice(["Powell", "Waller", "Cook", None], size=n, p=[0.3, 0.3, 0.3, 0.1]),
            "monetary_policy_relevance": 0.5,
            "delta_stance": delta_stance,
            "x_zscore": x,
            "x_value_change_zscore": x_change,
            "control_value": control,
        }
    )


def test_run_robustness_checks_includes_baseline_and_variants():
    panel = _synthetic_panel()
    measures = [MeasureRegistryEntry("X", "Measure X", "market", "near", "x_zscore", "x_value_change_zscore")]
    out = run_robustness_checks(panel, measures, control_cols=["control_value"], min_obs=20)
    assert "baseline_full_sample" in out["variant"].values
    assert (out["measure_id"] == "X").all()
    assert len(out) > 1


def test_run_robustness_checks_baseline_row_marked_same_sign():
    panel = _synthetic_panel()
    measures = [MeasureRegistryEntry("X", "Measure X", "market", "near", "x_zscore", "x_value_change_zscore")]
    out = run_robustness_checks(panel, measures, control_cols=["control_value"], min_obs=20)
    baseline_row = out[out["variant"] == "baseline_full_sample"].iloc[0]
    assert baseline_row["same_sign_as_baseline"] == True


def test_summarize_robustness_pass_rates_excludes_baseline():
    df = pd.DataFrame(
        {
            "measure_id": ["X", "X", "X", "Y", "Y"],
            "variant": ["baseline_full_sample", "pre_2020", "post_2022_tightening", "baseline_full_sample", "pre_2020"],
            "same_sign_as_baseline": [True, True, False, True, True],
        }
    )
    rates = summarize_robustness_pass_rates(df)
    assert rates["X"] == 0.5  # 1 of 2 non-baseline variants passed
    assert rates["Y"] == 1.0


def test_run_robustness_checks_skips_measure_with_insufficient_data():
    panel = _synthetic_panel(n=10)
    measures = [MeasureRegistryEntry("SPARSE", "Sparse", "market", "near", "x_zscore", "x_value_change_zscore")]
    out = run_robustness_checks(panel, measures, control_cols=["control_value"], min_obs=100)
    assert out.empty
