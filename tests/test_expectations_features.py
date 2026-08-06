import numpy as np
import pandas as pd

from src.features.expectations import anchoring_gap, compute_slope, transform_series


def _synthetic_daily(n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    values = 2.0 + np.cumsum(rng.normal(0, 0.01, n))
    return pd.DataFrame({"date": dates, "value": values})


def test_transform_series_adds_expected_columns_daily():
    df = _synthetic_daily()
    out = transform_series(df, "date", "value", "daily", target_anchor=2.0)
    for col in [
        "value_level", "value_change_1d", "value_change_5d", "value_change_20d",
        "value_zscore", "value_dev_from_trailing_mean", "value_pctile",
        "value_change_zscore", "value_dev_from_target",
    ]:
        assert col in out.columns


def test_transform_series_change_1d_matches_diff():
    df = _synthetic_daily()
    out = transform_series(df, "date", "value", "daily")
    expected = df["value"].diff(1)
    pd.testing.assert_series_equal(out["value_change_1d"], expected, check_names=False)


def test_transform_series_dev_from_target_is_signed():
    df = pd.DataFrame({"date": pd.bdate_range("2023-01-01", periods=5), "value": [1.5, 2.0, 2.5, 1.8, 2.2]})
    out = transform_series(df, "date", "value", "daily", target_anchor=2.0)
    expected = [-0.5, 0.0, 0.5, -0.2, 0.2]
    for actual, exp in zip(out["value_dev_from_target"].tolist(), expected):
        assert abs(actual - exp) < 1e-9


def test_transform_series_no_target_anchor_omits_column():
    df = _synthetic_daily(n=10)
    out = transform_series(df, "date", "value", "daily", target_anchor=None)
    assert "value_dev_from_target" not in out.columns


def test_transform_series_sorts_and_dedups():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02", "2023-01-01"]),
            "value": [3.0, 1.0, 2.0, 1.5],  # duplicate 2023-01-01, keep last (1.5)
        }
    )
    out = transform_series(df, "date", "value", "daily")
    assert list(out["date"]) == list(pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]))
    assert out.loc[out["date"] == pd.Timestamp("2023-01-01"), "value"].iloc[0] == 1.5


def test_transform_series_monthly_frequency_uses_monthly_windows():
    dates = pd.date_range("2015-01-01", periods=36, freq="MS")
    values = np.linspace(1.5, 2.5, 36)
    df = pd.DataFrame({"date": dates, "value": values})
    out = transform_series(df, "date", "value", "monthly")
    assert "value_change_12m" in out.columns
    assert "value_change_1d" not in out.columns


def test_compute_slope():
    df = pd.DataFrame({"short": [2.0, 2.1], "long": [2.3, 2.0]})
    out = compute_slope(df, "short", "long", "slope_10y_1y")
    for actual, exp in zip(out["slope_10y_1y"].tolist(), [0.3, -0.1]):
        assert abs(actual - exp) < 1e-9


def test_anchoring_gap_is_absolute_value():
    df = pd.DataFrame({"value": [1.5, 2.5]})
    out = anchoring_gap(df, "value", target_anchor=2.0)
    assert out["value_anchoring_gap"].tolist() == [0.5, 0.5]


def test_anchoring_gap_custom_output_column_name():
    df = pd.DataFrame({"value": [1.0]})
    out = anchoring_gap(df, "value", target_anchor=2.0, out_col="my_gap")
    assert "my_gap" in out.columns
    assert "value_anchoring_gap" not in out.columns
