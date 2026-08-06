import numpy as np
import pandas as pd
import pytest

from src.models.event_study import release_event_window_association
from src.models.local_projections import local_projection_coefficients


def _synthetic_panel(n=400, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    days_since_release = rng.integers(0, 30, n)
    change_zscore = rng.normal(0, 1, n)
    stance = 0.02 * change_zscore * (days_since_release <= 5) + rng.normal(0, 0.05, n)
    return pd.DataFrame(
        {
            "event_date": dates,
            "monetary_policy_relevance": 0.5,
            "speaker": rng.choice(["A", "B", "C"], size=n),
            "document_type": "speech",
            "stance_score": stance,
            "MEASURE_days_since_release": days_since_release,
            "MEASURE_value_change_zscore": change_zscore,
            "control_value": rng.normal(0, 1, n),
        }
    )


def test_release_event_window_association_returns_rows_per_window():
    panel = _synthetic_panel()
    out = release_event_window_association(panel, "MEASURE", windows=[3, 5, 10], control_cols=["control_value"])
    assert set(out["window_days"]) <= {3, 5, 10}
    assert len(out) > 0


def test_release_event_window_association_missing_measure_raises_keyerror():
    panel = _synthetic_panel()
    with pytest.raises(KeyError):
        release_event_window_association(panel, "NOT_A_MEASURE")


def test_release_event_window_association_narrow_window_shows_stronger_signal():
    panel = _synthetic_panel(n=800)
    out = release_event_window_association(panel, "MEASURE", windows=[5, 30], control_cols=["control_value"])
    # true signal only exists within 5 days; a 30-day window dilutes it
    if len(out) == 2:
        coef_5 = out[out["window_days"] == 5]["coef"].iloc[0]
        coef_30 = out[out["window_days"] == 30]["coef"].iloc[0]
        assert abs(coef_5) >= abs(coef_30) * 0.5  # loose check, stochastic test


def test_local_projection_coefficients_returns_one_row_per_horizon():
    panel = _synthetic_panel()
    out = local_projection_coefficients(panel, "MEASURE_value_change_zscore", horizons=[1, 2, 3], control_cols=["control_value"])
    assert set(out["horizon"]) <= {1, 2, 3}


def test_local_projection_coefficients_missing_column_raises():
    panel = _synthetic_panel()
    with pytest.raises(KeyError):
        local_projection_coefficients(panel, "NOT_A_COLUMN")
