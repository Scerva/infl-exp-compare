import os

import pandas as pd
import pytest

from src.data import livingston as liv_mod


def test_fix_two_digit_year_overflow_shifts_future_dates_back_a_century():
    dates = pd.Series(pd.to_datetime(["2046-06-01", "2024-06-01", "2099-12-01"]))
    fixed = liv_mod._fix_two_digit_year_overflow(dates)
    assert fixed.iloc[0] == pd.Timestamp("1946-06-01")
    assert fixed.iloc[1] == pd.Timestamp("2024-06-01")  # untouched, not in the future
    assert fixed.iloc[2] == pd.Timestamp("1999-12-01")


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SKIP_NETWORK_TESTS"), reason="network tests disabled")
def test_get_livingston_series_live_smoke():
    df = liv_mod.get_livingston_series()
    assert len(df) > 30
    assert (df["release_date"] >= df["reference_date"]).all()
    assert df["value"].between(-2, 15).all()
    # semiannual: reference months should only be June or December
    assert set(df["reference_date"].dt.month.unique()) <= {6, 12}
