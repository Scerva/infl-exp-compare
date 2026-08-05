import os

import pandas as pd
import pytest

from src.data import atsix as atsix_mod


def test_estimate_release_date_is_20th_of_following_month():
    d = atsix_mod._estimate_release_date(pd.Timestamp("2024-03-01"))
    assert d == pd.Timestamp("2024-04-20")


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SKIP_NETWORK_TESTS"), reason="network tests disabled")
def test_get_atsix_live_smoke():
    df = atsix_mod.get_atsix()
    assert len(df) > 1000
    assert df["horizon_months"].min() >= 3
    assert df["horizon_months"].max() <= 120
    assert (df["release_date"] > df["reference_month"]).all()
    assert df["value"].between(-2, 10).all()


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SKIP_NETWORK_TESTS"), reason="network tests disabled")
def test_get_atsix_selected_horizons_computes_slopes():
    wide = atsix_mod.get_atsix_selected_horizons()
    assert "atsix_slope_10y_1y" in wide.columns
    assert len(wide) > 100
