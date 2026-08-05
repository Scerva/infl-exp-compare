import os

import pandas as pd
import pytest

from src.features import macro_controls as mc


def test_first_friday_on_or_after():
    # July 1, 2026 is a Wednesday -> first Friday is July 3
    d = mc._first_friday_on_or_after(pd.Timestamp("2026-07-01"))
    assert d == pd.Timestamp("2026-07-03")
    assert d.weekday() == 4


def test_compute_release_date_fixed_day_of_month():
    series_def = mc.MacroSeriesDef("CPIAUCSL", "monthly", "fixed_day_of_month", 1, 13)
    rd, est = mc._compute_release_date(pd.Timestamp("2024-06-01"), series_def)
    assert rd == pd.Timestamp("2024-07-13")
    assert est is True


def test_compute_release_date_same_day():
    series_def = mc.MacroSeriesDef("DGS10", "daily", "same_day")
    rd, est = mc._compute_release_date(pd.Timestamp("2024-06-05"), series_def)
    assert rd == pd.Timestamp("2024-06-05")
    assert est is False


def test_compute_release_date_next_business_day_skips_weekend():
    series_def = mc.MacroSeriesDef("DCOILWTICO", "daily", "next_business_day")
    # 2024-06-07 is a Friday -> next business day should be Monday 06-10
    rd, est = mc._compute_release_date(pd.Timestamp("2024-06-07"), series_def)
    assert rd == pd.Timestamp("2024-06-10")


def test_get_macro_series_unknown_name_raises():
    with pytest.raises(KeyError):
        mc.get_macro_series("NOT_A_SERIES")


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SKIP_NETWORK_TESTS"), reason="network tests disabled")
def test_get_macro_series_live_smoke():
    df = mc.get_macro_series("CPI_HEADLINE", start_date="2020-01-01")
    assert len(df) > 20
    assert (df["release_date"] > df["reference_date"]).all()
