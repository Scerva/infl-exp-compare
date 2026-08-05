import os

import pandas as pd
import pytest

from src.data import nyfed_sce as sce_mod


def test_reconstruct_release_date_second_tuesday_of_following_month():
    d = sce_mod._reconstruct_release_date(pd.Timestamp("2024-01-01"))
    assert d.weekday() == 1  # Tuesday
    assert d.month == 2
    # second Tuesday of Feb 2024 is Feb 13
    assert d == pd.Timestamp("2024-02-13")


def test_find_header_row_locates_correct_row():
    rows = [
        ("junk",),
        (None, "Median one-year ahead expected inflation rate"),
    ]
    idx = sce_mod._find_header_row(rows, "Median one-year ahead expected inflation rate")
    assert idx == 1


def test_find_header_row_raises_if_missing():
    with pytest.raises(Exception):
        sce_mod._find_header_row([("a", "b")], "not present")


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SKIP_NETWORK_TESTS"), reason="network tests disabled")
def test_get_sce_inflation_expectations_live_smoke():
    df = sce_mod.get_sce_inflation_expectations()
    assert len(df) > 100
    assert df["reference_month"].min() <= pd.Timestamp("2013-07-01")
    # release_date must always be strictly after reference_month
    assert (df["release_date"] > df["reference_month"]).all()
    assert df["median_1yr"].between(-5, 30).all()
