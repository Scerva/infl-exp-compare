import os

import pandas as pd
import pytest

from src.data import philly_spf as spf_mod


def test_parse_release_dates_txt_handles_year_continuation_and_footnotes():
    text = (
        "Survey         True Deadline Date       News Release Date\n"
        "\n"
        "1990 Q2             8/23/90*            8/31/90*\n"
        "     Q3             8/23/90             8/31/90\n"
        "     Q4             11/22/90            11/28/90\n"
        "\n"
        "1991 Q1             2/16/91             2/21/91\n"
        "2026 Q1             3/2/26**            3/6/26** \n"
    )
    df = spf_mod._parse_release_dates_txt(text)
    assert len(df) == 5
    row = df[(df.year == 1990) & (df.quarter == 3)].iloc[0]
    assert row["release_date"] == pd.Timestamp("1990-08-31")
    row_q1 = df[(df.year == 1991) & (df.quarter == 1)].iloc[0]
    assert row_q1["release_date"] == pd.Timestamp("1991-02-21")
    row_2026 = df[(df.year == 2026) & (df.quarter == 1)].iloc[0]
    assert row_2026["release_date"] == pd.Timestamp("2026-03-06")


def test_parse_release_dates_txt_raises_on_garbage():
    with pytest.raises(Exception):
        spf_mod._parse_release_dates_txt("nothing useful here\nno dates at all\n")


def test_quarter_start():
    assert spf_mod._quarter_start(2024, 1) == pd.Timestamp("2024-01-01")
    assert spf_mod._quarter_start(2024, 4) == pd.Timestamp("2024-10-01")


def test_estimate_release_date_fallback_is_within_quarter():
    d = spf_mod._estimate_release_date(1985, 2)
    assert d.month == 5  # second month of Q2
    assert d.year == 1985


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SKIP_NETWORK_TESTS"), reason="network tests disabled")
def test_get_spf_series_live_smoke():
    df = spf_mod.get_spf_series("CPI10", "median")
    assert len(df) > 50
    assert (df["release_date"] > df["reference_quarter"]).all()
    assert df["value"].between(-2, 15).all()
    assert not df["release_date_estimated"].any()  # CPI10 starts well after 1990:Q2
