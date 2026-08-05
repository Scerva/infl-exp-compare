import os

import pandas as pd
import pytest

from src.data import atlanta_bie as bie_mod


def test_next_wednesday_after_friday_is_five_days_later():
    friday = pd.Timestamp("2026-07-17")  # Friday
    assert friday.weekday() == 4
    wed = bie_mod._next_wednesday_after(friday)
    assert wed == pd.Timestamp("2026-07-22")
    assert wed.weekday() == 2


def test_next_wednesday_after_wednesday_rolls_to_next_week():
    wed = pd.Timestamp("2026-07-22")
    nxt = bie_mod._next_wednesday_after(wed)
    assert nxt == pd.Timestamp("2026-07-29")


def test_find_year_ahead_mean_column_locates_question4_mean():
    raw = pd.DataFrame(
        [
            [None, None, None, None],
            [None, "Question 3: foo", None, "Question 4: unit costs"],
            [None, "Mean", "Variance", None],
            ["2020-01-01", 0.02, 0.001, None],
        ]
    )
    # Fix up so row1 col3 = Question4 marker, row2 col3='Mean' analog test:
    raw.iloc[1, 3] = "Question 4: unit costs"
    raw.iloc[2, 3] = "Mean"
    col = bie_mod._find_year_ahead_mean_column(raw)
    assert col == 3


def test_find_year_ahead_mean_column_raises_if_missing():
    raw = pd.DataFrame([[None], [None], [None], [None]])
    with pytest.raises(Exception):
        bie_mod._find_year_ahead_mean_column(raw)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SKIP_NETWORK_TESTS"), reason="network tests disabled")
def test_get_atlanta_bie_live_smoke():
    df = bie_mod.get_atlanta_bie()
    assert len(df) > 100
    assert (df["release_date"] > df["survey_period_end"]).all()
    # stored as decimal fraction, not percent
    assert df["value"].between(-0.05, 0.15).all()
