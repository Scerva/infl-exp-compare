import pandas as pd
import pytest

from src.utils.dates import asof_available, asof_merge_strict, prior_close_date, to_ts


def test_prior_close_date_date_only_uses_previous_business_day():
    # A date-only event (no intraday timestamp) is treated conservatively:
    # prior close is the business day before, never same-day.
    d = prior_close_date("2024-06-12")  # Wednesday
    assert d == pd.Timestamp("2024-06-11")


def test_prior_close_date_skips_weekend():
    d = prior_close_date("2024-06-10")  # Monday -> prior calendar day is Sunday
    assert d.weekday() < 5
    assert d == pd.Timestamp("2024-06-07")  # Friday


def test_asof_available_never_returns_future_value():
    panel = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0],
            "release_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        }
    )
    # as_of strictly before the Feb release -> only Jan value visible
    v = asof_available(panel, "value", "release_date", as_of=pd.Timestamp("2024-01-15"))
    assert v == 1.0

    # as_of exactly on Feb release with strict=True -> Feb not yet counted
    v_strict = asof_available(panel, "value", "release_date", as_of=pd.Timestamp("2024-02-01"), strict=True)
    assert v_strict == 1.0

    # as_of exactly on Feb release with strict=False -> Feb counted
    v_loose = asof_available(panel, "value", "release_date", as_of=pd.Timestamp("2024-02-01"), strict=False)
    assert v_loose == 2.0

    # nothing available yet -> None, not NaN silently coerced
    v_none = asof_available(panel, "value", "release_date", as_of=pd.Timestamp("2023-01-01"))
    assert v_none is None


def test_asof_merge_strict_matches_expected_lookback():
    events = pd.DataFrame(
        {
            "document_id": ["a", "b", "c"],
            "event_ts": pd.to_datetime(["2024-01-10", "2024-02-15", "2024-03-01"]),
        }
    )
    series = pd.DataFrame(
        {
            "value": [10.0, 20.0, 30.0],
            "release_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-02-20"]),
        }
    )
    out = asof_merge_strict(
        events,
        event_time_col="event_ts",
        series=series,
        availability_col="release_date",
        value_cols=["value"],
        prefix="x",
    )
    assert list(out["x_value"]) == [10.0, 20.0, 30.0]
    # order of input events must be preserved
    assert list(out["document_id"]) == ["a", "b", "c"]


def test_to_ts_strips_timezone():
    ts = to_ts(pd.Timestamp("2024-01-01", tz="US/Eastern"))
    assert ts.tzinfo is None
