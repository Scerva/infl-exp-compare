import pandas as pd

from src.features.align_information_set import (
    SeriesAlignmentSpec,
    align_series_to_events,
    coverage_report,
)


def _events():
    return pd.DataFrame(
        {
            "document_id": ["d1", "d2", "d3"],
            "event_date": pd.to_datetime(["2024-01-10", "2024-02-15", "2024-03-01"]),
        }
    )


def test_align_series_to_events_uses_release_date_not_reference_date():
    # A value with reference_month=Feb but release_date=Feb 20 must not
    # be visible to the Feb 15 event (before release) but must be visible
    # to the Mar 1 event (after release).
    series = pd.DataFrame(
        {
            "value": [1.0, 2.0],
            "release_date": pd.to_datetime(["2024-01-01", "2024-02-20"]),
        }
    )
    spec = SeriesAlignmentSpec(name="survey", df=series, availability_col="release_date", value_cols=["value"])
    out = align_series_to_events(_events(), "event_date", [spec])
    assert out.loc[out["document_id"] == "d2", "survey_value"].iloc[0] == 1.0
    assert out.loc[out["document_id"] == "d3", "survey_value"].iloc[0] == 2.0


def test_align_series_to_events_same_day_release_excluded_strict():
    # Simulates the "prior close" behavior: a market observation dated
    # the SAME day as the event must not count (strict=True default).
    series = pd.DataFrame(
        {
            "value": [10.0, 20.0],
            "release_date": pd.to_datetime(["2024-01-09", "2024-01-10"]),
        }
    )
    events = pd.DataFrame({"document_id": ["e1"], "event_date": pd.to_datetime(["2024-01-10"])})
    spec = SeriesAlignmentSpec(name="mkt", df=series, availability_col="release_date", value_cols=["value"])
    out = align_series_to_events(events, "event_date", [spec])
    assert out["mkt_value"].iloc[0] == 10.0  # yesterday's value, not today's


def test_align_series_to_events_adds_days_since_release():
    series = pd.DataFrame({"value": [5.0], "release_date": pd.to_datetime(["2024-01-01"])})
    spec = SeriesAlignmentSpec(name="s", df=series, availability_col="release_date", value_cols=["value"])
    out = align_series_to_events(_events(), "event_date", [spec])
    row = out[out["document_id"] == "d1"].iloc[0]
    assert row["s_days_since_release"] == 9


def test_align_series_to_events_empty_series_produces_nan_columns():
    empty = pd.DataFrame(columns=["value", "release_date"])
    spec = SeriesAlignmentSpec(name="empty_series", df=empty, availability_col="release_date", value_cols=["value"])
    out = align_series_to_events(_events(), "event_date", [spec])
    assert "empty_series_value" in out.columns
    assert out["empty_series_value"].isna().all()


def test_align_series_to_events_multiple_series_independent():
    s1 = pd.DataFrame({"value": [1.0], "release_date": pd.to_datetime(["2024-01-01"])})
    s2 = pd.DataFrame({"value": [99.0], "release_date": pd.to_datetime(["2024-01-05"])})
    specs = [
        SeriesAlignmentSpec(name="a", df=s1, availability_col="release_date", value_cols=["value"]),
        SeriesAlignmentSpec(name="b", df=s2, availability_col="release_date", value_cols=["value"]),
    ]
    out = align_series_to_events(_events(), "event_date", specs)
    assert "a_value" in out.columns and "b_value" in out.columns
    assert out.loc[out["document_id"] == "d1", "a_value"].iloc[0] == 1.0
    assert out.loc[out["document_id"] == "d1", "b_value"].iloc[0] == 99.0


def test_coverage_report_computes_pct_and_staleness():
    series = pd.DataFrame({"value": [1.0], "release_date": pd.to_datetime(["2024-01-01"])})
    spec = SeriesAlignmentSpec(name="s", df=series, availability_col="release_date", value_cols=["value"])
    out = align_series_to_events(_events(), "event_date", [spec])
    report = coverage_report(out, [spec])
    assert report.iloc[0]["series"] == "s"
    assert report.iloc[0]["n_available"] == 3  # all 3 events are after 2024-01-01
    assert report.iloc[0]["coverage_pct"] == 100.0
