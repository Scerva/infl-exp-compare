import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data import fomc


def test_get_minutes_skips_nan_minutes_url_without_crashing(monkeypatch):
    # Regression test: `not float('nan')` is False in Python, so a naive
    # `if not m["minutes_url"]` guard fails to skip NaN and passes the
    # literal string "nan" to requests, raising MissingSchema. Meetings
    # with no minutes at all (unscheduled conference calls; the current
    # in-progress meeting) must be skipped via pd.isna(), not truthiness.
    meetings = pd.DataFrame(
        [
            {
                "meeting_start_date": pd.Timestamp("2026-07-28"),
                "meeting_end_date": pd.Timestamp("2026-07-29"),
                "policy_decision_date": pd.Timestamp("2026-07-29"),
                "minutes_release_date": pd.NaT,
                "minutes_url": float("nan"),
                "statement_url": "https://example.com/statement.htm",
            }
        ]
    )
    monkeypatch.setattr(fomc, "get_fomc_meetings", lambda *a, **k: meetings)
    with pytest.raises(Exception) as exc_info:
        fomc.get_minutes(2026, 2026)
    # Must fail with "no minutes parsed" (correctly skipped), never a
    # requests MissingSchema error from trying to fetch url "nan".
    assert "MissingSchema" not in str(exc_info.value)
    assert "No FOMC minutes parsed" in str(exc_info.value)


def test_parse_meeting_dates_same_month():
    start, end = fomc._parse_meeting_dates("January", "27-28", 2024)
    assert start == pd.Timestamp("2024-01-27")
    assert end == pd.Timestamp("2024-01-28")


def test_parse_meeting_dates_cross_month_slash_form():
    start, end = fomc._parse_meeting_dates("Jan/Feb", "31-1", 2023)
    assert start == pd.Timestamp("2023-01-31")
    assert end == pd.Timestamp("2023-02-01")


def test_parse_meeting_dates_single_day():
    start, end = fomc._parse_meeting_dates("March", "13", 2012)
    assert start == pd.Timestamp("2012-03-13")
    assert end == pd.Timestamp("2012-03-13")


def test_parse_meeting_dates_strips_sep_asterisk():
    start, end = fomc._parse_meeting_dates("June", "17-18*", 2026)
    assert start == pd.Timestamp("2026-06-17")
    assert end == pd.Timestamp("2026-06-18")


def test_parse_meeting_dates_strips_notation_vote_suffix():
    # _parse_meeting_dates itself just strips parenthetical text and parses
    # the date; filtering out notation-vote entries entirely happens one
    # level up in _parse_calendar_page (they have no minutes/statement links).
    start, end = fomc._parse_meeting_dates("August", "22 (notation vote)", 2025)
    assert start == pd.Timestamp("2025-08-22")
    assert end == pd.Timestamp("2025-08-22")


def test_extract_meeting_links_modern_scheme():
    html = '<a href="/newsevents/pressreleases/monetary20240131a.htm">Statement</a> <a href="/monetarypolicy/fomcminutes20240131.htm">HTML</a> Minutes (Released February 21, 2024)'
    links = fomc._extract_meeting_links(html)
    assert links["decision_yyyymmdd"] == "20240131"
    assert links["minutes_release_date"] == pd.Timestamp("2024-02-21")
    assert "pressreleases/monetary20240131a.htm" in links["statement_url"]


def test_extract_meeting_links_legacy_2003_scheme():
    html = (
        '<a href="/boarddocs/press/monetary/2003/20030129/default.htm">Statement</a> '
        '<a href="/fomc/minutes/20030129.htm">Minutes</a> (Released March 20, 2003)'
    )
    links = fomc._extract_meeting_links(html)
    assert links["decision_yyyymmdd"] == "20030129"
    assert links["minutes_release_date"] == pd.Timestamp("2003-03-20")
    assert links["minutes_url"].endswith("/fomc/minutes/20030129.htm")


def test_extract_meeting_links_2006_2010_scheme():
    html = '<a href="/newsevents/press/monetary/20070131a.htm">Statement</a> <a href="/monetarypolicy/fomcminutes20070131.htm">HTML</a> Minutes (Released March 20, 2007)'
    links = fomc._extract_meeting_links(html)
    assert links["decision_yyyymmdd"] == "20070131"


def test_parse_historical_page_header_regex_matches_conference_call():
    html = """
    <div class="panel panel-default">
      <div class="panel-heading"><h5>August 10 Conference Call - 2007</h5></div>
      <div class="row divided-row panel-body">
        <p><a href="/boarddocs/press/monetary/2007/20070810/default.htm">Statement</a></p>
      </div>
    </div>
    """
    meetings = fomc._parse_historical_page(html, 2007)
    assert len(meetings) == 1
    assert meetings[0]["policy_decision_date"] == pd.Timestamp("2007-08-10")


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SKIP_NETWORK_TESTS"), reason="network tests disabled")
def test_get_fomc_meetings_live_smoke_spans_url_eras():
    df = fomc.get_fomc_meetings(2003, 2024)
    assert len(df) > 150
    assert df["policy_decision_date"].is_monotonic_increasing
    known = df[df["policy_decision_date"] == pd.Timestamp("2015-01-28")]
    assert len(known) == 1
    assert known.iloc[0]["minutes_release_date"] == pd.Timestamp("2015-02-18")


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SKIP_NETWORK_TESTS"), reason="network tests disabled")
def test_get_statements_live_smoke_extracts_rate_language():
    df = fomc.get_statements(2024, 2024)
    assert len(df) == 8
    assert df["rate_decision_text"].notna().sum() >= 7
    assert (df["event_date"] == df["publication_date"]).all()


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SKIP_NETWORK_TESTS"), reason="network tests disabled")
def test_get_minutes_live_smoke_publication_after_decision():
    df = fomc.get_minutes(2024, 2024)
    assert len(df) == 8
    assert (df["publication_date"] > df["event_date"]).all()
    assert df["clean_text"].str.len().min() > 5000
