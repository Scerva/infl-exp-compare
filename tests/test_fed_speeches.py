import os

import pandas as pd
import pytest

from src.data import fed_speeches as fs


def test_split_role_name_handles_known_roles():
    assert fs._split_role_name("Vice Chair for Supervision Michael S. Barr") == (
        "Vice Chair for Supervision",
        "Michael S. Barr",
    )
    assert fs._split_role_name("Governor Michelle W. Bowman") == ("Governor", "Michelle W. Bowman")
    assert fs._split_role_name("Chairman Alan Greenspan") == ("Chairman", "Alan Greenspan")
    assert fs._split_role_name("Chair Jerome H. Powell") == ("Chair", "Jerome H. Powell")


def test_split_role_name_no_known_role_returns_none_role():
    role, name = fs._split_role_name("Some Unlisted Title Jane Doe")
    assert role is None
    assert name == "Some Unlisted Title Jane Doe"


def test_parse_modern_index_extracts_expected_fields():
    html = """
    <div class="row">
        <div class="col-xs-3 col-md-2 eventlist__time"><time>1/17/2024</time></div>
        <div class="col-xs-9 col-md-10 eventlist__event">
            <p><a href="/newsevents/speech/barr20240117a.htm"><em>Opening Remarks</em></a></p>
            <p class="news__speaker">Vice Chair for Supervision Michael S. Barr</p>
            <p>At the Conference, Boston, Massachusetts</p>
        </div>
    </div>
    """
    records = fs._parse_modern_index(html, 2024)
    assert len(records) == 1
    r = records[0]
    assert r["document_id"] == "barr20240117a"
    assert r["event_date"] == pd.Timestamp("2024-01-17")
    assert r["title"] == "Opening Remarks"
    assert r["speaker_raw"] == "Vice Chair for Supervision Michael S. Barr"
    assert r["venue"] == "At the Conference, Boston, Massachusetts"


def test_parse_modern_index_raises_on_empty():
    with pytest.raises(Exception):
        fs._parse_modern_index("<html><body>nothing here</body></html>", 2024)


def test_parse_legacy_index_extracts_expected_fields():
    html = """
    <ul id="speechIndex">
      <li>
        December 11, 2003
        <div class="title"><a href="/boarddocs/speeches/2003/20031211/default.htm">Capitalism and globalization</a></div>
        <div class="speaker">Chairman Alan Greenspan</div>
        <div class="location">Before the World Affairs Council, Dallas, Texas</div>
      </li>
    </ul>
    """
    records = fs._parse_legacy_index(html, 2003)
    assert len(records) == 1
    r = records[0]
    assert r["document_id"] == "legacy_20031211"
    assert r["event_date"] == pd.Timestamp("2003-12-11")
    assert r["speaker_raw"] == "Chairman Alan Greenspan"


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SKIP_NETWORK_TESTS"), reason="network tests disabled")
def test_parse_modern_speech_body_live_sample():
    from src.utils.cache import fetch

    resp = fetch(
        "https://www.federalreserve.gov/newsevents/speech/barr20240117a.htm",
        cache_key="test_barr20240117a",
        subdir="fed_speeches",
        headers=fs.BROWSER_HEADERS,
        ext="html",
    )
    paragraphs, footnotes = fs._parse_modern_speech_body(resp.text())
    assert len(paragraphs) > 3
    assert len(footnotes) > 0
    assert "cyber" in " ".join(paragraphs).lower()


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SKIP_NETWORK_TESTS"), reason="network tests disabled")
def test_get_speeches_live_smoke_recent_partial_year():
    df = fs.get_speeches(2026, 2026)
    assert len(df) > 10
    assert (df["event_date"] == df["publication_date"]).all()
    assert df["clean_text"].str.len().min() > 100
    assert df["document_type"].eq("speech").all()
