"""Unit tests for src.data.fred using a monkeypatched cache layer (no network),
plus a real-network smoke test gated behind FRED_API_KEY + --run-network.
"""
import os

import pandas as pd
import pytest

from src.utils.cache import CachedResponse, FetchError
import src.data.fred as fred_mod


def _fake_cached_response(payload_meta: dict) -> CachedResponse:
    return CachedResponse(content=b"{}", from_cache=False, raw_path=None, meta_path=None, meta=payload_meta)


def test_get_fred_series_parses_and_validates(monkeypatch):
    payload = {
        "observations": [
            {"date": "2024-01-01", "value": "2.10", "realtime_start": "2024-06-01", "realtime_end": "2024-06-01"},
            {"date": "2024-01-02", "value": ".", "realtime_start": "2024-06-01", "realtime_end": "2024-06-01"},
            {"date": "2024-01-02", "value": "2.15", "realtime_start": "2024-06-01", "realtime_end": "2024-06-01"},
        ]
    }

    def fake_fetch_json(url, cache_key, subdir, params=None, headers=None, force_refresh=False, max_age_seconds=None):
        return payload, _fake_cached_response({})

    monkeypatch.setattr(fred_mod, "fetch_json", fake_fetch_json)
    monkeypatch.setenv("FRED_API_KEY", "dummy")

    df = fred_mod.get_fred_series("T5YIE")
    assert list(df.columns) == ["series_id", "date", "value", "realtime_start", "realtime_end"]
    # duplicate date collapsed to last (keep="last") -> 2.15 wins
    assert len(df) == 2
    row = df[df["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert row["value"] == 2.15
    assert (df["series_id"] == "T5YIE").all()


def test_get_fred_series_raises_on_missing_observations_key(monkeypatch):
    def fake_fetch_json(*a, **k):
        return {"error_message": "Bad Request"}, _fake_cached_response({})

    monkeypatch.setattr(fred_mod, "fetch_json", fake_fetch_json)
    monkeypatch.setenv("FRED_API_KEY", "dummy")

    with pytest.raises(FetchError):
        fred_mod.get_fred_series("BOGUS")


def test_get_fred_series_raises_on_empty_observations(monkeypatch):
    def fake_fetch_json(*a, **k):
        return {"observations": []}, _fake_cached_response({})

    monkeypatch.setattr(fred_mod, "fetch_json", fake_fetch_json)
    monkeypatch.setenv("FRED_API_KEY", "dummy")

    with pytest.raises(FetchError):
        fred_mod.get_fred_series("T5YIE")


def test_get_fred_series_raises_on_schema_change(monkeypatch):
    def fake_fetch_json(*a, **k):
        return {"observations": [{"date": "2024-01-01", "val": "2.1"}]}, _fake_cached_response({})

    monkeypatch.setattr(fred_mod, "fetch_json", fake_fetch_json)
    monkeypatch.setenv("FRED_API_KEY", "dummy")

    with pytest.raises(FetchError):
        fred_mod.get_fred_series("T5YIE")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        fred_mod._api_key()


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("FRED_API_KEY"), reason="FRED_API_KEY not set")
def test_get_fred_series_live_smoke():
    df = fred_mod.get_fred_series("T5YIE", start_date="2024-01-01", end_date="2024-01-31")
    assert len(df) > 15
    assert df["date"].is_monotonic_increasing
    assert df["value"].between(-5, 10).all() or df["value"].isna().any()
