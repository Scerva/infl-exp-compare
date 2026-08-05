"""FRED / ALFRED downloader.

Implements the required ``get_fred_series`` primitive (spec section 3A) on
top of the shared cache layer: raw JSON responses are persisted before any
transformation, and ALFRED vintage parameters (``realtime_start``,
``realtime_end``) are supported so historical analyses can use only the
data vintage that was actually public at the time, rather than today's
revised history.

Series covered from the source catalog (config/sources.yaml, provider=FRED):
  market:  T5YIE, T10YIE, T5YIFR
  model:   EXPINF1YR, EXPINF5YR, EXPINF10YR, EXPINF30YR
  household: MICH
plus arbitrary macro-control series (CPI, PCE, unemployment, yields, ...).
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from src.utils.cache import FetchError, fetch_json
from src.utils.logging import get_logger

load_dotenv()

logger = get_logger(__name__)

FRED_API_HOST = "https://api.stlouisfed.org"
FRED_OBS_ENDPOINT = f"{FRED_API_HOST}/fred/series/observations"
FRED_SERIES_ENDPOINT = f"{FRED_API_HOST}/fred/series"


def _api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or export FRED_API_KEY in your shell."
        )
    return key


def get_fred_series(
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    realtime_start: Optional[str] = None,
    realtime_end: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch one FRED series via ``fred/series/observations``.

    Returns a validated, sorted, de-duplicated DataFrame with columns:
    ``date`` (observation/reference date), ``value`` (float, NaN for
    FRED's "." missing marker), ``realtime_start``, ``realtime_end``,
    ``series_id``.

    ``realtime_start``/``realtime_end`` (ALFRED vintage params) let callers
    pin the data vintage as it was known as of a given date, rather than
    assuming today's revised series applies retroactively.
    """
    params = {
        "series_id": series_id,
        "api_key": _api_key(),
        "file_type": "json",
    }
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date
    if realtime_start:
        params["realtime_start"] = realtime_start
    if realtime_end:
        params["realtime_end"] = realtime_end

    cache_key = "_".join(
        [series_id]
        + [f"{k}-{v}" for k, v in sorted(params.items()) if k not in ("api_key", "file_type")]
    ) or series_id

    payload, resp = fetch_json(
        FRED_OBS_ENDPOINT,
        cache_key=cache_key,
        subdir="fred",
        params=params,
        force_refresh=force_refresh,
    )

    if "observations" not in payload:
        raise FetchError(
            f"FRED response for {series_id} missing 'observations' key. "
            f"Response keys: {list(payload.keys())}. This usually means an "
            f"invalid series_id or API key."
        )
    obs = payload["observations"]
    if len(obs) == 0:
        raise FetchError(f"FRED returned zero observations for {series_id} (params={params}).")

    df = pd.DataFrame(obs)
    expected_cols = {"date", "value", "realtime_start", "realtime_end"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise FetchError(
            f"FRED observations for {series_id} missing expected columns {missing_cols}. "
            f"Got columns: {list(df.columns)}. FRED schema may have changed."
        )

    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df["realtime_end"] = pd.to_datetime(df["realtime_end"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # "." -> NaN
    df["series_id"] = series_id

    n_before = len(df)
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    if len(df) != n_before:
        logger.warning("%s: dropped %d duplicate-date rows", series_id, n_before - len(df))

    n_missing = df["value"].isna().sum()
    if n_missing:
        logger.info("%s: %d/%d observations missing (FRED '.')", series_id, n_missing, len(df))

    logger.info(
        "%s: %d obs, %s to %s (%s)",
        series_id,
        len(df),
        df["date"].min().date(),
        df["date"].max().date(),
        "cache" if resp.from_cache else "downloaded",
    )
    return df[["series_id", "date", "value", "realtime_start", "realtime_end"]]


def get_fred_series_multi(series_ids: list[str], **kwargs) -> pd.DataFrame:
    """Fetch several FRED series and stack them long-format."""
    frames = []
    for sid in series_ids:
        try:
            frames.append(get_fred_series(sid, **kwargs))
        except FetchError as exc:
            logger.error("Skipping %s: %s", sid, exc)
    if not frames:
        raise FetchError(f"All requested FRED series failed: {series_ids}")
    return pd.concat(frames, ignore_index=True)
