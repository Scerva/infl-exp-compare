"""Shared HTTP fetch + local disk cache used by every data-source adapter.

Design goals (see project spec sections 29-31):
  - never hit a remote source twice for the same content unless forced;
  - always persist the raw response bytes before any transformation, for
    provenance;
  - always write a JSON metadata sidecar (download timestamp, source URL,
    sha256, HTTP status, byte count) next to the raw file;
  - retry transient network failures with backoff, but fail loudly (no
    silent empty-DataFrame returns) on persistent failure or unexpected
    content;
  - use a descriptive User-Agent and reasonable timeouts so we are polite
    to the (mostly small, public-sector) servers we depend on.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"

DEFAULT_TIMEOUT = 30
DEFAULT_USER_AGENT = os.environ.get(
    "SCRAPER_USER_AGENT",
    "infl-exp-compare-research/0.1 (academic research; contact unset)",
)


class FetchError(RuntimeError):
    """Raised when a remote source cannot be retrieved or fails validation."""


@dataclass
class CachedResponse:
    content: bytes
    from_cache: bool
    raw_path: Path
    meta_path: Path
    meta: dict = field(default_factory=dict)

    def text(self, encoding: str = "utf-8") -> str:
        return self.content.decode(encoding, errors="replace")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _meta_path_for(raw_path: Path) -> Path:
    return raw_path.with_suffix(raw_path.suffix + ".meta.json")


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=20),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout, FetchError)),
)
def _do_request(
    url: str,
    method: str = "GET",
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> requests.Response:
    merged_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        merged_headers.update(headers)
    resp = requests.request(method, url, params=params, headers=merged_headers, timeout=timeout)
    if resp.status_code >= 500 or resp.status_code == 429:
        raise FetchError(f"Retryable HTTP {resp.status_code} from {url}")
    return resp


def fetch(
    url: str,
    cache_key: str,
    subdir: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    method: str = "GET",
    ext: str = "bin",
    force_refresh: bool = False,
    max_age_seconds: Optional[float] = None,
    expect_status: int = 200,
) -> CachedResponse:
    """Fetch ``url`` (GET/POST) with disk caching + provenance metadata.

    Raw bytes are stored at ``data/raw/<subdir>/<cache_key>.<ext>``. A sidecar
    ``.meta.json`` records download timestamp, URL, params, HTTP status,
    byte count, and sha256 so retrieval is auditable.
    """
    out_dir = RAW_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{cache_key}.{ext}"
    meta_path = _meta_path_for(raw_path)

    if not force_refresh and raw_path.exists() and meta_path.exists():
        if max_age_seconds is None or (time.time() - raw_path.stat().st_mtime) < max_age_seconds:
            meta = json.loads(meta_path.read_text())
            logger.debug("cache hit: %s", raw_path)
            return CachedResponse(
                content=raw_path.read_bytes(),
                from_cache=True,
                raw_path=raw_path,
                meta_path=meta_path,
                meta=meta,
            )

    logger.info("fetching %s -> %s", url, raw_path)
    try:
        resp = _do_request(url, method=method, params=params, headers=headers)
    except (requests.ConnectionError, requests.Timeout, FetchError) as exc:
        raise FetchError(f"Failed to fetch {url}: {exc}") from exc

    if resp.status_code != expect_status:
        raise FetchError(
            f"Unexpected HTTP status {resp.status_code} (expected {expect_status}) for {url}. "
            f"Body preview: {resp.text[:300]!r}"
        )
    content = resp.content
    if len(content) == 0:
        raise FetchError(f"Empty response body from {url}")

    raw_path.write_bytes(content)
    meta = {
        "source_url": resp.url,
        "requested_url": url,
        "params": params or {},
        "method": method,
        "http_status": resp.status_code,
        "downloaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_bytes": len(content),
        "sha256": _sha256(content),
        "cache_key": cache_key,
        "subdir": subdir,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return CachedResponse(content=content, from_cache=False, raw_path=raw_path, meta_path=meta_path, meta=meta)


def fetch_json(
    url: str,
    cache_key: str,
    subdir: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    force_refresh: bool = False,
    max_age_seconds: Optional[float] = None,
) -> tuple[Any, CachedResponse]:
    resp = fetch(
        url,
        cache_key=cache_key,
        subdir=subdir,
        params=params,
        headers=headers,
        ext="json",
        force_refresh=force_refresh,
        max_age_seconds=max_age_seconds,
    )
    try:
        return json.loads(resp.content), resp
    except json.JSONDecodeError as exc:
        raise FetchError(f"Response from {url} was not valid JSON: {exc}") from exc
