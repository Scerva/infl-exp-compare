"""Federal Reserve Board of Governors speech archive scraper.

COVERAGE WARNING (spec section 4A): this archive principally covers Board
Governors (Chair, Vice Chairs, Governors). It is NOT a comprehensive
archive of the 12 regional Reserve Bank presidents' speeches, who publish
independently on their own Bank websites. This is a real selection issue
for any claim about "the Fed's" communication as a whole -- document it,
do not paper over it. The architecture here (one adapter module per
source) is meant to let bank-specific adapters be added later without
touching this module.

TWO ARCHIVE ERAS, TWO PAGE TEMPLATES:
  - 2011-present: "https://www.federalreserve.gov/newsevents/speech/
    {year}-speeches.htm" index pages (Bootstrap-era template), individual
    speech pages at "/newsevents/speech/{speaker}{YYYYMMDD}{seq}.htm".
  - 1996-2010: index pages at "/newsevents/speech/{year}speech.htm" (a
    later-generation replacement for the page that redirects from the
    original "/boarddocs/speeches/{year}/" URL), individual speech pages
    at "/boarddocs/speeches/{year}/{YYYYMMDD}[seq]/default.htm" (a much
    older, table-based, largely unclosed-tag HTML template).
Both were discovered by fetching the live pages for this project, not
assumed from memory -- see the module-level tests for the concrete
structural assumptions each parser relies on.

Every downloaded page is cached to disk (raw HTML) with provenance
metadata via src.utils.cache; body text is split into pre-footnote
paragraphs (used for stance scoring) and footnotes (retained separately,
not fed to the stance model, since footnotes are citations/methodological
asides rather than the speaker's policy language).
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

from src.utils.cache import FetchError, fetch
from src.utils.logging import get_logger

logger = get_logger(__name__)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

BASE = "https://www.federalreserve.gov"
MODERN_MIN_YEAR = 2011
LEGACY_MIN_YEAR = 1996

# Longest-match-first so "Vice Chair for Supervision" isn't cut short by "Vice Chair".
ROLE_PATTERNS = [
    "Vice Chair for Supervision",
    "Vice Chairman for Supervision",
    "Vice Chairwoman",
    "Vice Chairman",
    "Vice Chair",
    "Chairman",
    "Chairwoman",
    "Chair",
    "Governor",
]
_ROLE_RE = re.compile(r"^(" + "|".join(re.escape(r) for r in ROLE_PATTERNS) + r")\b\s*", re.IGNORECASE)


def _split_role_name(speaker_raw: str) -> tuple[Optional[str], str]:
    speaker_raw = re.sub(r"\s+", " ", speaker_raw).strip()
    m = _ROLE_RE.match(speaker_raw)
    if m:
        role = m.group(1)
        name = speaker_raw[m.end():].strip()
        return role, name
    return None, speaker_raw


def _index_cache_key(year: int) -> str:
    return f"index_{year}"


def _fetch_index_html(year: int, force_refresh: bool = False) -> tuple[str, str]:
    """Return (html, era) where era is 'modern' or 'legacy'."""
    if year >= MODERN_MIN_YEAR:
        url = f"{BASE}/newsevents/speech/{year}-speeches.htm"
        era = "modern"
    else:
        url = f"{BASE}/newsevents/speech/{year}speech.htm"
        era = "legacy"
    resp = fetch(
        url,
        cache_key=_index_cache_key(year),
        subdir="fed_speeches",
        headers=BROWSER_HEADERS,
        ext="html",
        force_refresh=force_refresh,
        max_age_seconds=24 * 3600,
    )
    return resp.text(), era


def _parse_modern_index(html: str, year: int) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    records = []
    for row in soup.select("div.row"):
        time_tag = row.select_one("time")
        title_link = row.select_one("div.eventlist__event p a") or row.select_one("p a")
        if time_tag is None or title_link is None:
            continue
        href = title_link.get("href", "")
        if "/newsevents/speech/" not in href or href.endswith("-speeches.htm"):
            continue
        speaker_tag = row.select_one("p.news__speaker")
        location_tags = row.select("div.eventlist__event p")
        location = None
        for p in location_tags:
            if p.get("class") and "news__speaker" in p.get("class"):
                continue
            if p.find("a") is not None:
                continue
            text = p.get_text(strip=True)
            if text:
                location = text
                break
        date_str = time_tag.get_text(strip=True)
        try:
            event_date = pd.to_datetime(date_str)
        except (ValueError, TypeError):
            continue
        url = href if href.startswith("http") else BASE + href
        document_id = href.rsplit("/", 1)[-1].replace(".htm", "")
        records.append(
            {
                "document_id": document_id,
                "event_date": event_date,
                "title": title_link.get_text(strip=True),
                "source_url": url,
                "speaker_raw": speaker_tag.get_text(strip=True) if speaker_tag else "",
                "venue": location,
            }
        )
    if not records:
        raise FetchError(f"Parsed zero speech records from modern index for {year}; page structure may have changed.")
    return records


def _parse_legacy_index(html: str, year: int) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    records = []
    for li in soup.select("ul#speechIndex li"):
        title_link = li.select_one("div.title a")
        if title_link is None:
            continue
        href = title_link.get("href", "")
        date_text = li.get_text(" ", strip=True)
        date_match = re.match(r"^([A-Za-z]+ \d{1,2}, \d{4})", date_text)
        if not date_match:
            continue
        try:
            event_date = pd.to_datetime(date_match.group(1))
        except (ValueError, TypeError):
            continue
        speaker_tag = li.select_one("div.speaker")
        location_tag = li.select_one("div.location")
        url = href if href.startswith("http") else BASE + href
        # document id from the /boarddocs/speeches/{year}/{seq}/default.htm folder segment
        seq_match = re.search(r"/speeches/\d{4}/([^/]+)/", href)
        document_id = f"legacy_{seq_match.group(1)}" if seq_match else f"legacy_{year}_{len(records)}"
        records.append(
            {
                "document_id": document_id,
                "event_date": event_date,
                "title": title_link.get_text(strip=True),
                "source_url": url,
                "speaker_raw": speaker_tag.get_text(strip=True) if speaker_tag else "",
                "venue": location_tag.get_text(strip=True) if location_tag else None,
            }
        )
    if not records:
        raise FetchError(f"Parsed zero speech records from legacy index for {year}; page structure may have changed.")
    return records


def get_speech_index(year: int, force_refresh: bool = False) -> list[dict]:
    if year < LEGACY_MIN_YEAR:
        raise ValueError(f"No Fed Board speech archive before {LEGACY_MIN_YEAR}.")
    html, era = _fetch_index_html(year, force_refresh=force_refresh)
    return _parse_modern_index(html, year) if era == "modern" else _parse_legacy_index(html, year)


def _clean_paragraph_text(tag) -> str:
    for sup in tag.select("sup"):
        sup.decompose()
    for anchor in tag.select('a[name]'):
        if not anchor.get_text(strip=True):
            anchor.decompose()
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()


def _parse_modern_speech_body(html: str) -> tuple[list[str], list[str]]:
    soup = BeautifulSoup(html, "lxml")
    article = soup.select_one("div#article")
    if article is None:
        raise FetchError("Could not find div#article on speech page.")
    body_divs = [
        d for d in article.select("div.col-xs-12.col-sm-8.col-md-8") if "heading" not in (d.get("class") or [])
    ]
    if not body_divs:
        raise FetchError("Could not find speech body container on modern speech page.")
    body_div = body_divs[0]

    hr = body_div.find("hr")
    paragraphs, footnotes = [], []
    in_footnotes = False
    for el in body_div.find_all(["p", "hr"], recursive=True):
        if el.name == "hr":
            in_footnotes = True
            continue
        text = _clean_paragraph_text(el)
        if not text:
            continue
        (footnotes if in_footnotes else paragraphs).append(text)
    return paragraphs, footnotes


def _parse_legacy_speech_body(html: str) -> tuple[list[str], list[str]]:
    soup = BeautifulSoup(html, "lxml")
    tables = soup.select('table[width="600"]')
    if len(tables) < 2:
        raise FetchError("Expected at least 2 width=600 tables on legacy speech page.")
    body_td = tables[1].select_one("td")
    if body_td is None:
        raise FetchError("Could not find body <td> on legacy speech page.")

    hr = body_td.find("hr")
    paragraphs, footnotes = [], []
    in_footnotes = False
    for el in body_td.find_all(["p", "hr"], recursive=True):
        if el.name == "hr":
            in_footnotes = True
            continue
        text = _clean_paragraph_text(el)
        if not text or text.lower() == "footnotes":
            continue
        (footnotes if in_footnotes else paragraphs).append(text)
    return paragraphs, footnotes


def get_speeches(start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch every Fed Board speech from start_year through end_year (inclusive).

    Returns one row per speech with document_id, event_date,
    publication_date (== event_date for speeches), speaker, speaker_role,
    document_type='speech', title, venue, source_url, raw_text,
    clean_text (pre-footnote body), footnotes_text.
    """
    rows = []
    for year in range(start_year, end_year + 1):
        try:
            index = get_speech_index(year, force_refresh=force_refresh)
        except FetchError:
            logger.warning("No speech index available for %d; skipping.", year)
            continue
        logger.info("%d: %d speeches in index", year, len(index))
        for rec in index:
            try:
                resp = fetch(
                    rec["source_url"],
                    cache_key=rec["document_id"],
                    subdir="fed_speeches",
                    headers=BROWSER_HEADERS,
                    ext="html",
                    force_refresh=force_refresh,
                    max_age_seconds=365 * 24 * 3600,
                )
                html = resp.text()
                if year >= MODERN_MIN_YEAR:
                    paragraphs, footnotes = _parse_modern_speech_body(html)
                else:
                    paragraphs, footnotes = _parse_legacy_speech_body(html)
            except Exception:
                logger.exception("Failed to fetch/parse speech %s (%s)", rec["document_id"], rec["source_url"])
                continue

            role, name = _split_role_name(rec["speaker_raw"])
            rows.append(
                {
                    "document_id": rec["document_id"],
                    "document_type": "speech",
                    "event_date": rec["event_date"],
                    "publication_date": rec["event_date"],
                    "meeting_date": pd.NaT,
                    "speaker": name,
                    "speaker_role": role,
                    "title": rec["title"],
                    "venue": rec["venue"],
                    "source_url": rec["source_url"],
                    "raw_text": html,
                    "clean_text": "\n\n".join(paragraphs),
                    "footnotes_text": "\n\n".join(footnotes),
                }
            )

    if not rows:
        raise FetchError(f"Parsed zero speeches for range {start_year}-{end_year}.")
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="document_id").sort_values("event_date").reset_index(drop=True)
    logger.info("Fed speeches: %d total, %s to %s", len(df), df["event_date"].min().date(), df["event_date"].max().date())
    return df
