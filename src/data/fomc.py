"""FOMC meeting calendar, minutes, and statements scraper.

Two page templates, both discovered by fetching the live Fed site for
this project (not assumed from memory):

  - 2021-present: a single combined calendar page
    (monetarypolicy/fomccalendars.htm) covering the current ~6-year
    window, with one ``div.row.fomc-meeting`` block per meeting nested
    under a ``div.panel`` per year.
  - 1996-2020: one page per year
    (monetarypolicy/fomchistorical{YEAR}.htm; pages exist back to 1936
    but only carry parseable Minutes/Statement links from 1996 on), with
    one ``h5`` heading ("January 27-28 Meeting - 2015") per meeting
    followed by a sibling content block. Both the heading markup and the
    statement/minutes URL scheme changed multiple times across this
    range (see the regex alternations below), each verified against the
    live archive for this project.

Both templates state the EXACT minutes release date in text (e.g.
"Minutes (Released February 18, 2015)"), so -- unlike several of the
survey sources in this project -- no release-date reconstruction is
needed for minutes. The meeting/decision date is read directly from the
YYYYMMDD embedded in each meeting's statement/minutes URL (the
Fed's own canonical identifier for "the day the decision was made"),
which is more robust than parsing the human-readable date-range text;
that text is still parsed too, to populate meeting_start_date /
meeting_end_date for descriptive completeness.

CRITICAL TIMING DISTINCTION (spec section 1): every minutes row carries
BOTH ``policy_decision_date`` (meeting end date -- use this to model the
Committee's decision-time information set) and ``minutes_release_date``
(3ish weeks later -- use this when modeling the public communication
event the minutes release itself constitutes). Never conflate the two.
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
CALENDAR_URL = f"{BASE}/monetarypolicy/fomccalendars.htm"
CALENDAR_MIN_YEAR = 2021  # first year covered by the combined calendar page
# 1993-1995 historical pages exist but contain no parseable Minutes/Statement
# links in this format (the modern "Minutes of the FOMC" HTML document and
# routine post-meeting statements only became consistently web-published
# starting in 1996); verified empirically against the live site for this
# project rather than assumed. Statements themselves are further sparse
# 1996-2002 (the FOMC did not yet issue a statement after every meeting).
HISTORICAL_MIN_YEAR = 1996

# The Fed has used three different statement URL schemes and two
# different minutes URL schemes since 2003; alternation covers all of
# them so both the modern calendar page and every historical year page
# parse with the same logic.
_STATEMENT_HREF_RE = re.compile(
    r'href="('
    r'[^"]*/newsevents/pressreleases/monetary(\d{8})a\.htm'  # 2011-present
    r'|[^"]*/newsevents/press/monetary/(\d{8})a\.htm'  # 2006-2010
    r'|[^"]*/boarddocs/press/monetary/\d{4}/(\d{8})/default\.htm'  # 2003-2005
    r')"'
)
_MINUTES_HREF_RE = re.compile(
    r'href="('
    r'[^"]*/monetarypolicy/fomcminutes(\d{8})\.htm'  # 2007-present
    r'|[^"]*/fomc/minutes/(\d{8})\.htm'  # 2003-2006
    r')"'
)
_MINUTES_RELEASED_RE = re.compile(r"Minutes[^(]*\(Released\s+([A-Za-z]+ \d{1,2},\s*\d{4})\)")
_RATE_RANGE_RE = re.compile(
    r"target range for the federal funds rate[^.]*?(\d{1,2}(?:[\.\-][\d/]+)?\s*(?:to|and)\s*\d{1,2}(?:[\.\-][\d/]+)?\s*percent)",
    re.IGNORECASE,
)


def _parse_meeting_dates(month_field: str, day_range_field: str, year: int) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    month_field = month_field.strip()
    day_range_field = re.sub(r"\*", "", day_range_field).strip()
    if "(" in day_range_field:  # e.g. "22 (notation vote)"
        day_range_field = day_range_field.split("(")[0].strip()
    if not day_range_field:
        return None, None

    m1, m2 = (month_field.split("/", 1) + [month_field])[:2] if "/" in month_field else (month_field, month_field)
    if "-" in day_range_field:
        d1, d2 = day_range_field.split("-", 1)
    else:
        d1 = d2 = day_range_field
    d1, d2 = d1.strip(), d2.strip()
    if not (d1.isdigit() and d2.isdigit()):
        return None, None
    try:
        start = pd.to_datetime(f"{m1} {d1}, {year}")
        end_year = year + 1 if (m1.lower().startswith("dec") and m2.lower().startswith("jan")) else year
        end = pd.to_datetime(f"{m2} {d2}, {end_year}")
    except (ValueError, TypeError):
        return None, None
    return start, end


def _first_url(match: Optional[re.Match]) -> Optional[str]:
    if match is None:
        return None
    url = match.group(1)
    return url if url.startswith("http") else BASE + url


def _first_date_group(match: Optional[re.Match]) -> Optional[str]:
    if match is None:
        return None
    return next((g for g in match.groups()[1:] if g), None)


def _extract_meeting_links(block_html: str) -> dict:
    stmt_m = _STATEMENT_HREF_RE.search(block_html)
    minutes_m = _MINUTES_HREF_RE.search(block_html)
    released_m = _MINUTES_RELEASED_RE.search(block_html)
    decision_date = _first_date_group(stmt_m) or _first_date_group(minutes_m)
    return {
        "statement_url": _first_url(stmt_m),
        "minutes_url": _first_url(minutes_m),
        "decision_yyyymmdd": decision_date,
        "minutes_release_date": pd.to_datetime(released_m.group(1)) if released_m else pd.NaT,
    }


def _parse_calendar_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    meetings = []
    for panel in soup.select("div.panel.panel-default"):
        heading = panel.select_one("h4")
        if heading is None:
            continue
        year_m = re.search(r"(\d{4})\s+FOMC Meetings", heading.get_text())
        if not year_m:
            continue
        year = int(year_m.group(1))
        for block in panel.select("div.row.fomc-meeting"):
            month_div = block.select_one("div.fomc-meeting__month")
            date_div = block.select_one("div.fomc-meeting__date")
            if month_div is None or date_div is None:
                continue
            date_text = date_div.get_text(strip=True)
            if "notation vote" in date_text.lower():
                continue
            links = _extract_meeting_links(str(block))
            if not links["decision_yyyymmdd"]:
                continue
            start, end = _parse_meeting_dates(month_div.get_text(strip=True), date_text, year)
            meetings.append(
                {
                    "meeting_start_date": start,
                    "meeting_end_date": end,
                    "policy_decision_date": pd.to_datetime(links["decision_yyyymmdd"], format="%Y%m%d"),
                    "minutes_release_date": links["minutes_release_date"],
                    "minutes_url": links["minutes_url"],
                    "statement_url": links["statement_url"],
                }
            )
    return meetings


def _parse_historical_page(html: str, year: int) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    meetings = []
    header_re = re.compile(
        r"^([A-Za-z]+)\s+(\d{1,2})(?:\s*-\s*(?:([A-Za-z]+)\s+)?(\d{1,2}))?\s+(?:Meeting|Conference Call)\s*-\s*(\d{4})$"
    )
    for heading in soup.select("h5"):
        text = re.sub(r"\s+", " ", heading.get_text(strip=True))
        m = header_re.match(text)
        if not m:
            continue
        month1, day1, month2, day2, hdr_year = m.groups()
        month2 = month2 or month1
        day2 = day2 or day1
        start, end = _parse_meeting_dates(f"{month1}/{month2}" if month2 != month1 else month1, f"{day1}-{day2}", int(hdr_year))

        panel = heading.find_parent("div", class_="panel-default") or heading.parent
        links = _extract_meeting_links(str(panel))
        if not links["decision_yyyymmdd"]:
            continue
        meetings.append(
            {
                "meeting_start_date": start,
                "meeting_end_date": end,
                "policy_decision_date": pd.to_datetime(links["decision_yyyymmdd"], format="%Y%m%d"),
                "minutes_release_date": links["minutes_release_date"],
                "minutes_url": links["minutes_url"],
                "statement_url": links["statement_url"],
            }
        )
    return meetings


def get_fomc_meetings(start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch the FOMC meeting calendar (dates + minutes/statement URLs).

    Returns columns: meeting_start_date, meeting_end_date,
    policy_decision_date, minutes_release_date, minutes_url, statement_url.
    """
    all_meetings: list[dict] = []

    calendar_needed = end_year >= CALENDAR_MIN_YEAR
    if calendar_needed:
        resp = fetch(
            CALENDAR_URL,
            cache_key="calendar",
            subdir="fomc",
            headers=BROWSER_HEADERS,
            ext="html",
            force_refresh=force_refresh,
            max_age_seconds=24 * 3600,
        )
        calendar_meetings = _parse_calendar_page(resp.text())
        if not calendar_meetings:
            raise FetchError("Parsed zero meetings from FOMC calendar page; structure may have changed.")
        all_meetings.extend(
            m for m in calendar_meetings if start_year <= m["policy_decision_date"].year <= end_year
        )

    for year in range(max(start_year, HISTORICAL_MIN_YEAR), min(end_year, CALENDAR_MIN_YEAR - 1) + 1):
        url = f"{BASE}/monetarypolicy/fomchistorical{year}.htm"
        try:
            resp = fetch(
                url,
                cache_key=f"historical_{year}",
                subdir="fomc",
                headers=BROWSER_HEADERS,
                ext="html",
                force_refresh=force_refresh,
                max_age_seconds=30 * 24 * 3600,
            )
        except FetchError:
            logger.warning("No FOMC historical page for %d; skipping.", year)
            continue
        year_meetings = _parse_historical_page(resp.text(), year)
        if not year_meetings:
            logger.warning("Parsed zero meetings from FOMC historical page for %d.", year)
        all_meetings.extend(year_meetings)

    if not all_meetings:
        raise FetchError(f"No FOMC meetings found for {start_year}-{end_year}.")
    df = pd.DataFrame(all_meetings).drop_duplicates(subset="policy_decision_date").sort_values("policy_decision_date")
    return df.reset_index(drop=True)


def _parse_article_body(html: str, body_selector: str) -> tuple[list[str], list[str]]:
    soup = BeautifulSoup(html, "lxml")
    article = soup.select_one("div#article")
    if article is None:
        raise FetchError("Could not find div#article.")
    footnotes_container = article.select_one("div.footnotes")
    footnotes: list[str] = []
    if footnotes_container is not None:
        for p in footnotes_container.find_all("p"):
            for sup in p.select("sup"):
                sup.decompose()
            text = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
            if text:
                footnotes.append(text)
        footnotes_container.decompose()

    # For the shared Fed Board template, both the heading block and the
    # body block carry "col-xs-12 col-sm-8 col-md-8"; the heading block
    # additionally carries "heading" and appears first in the DOM, so a
    # plain select_one() would wrongly grab the heading. Prefer the first
    # match that does NOT have the "heading" class.
    candidates = article.select(body_selector)
    non_heading = [c for c in candidates if "heading" not in (c.get("class") or [])]
    body_container = (non_heading[0] if non_heading else (candidates[0] if candidates else article))
    paragraphs = []
    for p in body_container.find_all("p", recursive=True):
        if p.find_parent("div", class_="footnotes") is not None:
            continue
        for sup in p.select("sup"):
            sup.decompose()
        text = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        if text:
            paragraphs.append(text)
    return paragraphs, footnotes


def _parse_legacy_minutes_body(html: str) -> tuple[list[str], list[str]]:
    """Parser for the pre-2007 ``/fomc/minutes/{yyyymmdd}.htm`` template:
    an old table-based layout with no ``div#article``. The attendee list
    and the substantive minutes both live in ``<td width="600">``
    elements; empirically (verified against a 2003 sample for this
    project) the attendee list is the first such element and the real
    minutes body is the last.
    """
    soup = BeautifulSoup(html, "lxml")
    td_blocks = soup.find_all("td", attrs={"width": "600"})
    if len(td_blocks) < 2:
        raise FetchError("Expected >=2 td[width=600] blocks on legacy minutes page.")
    body_td = td_blocks[-1]

    paragraphs = []
    for p in body_td.find_all("p", recursive=True):
        for sup in p.select("sup"):
            sup.decompose()
        text = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        if text:
            paragraphs.append(text)

    footnotes: list[str] = []
    marker = soup.find(lambda t: t.name in ("b", "strong") and t.get_text(strip=True) == "Footnotes")
    if marker is not None:
        container = marker.find_parent("table") or marker.parent
        if container is not None:
            text = re.sub(r"\s+", " ", container.get_text(" ", strip=True)).strip()
            if text:
                footnotes.append(text)
    return paragraphs, footnotes


def get_minutes(start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch full-text FOMC minutes for meetings decided in [start_year, end_year].

    Returns one row per meeting: document_id, document_type='fomc_minutes',
    event_date=policy_decision_date, publication_date=minutes_release_date,
    meeting_date=policy_decision_date, meeting_start_date, meeting_end_date,
    speaker=None, speaker_role='FOMC', title, source_url, raw_text, clean_text,
    footnotes_text.
    """
    meetings = get_fomc_meetings(start_year, end_year, force_refresh=force_refresh)
    rows = []
    for _, m in meetings.iterrows():
        if pd.isna(m["minutes_url"]) or pd.isna(m["minutes_release_date"]):
            continue
        try:
            resp = fetch(
                m["minutes_url"],
                cache_key=f"minutes_{m['policy_decision_date'].strftime('%Y%m%d')}",
                subdir="fomc",
                headers=BROWSER_HEADERS,
                ext="html",
                force_refresh=force_refresh,
                max_age_seconds=365 * 24 * 3600,
            )
            if "/fomc/minutes/" in m["minutes_url"]:
                paragraphs, footnotes = _parse_legacy_minutes_body(resp.text())
            else:
                paragraphs, footnotes = _parse_article_body(resp.text(), "div#article")
        except Exception:
            logger.exception("Failed to fetch/parse minutes at %s", m["minutes_url"])
            continue
        yyyymmdd = m["policy_decision_date"].strftime("%Y%m%d")
        rows.append(
            {
                "document_id": f"fomc_minutes_{yyyymmdd}",
                "document_type": "fomc_minutes",
                "event_date": m["policy_decision_date"],
                "publication_date": m["minutes_release_date"],
                "meeting_date": m["policy_decision_date"],
                "meeting_start_date": m["meeting_start_date"],
                "meeting_end_date": m["meeting_end_date"],
                "speaker": None,
                "speaker_role": "FOMC",
                "title": f"Minutes of the Federal Open Market Committee, {m['policy_decision_date'].date()}",
                "source_url": m["minutes_url"],
                "raw_text": resp.text(),
                "clean_text": "\n\n".join(paragraphs),
                "footnotes_text": "\n\n".join(footnotes),
            }
        )
    if not rows:
        raise FetchError(f"No FOMC minutes parsed for {start_year}-{end_year}.")
    df = pd.DataFrame(rows).sort_values("event_date").reset_index(drop=True)
    logger.info("FOMC minutes: %d meetings, %s to %s", len(df), df["event_date"].min().date(), df["event_date"].max().date())
    return df


def get_statements(start_year: int, end_year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch full-text FOMC post-meeting statements.

    Returns one row per meeting: document_id, document_type='fomc_statement',
    event_date=publication_date=policy_decision_date, meeting_date,
    title, source_url, raw_text, clean_text, footnotes_text, rate_decision_text.
    """
    meetings = get_fomc_meetings(start_year, end_year, force_refresh=force_refresh)
    rows = []
    for _, m in meetings.iterrows():
        if pd.isna(m["statement_url"]):
            continue
        try:
            resp = fetch(
                m["statement_url"],
                cache_key=f"statement_{m['policy_decision_date'].strftime('%Y%m%d')}",
                subdir="fomc",
                headers=BROWSER_HEADERS,
                ext="html",
                force_refresh=force_refresh,
                max_age_seconds=365 * 24 * 3600,
            )
            paragraphs, footnotes = _parse_article_body(resp.text(), "div.col-xs-12.col-sm-8.col-md-8")
        except Exception:
            logger.exception("Failed to fetch/parse statement at %s", m["statement_url"])
            continue
        clean_text = "\n\n".join(paragraphs)
        rate_m = _RATE_RANGE_RE.search(clean_text)
        yyyymmdd = m["policy_decision_date"].strftime("%Y%m%d")
        rows.append(
            {
                "document_id": f"fomc_statement_{yyyymmdd}",
                "document_type": "fomc_statement",
                "event_date": m["policy_decision_date"],
                "publication_date": m["policy_decision_date"],
                "meeting_date": m["policy_decision_date"],
                "meeting_start_date": m["meeting_start_date"],
                "meeting_end_date": m["meeting_end_date"],
                "speaker": None,
                "speaker_role": "FOMC",
                "title": f"FOMC Statement, {m['policy_decision_date'].date()}",
                "source_url": m["statement_url"],
                "raw_text": resp.text(),
                "clean_text": clean_text,
                "footnotes_text": "\n\n".join(footnotes),
                "rate_decision_text": rate_m.group(1) if rate_m else None,
            }
        )
    if not rows:
        raise FetchError(f"No FOMC statements parsed for {start_year}-{end_year}.")
    df = pd.DataFrame(rows).sort_values("event_date").reset_index(drop=True)
    logger.info("FOMC statements: %d meetings, %s to %s", len(df), df["event_date"].min().date(), df["event_date"].max().date())
    return df
