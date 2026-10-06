"""
@file imd_public_client.py
@description IMD Agromet District Bulletin scraper using public imdagrimet.gov.in endpoints.
             No API token required — uses standard desktop browser User-Agent headers.

             Three data sources:
               1. Agromet advisory bulletin from imdagrimet.gov.in (HTML scrape via BS4)
               2. District nowcast from mausam.imd.gov.in (JSON API)
               3. Current station observations from mausam.imd.gov.in (JSON API)

             All functions return RawBulletin instances for downstream LLM extraction.
             A mock fallback is always available when government portals are unreachable.

@module services/ingestion
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from services.api.schemas import RawBulletin

# ─────────────────────────────────────────────────────────────────────────────
#  Endpoint constants
# ─────────────────────────────────────────────────────────────────────────────

AGROMET_BULLETIN_URL: str = (
    "https://imdagrimet.gov.in/Services/DistrictBulletin.php"
)
IMD_NOWCAST_URL: str = "https://mausam.imd.gov.in/api/nowcast_district_api.php"
IMD_CURRENT_WX_URL: str = "https://mausam.imd.gov.in/api/current_wx_api.php"

# Realistic desktop browser headers — IMD Nginx drops connections without them
_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
    "Referer": "https://imdagrimet.gov.in/",
    "Connection": "keep-alive",
}

_JSON_HEADERS: dict[str, str] = {
    **_BROWSER_HEADERS,
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://mausam.imd.gov.in/",
}

# Generous timeout for the notoriously slow IMD portal (connect, read)
_IMD_TIMEOUT: tuple[float, float] = (10.0, 60.0)

_RETRY_POLICY = dict(
    retry=retry_if_exception_type(
        (requests.Timeout, requests.ConnectionError, requests.HTTPError)
    ),
    wait=wait_exponential(multiplier=2, min=3, max=60),
    stop=stop_after_attempt(3),
    reraise=False,  # NOTE: Fall through to mock on total failure
)

# ─────────────────────────────────────────────────────────────────────────────
#  Mock fixtures (used when IMD portals are unreachable)
#  NOTE: These fixtures are strictly SYNTHETIC / MOCK test data for offline
#  resilience and demo simulation. They must NOT be mistaken for real IMD feeds.
# ─────────────────────────────────────────────────────────────────────────────

_MOCK_AGROMET_TEXTS: dict[str, str] = {
    "Jaipur": """
DISTRICT AGROMET ADVISORY BULLETIN
Issued by: IMD Agromet Division | Valid: 04-09-2026 to 08-09-2026

WEATHER FORECAST SUMMARY (Jaipur, Rajasthan):
Partly cloudy to cloudy sky is expected during next 5 days. Light to moderate
rainfall is likely on 05-09-2026 and 06-09-2026. Maximum temperature will
remain between 34–36°C and minimum temperature between 24–26°C.
Wind speed 10–20 km/h from SW direction.

CROP ADVISORY — Kharif 2026:
1. Soybean (Vegetative stage): Spray of recommended fungicide advised if leaf
   yellowing is observed due to excess moisture. Ensure proper drainage.
2. Bajra (Flowering): Avoid irrigation during flowering if rainfall >= 15 mm.
   Apply 2 kg/ha potassium nitrate for grain setting.
3. Cluster Bean / Guar (Vegetative): Provide drainage channels to avoid
   waterlogging in standing crop during rainy spells.
4. Maize (Tasseling): Spray of 0.5% ZnSO4 for micronutrient deficiency.
   Ensure proper spacing for pollination.
5. Cotton (Boll Formation): Monitor for bollworm infestation; use pheromone
   traps. Avoid pesticide spray on rainy days.

WARNING: No severe weather warning issued for Jaipur district.
ADVISORY: [SYNTHETIC / MOCK FIXTURE — imdagrimet.gov.in fallback for Jaipur district]
""".strip(),

    "Alwar": """
DISTRICT AGROMET ADVISORY BULLETIN
Issued by: IMD Agromet Division | Valid: 04-09-2026 to 08-09-2026

WEATHER FORECAST SUMMARY (Alwar, Rajasthan):
Mainly clear to partly cloudy sky is expected during next 5 days. Weather will
remain dry over the district with no rainfall expected. Maximum temperature
will remain between 36–38°C and minimum temperature between 25–27°C.
Wind speed 8–14 km/h from NW direction.

CROP ADVISORY — Kharif/Rabi Transition 2026:
1. Mustard (Sowing / Early Vegetative): Conserve residual soil moisture by field
   bunding and shallow hoeing. Ensure certified seed treatment before sowing.
2. Groundnut (Pod Formation): Maintain adequate soil moisture through light
   irrigation if dry spell persists; avoid extreme moisture stress during pod filling.
3. Pearl Millet / Bajra (Grain Filling): Undertake intercultural operations for
   moisture conservation if soil crusting occurs.
4. Cluster Bean / Guar (Pod Formation): Monitor for aphid infestation in warm,
   dry weather; spray neem-based formulations if threshold crossed.

WARNING: No severe weather warning issued for Alwar district.
ADVISORY: [SYNTHETIC / MOCK FIXTURE — imdagrimet.gov.in fallback for Alwar district]
""".strip(),
}

# Backward-compatible alias for default mock text (Jaipur)
_MOCK_AGROMET_TEXT: str = _MOCK_AGROMET_TEXTS["Jaipur"]


def _get_mock_agromet_text(district: str = "Jaipur") -> str:
    """
    Retrieve district-specific synthetic mock Agromet bulletin.

    NOTE: Strictly SYNTHETIC / MOCK fixture for offline fallback when
    imdagrimet.gov.in is unreachable. Real live fetching per district is
    the production target.
    """
    norm_dist = district.strip().title() if district else "Jaipur"
    return _MOCK_AGROMET_TEXTS.get(norm_dist, _MOCK_AGROMET_TEXTS["Jaipur"])


_MOCK_NOWCAST_FIXTURE: dict[str, Any] = {
    "district": "Jaipur",
    "state": "Rajasthan",
    "valid_time": "2026-09-04T18:00:00",
    "issued_at": "2026-09-04T15:00:00",
    "weather_description": "Partly Cloudy with possibility of light showers",
    "max_temp_c": 34.2,
    "min_temp_c": 25.8,
    "rainfall_probability_pct": 40,
    "wind_direction": "SW",
    "wind_speed_kmh": 18,
    "relative_humidity_pct": 72,
    "warnings": ["No warning issued"],
    "advisory": "Farmers advised to monitor soil moisture before irrigation.",
    "source": "IMD_NOWCAST_MOCK",
    "note": "MOCK DATA — IMD endpoint unreachable during generation",
}

_MOCK_CURRENT_WX_FIXTURE: dict[str, Any] = {
    "station_id": "42001",
    "station_name": "Jaipur Airport",
    "observed_at": "2026-09-04T17:30:00Z",
    "temperature_c": 30.5,
    "dew_point_c": 22.1,
    "relative_humidity_pct": 65,
    "sea_level_pressure_hpa": 998.2,
    "wind_direction_deg": 225,
    "wind_speed_kmh": 14,
    "visibility_km": 8.0,
    "weather_condition": "Scattered Clouds",
    "source": "IMD_CURRENT_WX_MOCK",
    "note": "MOCK DATA — IMD endpoint unreachable during generation",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


@retry(**_RETRY_POLICY)
def _get_html(url: str, params: dict[str, str]) -> str | None:
    """
    Perform a browser-mimicking GET request and return the raw HTML string.
    Returns None on any failure so callers can fall through to mock.
    """
    logger.debug(f"[imd_public] GET {url} params={params}")
    try:
        resp = requests.get(
            url,
            params=params,
            headers=_BROWSER_HEADERS,
            timeout=_IMD_TIMEOUT,
            allow_redirects=True,
        )
        resp.raise_for_status()
        logger.info(f"[imd_public] HTML fetched from {url} ({len(resp.text)} chars)")
        return resp.text
    except (requests.Timeout, requests.ConnectionError) as exc:
        logger.warning(f"[imd_public] Network error: {exc}")
        raise  # Trigger tenacity retry
    except (requests.HTTPError, UnicodeDecodeError) as exc:
        logger.error(f"[imd_public] Non-retryable error: {exc}")
        return None


@retry(**_RETRY_POLICY)
def _get_json(url: str, params: dict[str, str]) -> dict[str, Any] | None:
    """
    Perform a browser-mimicking GET request and return parsed JSON dict.
    Handles the IMD anti-pattern of returning HTML 200 on error pages.
    """
    logger.debug(f"[imd_public] GET JSON {url} params={params}")
    try:
        resp = requests.get(
            url,
            params=params,
            headers=_JSON_HEADERS,
            timeout=_IMD_TIMEOUT,
            allow_redirects=True,
        )
        resp.raise_for_status()

        # IMD occasionally serves HTML error pages with status 200
        if "html" in resp.headers.get("Content-Type", "").lower():
            logger.warning("[imd_public] Received HTML instead of JSON — treating as failure")
            return None

        data: dict[str, Any] = resp.json()
        logger.info(f"[imd_public] JSON fetched from {url}")
        return data

    except (requests.Timeout, requests.ConnectionError) as exc:
        logger.warning(f"[imd_public] Network error: {exc}")
        raise  # Trigger tenacity retry
    except (requests.HTTPError, ValueError, json.JSONDecodeError) as exc:
        logger.error(f"[imd_public] Non-retryable error: {exc}")
        return None


def _parse_agromet_html(html: str) -> str:
    """
    Parse imdagrimet.gov.in district bulletin HTML using BeautifulSoup4.

    Extracts:
    - Weather forecast summary paragraph(s)
    - Crop advisory text paragraphs
    - Any warning notices

    Returns the concatenated advisory text as a clean string, or an empty
    string if the page structure is unrecognised.

    Args:
        html: Raw HTML content from imdagrimet.gov.in

    Returns:
        Extracted advisory text (multi-paragraph string).
    """
    soup = BeautifulSoup(html, "lxml")
    extracted_parts: list[str] = []

    # Strategy 1: look for the bulletin content container div
    # The portal wraps advisory text in <div class="bulletin-content"> or similar
    content_candidates = [
        soup.find("div", class_="bulletin-content"),
        soup.find("div", id="bulletin"),
        soup.find("div", class_="advisory"),
        soup.find("div", class_="content"),
        soup.find("div", class_="report"),
        soup.find("table", class_="bulletin"),
    ]

    container = next((c for c in content_candidates if c is not None), None)

    if container:
        # Extract all paragraph text from the container
        paragraphs = container.find_all(["p", "td", "li"])
        for para in paragraphs:
            text = para.get_text(separator=" ", strip=True)
            if len(text) > 30:  # Skip trivially short strings
                extracted_parts.append(text)
    else:
        # Fallback: extract all <p> and <td> tags with substantial text
        logger.warning("[imd_public] Known container not found; falling back to generic extraction")
        for tag in soup.find_all(["p", "td"]):
            text = tag.get_text(separator=" ", strip=True)
            # Heuristic: advisory paragraphs are at least 60 chars
            if len(text) >= 60:
                extracted_parts.append(text)

    if not extracted_parts:
        logger.warning("[imd_public] No advisory text could be extracted from HTML")
        return ""

    return "\n\n".join(extracted_parts)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────


def fetch_agromet_bulletin(
    state: str = "Rajasthan",
    district: str = "Jaipur",
    language: str = "English",
    fallback_to_mock: bool = True,
) -> RawBulletin:
    """
    Scrape the IMD Agromet district advisory bulletin from imdagrimet.gov.in.

    Endpoint:
        GET https://imdagrimet.gov.in/Services/DistrictBulletin.php
        ?state={state}&district={district}&language={language}

    Args:
        state:           State name (e.g., "Rajasthan").
        district:        District name (e.g., "Jaipur").
        language:        Language for the bulletin ("English" by default).
        fallback_to_mock: If True, returns realistic mock data when the portal
                          is unreachable. Set to False to raise on failure.

    Returns:
        RawBulletin — validated Pydantic instance with extracted advisory text.

    Raises:
        RuntimeError: If fallback_to_mock=False and the endpoint is unreachable.

    NOTE: The raw_text field contains the extracted advisory paragraphs, not
          the full raw HTML. This keeps it within the 200k char RawBulletin
          limit and avoids feeding boilerplate HTML to the LLM extractor.
    """
    params = {"state": state, "district": district, "language": language}
    html = _get_html(AGROMET_BULLETIN_URL, params=params)

    if html is not None:
        advisory_text = _parse_agromet_html(html)
        if advisory_text:
            source = "IMD_AGROMET_BULLETIN"
            is_mock = False
        else:
            logger.warning("[imd_public] HTML received but extraction yielded empty text — using mock for district '%s'", district)
            advisory_text = _get_mock_agromet_text(district)
            source = "IMD_AGROMET_BULLETIN_MOCK"
            is_mock = True
    else:
        if fallback_to_mock:
            logger.warning("[imd_public] Agromet portal unreachable — using mock fixture for district '%s'", district)
            advisory_text = _get_mock_agromet_text(district)
            source = "IMD_AGROMET_BULLETIN_MOCK"
            is_mock = True
        else:
            raise RuntimeError(
                f"IMD Agromet bulletin unavailable for {state}/{district}"
            )

    bulletin = RawBulletin(
        source=source,
        timestamp=datetime.now(tz=timezone.utc),
        raw_text=advisory_text,
        metadata={
            "state": state,
            "district": district,
            "language": language,
            "endpoint": AGROMET_BULLETIN_URL,
            "is_mock": is_mock,
        },
        source_status="mock" if is_mock else "live",
    )
    logger.debug(f"[imd_public] Agromet bulletin: source={bulletin.source}, chars={len(bulletin.raw_text)}")
    return bulletin


def fetch_imd_nowcast(
    district_id: str,
    fallback_to_mock: bool = True,
) -> RawBulletin:
    """
    Fetch district-level nowcast bulletin from IMD Mausam portal (JSON endpoint).

    Endpoint: GET https://mausam.imd.gov.in/api/nowcast_district_api.php?id={district_id}

    Args:
        district_id:      IMD district code string (e.g., "26" for Jaipur).
        fallback_to_mock: Returns mock data on failure if True.

    Returns:
        RawBulletin validated Pydantic instance.
    """
    raw_data = _get_json(IMD_NOWCAST_URL, params={"id": district_id})

    if raw_data is None:
        if fallback_to_mock:
            logger.warning("[imd_public] Nowcast endpoint unreachable — using mock fixture")
            raw_data = _MOCK_NOWCAST_FIXTURE
            source = "IMD_NOWCAST_MOCK"
            is_mock = True
        else:
            raise RuntimeError(
                f"IMD Nowcast endpoint unreachable for district_id={district_id!r}"
            )
    else:
        source = "IMD_NOWCAST"
        is_mock = False

    return RawBulletin(
        source=source,
        timestamp=datetime.now(tz=timezone.utc),
        raw_text=json.dumps(raw_data, ensure_ascii=False, indent=2),
        metadata={
            "district_id": district_id,
            "endpoint": IMD_NOWCAST_URL,
            "is_mock": is_mock,
        },
        source_status="mock" if is_mock else "live",
    )


def fetch_imd_current_weather(
    station_id: str,
    fallback_to_mock: bool = True,
) -> RawBulletin:
    """
    Fetch current observed weather for an IMD surface station.

    Endpoint: GET https://mausam.imd.gov.in/api/current_wx_api.php?id={station_id}

    Args:
        station_id:       IMD station ID (e.g., "42001" for Jaipur Airport).
        fallback_to_mock: Falls back to fixture data if endpoint is unreachable.

    Returns:
        RawBulletin validated Pydantic instance.
    """
    raw_data = _get_json(IMD_CURRENT_WX_URL, params={"id": station_id})

    if raw_data is None:
        if fallback_to_mock:
            logger.warning("[imd_public] Current WX endpoint unreachable — using mock fixture")
            raw_data = _MOCK_CURRENT_WX_FIXTURE
            source = "IMD_CURRENT_WX_MOCK"
            is_mock = True
        else:
            raise RuntimeError(
                f"IMD Current WX endpoint unreachable for station_id={station_id!r}"
            )
    else:
        source = "IMD_CURRENT_WX"
        is_mock = False

    return RawBulletin(
        source=source,
        timestamp=datetime.now(tz=timezone.utc),
        raw_text=json.dumps(raw_data, ensure_ascii=False, indent=2),
        metadata={
            "station_id": station_id,
            "endpoint": IMD_CURRENT_WX_URL,
            "is_mock": is_mock,
        },
        source_status="mock" if is_mock else "live",
    )


def read_mock_bulletin(district: str = "Jaipur", state: str = "Rajasthan") -> RawBulletin:
    """
    Return a hard-coded synthetic mock Agromet bulletin directly (no HTTP call at all).

    Intended for use in unit tests and offline CI pipelines.

    Args:
        district: District name to select matching mock fixture and embed in metadata.
        state:    State name to embed in metadata.

    Returns:
        RawBulletin with MOCK source tag.
    """
    return RawBulletin(
        source="IMD_AGROMET_MOCK_READER",
        timestamp=datetime.now(tz=timezone.utc),
        raw_text=_get_mock_agromet_text(district),
        metadata={"district": district, "state": state, "is_mock": True},
        source_status="mock",
    )
