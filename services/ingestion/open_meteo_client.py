"""
@file open_meteo_client.py
@description Open-Meteo REST API client for 7-day hourly/daily weather forecasts
             and DEM-based elevation lookup. No API key required (free tier).
             All responses are validated through Pydantic v2 schemas before return.
@module services/ingestion
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Internal schemas — single source of truth for all data shapes
from services.api.schemas import (
    BaselineForecast,
    DailyTemperatureRecord,
    GeoLocation,
    HourlyForecastPoint,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL: str = os.getenv("OPEN_METEO_BASE_URL", "https://api.open-meteo.com/v1")
ELEVATION_URL: str = os.getenv(
    "OPEN_METEO_ELEVATION_URL", "https://api.open-meteo.com/v1/elevation"
)
ARCHIVE_URL: str = os.getenv(
    "OPEN_METEO_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1/archive"
)

# Default request timeout (connect, read) in seconds
DEFAULT_TIMEOUT: tuple[float, float] = (5.0, 30.0)

# Hourly variables to request from Open-Meteo
HOURLY_VARIABLES: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "direct_normal_irradiance",
    "shortwave_radiation_instant",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
    "et0_fao_evapotranspiration",
    # Agromet operations (spray-window optimizer, advisory rules). Not model features.
    "precipitation_probability",
    "dew_point_2m",
    "wind_gusts_10m",
]

# Daily aggregations to request
DAILY_VARIABLES: list[str] = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "weathercode",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Retry decorator — exponential back-off for transient network errors
# ─────────────────────────────────────────────────────────────────────────────

_RETRY_POLICY = dict(
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────


@retry(**_RETRY_POLICY)
def fetch_elevation(lat: float, lon: float) -> float | None:
    """
    Query the Open-Meteo Elevation API for a single (lat, lon) pair.

    Args:
        lat: WGS-84 latitude.
        lon: WGS-84 longitude.

    Returns:
        Elevation in metres above sea level, or None on failure.

    Example:
        >>> elev = fetch_elevation(26.9124, 75.7873)
        >>> print(f"Elevation: {elev} m")
    """
    params = {"latitude": lat, "longitude": lon}
    url = f"{ELEVATION_URL}?{urlencode(params)}"
    logger.debug(f"[elevation] GET {url}")

    try:
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        elevations: list[float] = payload.get("elevation", [None])
        elevation = elevations[0] if elevations else None
        logger.info(f"[elevation] lat={lat}, lon={lon} → {elevation} m")
        return elevation
    except requests.HTTPError as exc:
        logger.error(f"[elevation] HTTP error: {exc}")
        return None


@retry(**_RETRY_POLICY)
def fetch_baseline_forecast(
    lat: float,
    lon: float,
    district: str,
    panchayat: str | None = None,
    forecast_days: int = 7,
) -> BaselineForecast:
    """
    Fetch 7-day hourly + daily forecast from Open-Meteo and return a validated
    BaselineForecast schema instance.

    Args:
        lat:           WGS-84 latitude of the target location.
        lon:           WGS-84 longitude of the target location.
        district:      Revenue district name (for GeoLocation context).
        panchayat:     Gram-panchayat name (optional).
        forecast_days: Number of forecast days (1–16). Default 7.

    Returns:
        BaselineForecast — fully validated Pydantic model.

    Raises:
        requests.HTTPError: On non-2xx response after all retries exhausted.
        requests.Timeout:   On timeout after all retries exhausted.
        ValueError:         On malformed API response.

    NOTE: Open-Meteo uses UTC timestamps. All datetimes in the returned model
          are timezone-naive UTC (ISO 8601 without 'Z') as delivered by the API.
    """
    # ── Build request ────────────────────────────────────────────────────────
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARIABLES),
        "daily": ",".join(DAILY_VARIABLES),
        "forecast_days": min(max(forecast_days, 1), 16),
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }
    url = f"{BASE_URL}/forecast?{urlencode(params)}"
    logger.debug(f"[open_meteo] GET {url}")

    # ── HTTP call ────────────────────────────────────────────────────────────
    start = time.perf_counter()
    resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    raw: dict[str, Any] = resp.json()
    logger.info(f"[open_meteo] forecast fetched in {elapsed:.2f}s for ({lat}, {lon})")

    # ── Resolve elevation ────────────────────────────────────────────────────
    # Open-Meteo forecast also returns elevation; fall back to dedicated call.
    elevation_m: float | None = raw.get("elevation")
    if elevation_m is None:
        elevation_m = fetch_elevation(lat, lon)

    # ── Build GeoLocation ────────────────────────────────────────────────────
    location = GeoLocation(
        latitude=lat,
        longitude=lon,
        elevation_m=elevation_m,
        district=district,
        panchayat=panchayat,
    )

    # ── Parse hourly rows ────────────────────────────────────────────────────
    hourly_raw: dict[str, list[Any]] = raw.get("hourly", {})
    times: list[str] = hourly_raw.get("time", [])
    temp_2m: list[float | None] = hourly_raw.get("temperature_2m", [])
    rh: list[float | None] = hourly_raw.get("relative_humidity_2m", [])
    precip: list[float | None] = hourly_raw.get("precipitation", [])
    pressure: list[float | None] = hourly_raw.get("surface_pressure", [])
    wind: list[float | None] = hourly_raw.get("wind_speed_10m", [])
    shortwave: list[float | None] = hourly_raw.get("shortwave_radiation_instant", [])
    dni: list[float | None] = hourly_raw.get("direct_normal_irradiance", [])
    soil_temp: list[float | None] = hourly_raw.get("soil_temperature_0_to_7cm", [])
    soil_moist: list[float | None] = hourly_raw.get("soil_moisture_0_to_7cm", [])
    et0: list[float | None] = hourly_raw.get("et0_fao_evapotranspiration", [])
    precip_prob: list[float | None] = hourly_raw.get("precipitation_probability", [])
    dew_point: list[float | None] = hourly_raw.get("dew_point_2m", [])
    gusts: list[float | None] = hourly_raw.get("wind_gusts_10m", [])

    hourly_points: list[HourlyForecastPoint] = []
    for i, ts in enumerate(times):
        # TRAIN/SERVE PARITY (do not "fix" this to GHI again):
        # The training dataset puts direct_normal_irradiance (DNI) in the
        # solar_radiation_w_m2 column and shortwave_radiation_instant (GHI) in
        # shortwave_radiation_w_m2 -- see scripts/ingest_multi_year_dataset.py:138
        # and services/ml_downscaler/dataset.py:187. The model therefore learned
        # solar_radiation_w_m2 == DNI, and four features derive from it
        # (nocturnal_inversion_index, solar_heating_interaction,
        #  sloped_solar_insolation, and the solar fraction inside VPD-related terms).
        # An earlier change sent GHI here on the theory that GHI matches ERA5-Land
        # downward surface solar radiation. That reasoning is about physical
        # definitions, not about what the model was fitted on, and it introduced a
        # train/serve mismatch worth hundreds of W/m2 at low sun angles.
        # Send DNI, and keep GHI in its own field.
        sw_val = _safe_idx(shortwave, i)
        dni_val = _safe_idx(dni, i)
        solar_val = dni_val if dni_val is not None else sw_val

        hourly_points.append(
            HourlyForecastPoint(
                time=datetime.fromisoformat(ts).replace(tzinfo=timezone.utc),
                temperature_2m_c=_safe_idx(temp_2m, i),
                relative_humidity_2m_pct=_safe_idx(rh, i),
                precipitation_mm=_safe_idx(precip, i),
                surface_pressure_hpa=_safe_idx(pressure, i),
                wind_speed_10m_kmh=_safe_idx(wind, i),
                solar_radiation_w_m2=solar_val,
                shortwave_radiation_w_m2=sw_val,
                soil_temperature_0_to_7cm_c=_safe_idx(soil_temp, i),
                soil_moisture_0_to_7cm_m3m3=_safe_idx(soil_moist, i),
                et0_evapotranspiration_mm=_safe_idx(et0, i),
                precipitation_probability_pct=_safe_idx(precip_prob, i),
                dew_point_2m_c=_safe_idx(dew_point, i),
                wind_gusts_10m_kmh=_safe_idx(gusts, i),
            )
        )

    # ── Parse daily aggregates ────────────────────────────────────────────────
    daily_raw: dict[str, list[Any]] = raw.get("daily", {})
    daily_dates: list[str] = daily_raw.get("time", [])
    daily_max: list[float] = daily_raw.get("temperature_2m_max", [])
    daily_min: list[float] = daily_raw.get("temperature_2m_min", [])
    daily_rain: list[float] = daily_raw.get("precipitation_sum", [])
    daily_pop_raw: list[Any] = daily_raw.get("precipitation_probability_max", [])
    daily_codes_raw: list[Any] = daily_raw.get("weathercode", [])

    daily_pop: list[int] = [int(p) if p is not None else 0 for p in daily_pop_raw]
    daily_codes: list[int] = [int(c) if c is not None else 0 for c in daily_codes_raw]

    return BaselineForecast(
        location=location,
        hourly=hourly_points,
        daily_dates=daily_dates,
        daily_max_temp=daily_max,
        daily_min_temp=daily_min,
        daily_rain_sum=daily_rain,
        daily_pop=daily_pop,
        daily_weather_code=daily_codes,
        fetched_at=datetime.now(tz=timezone.utc),
        source="OPEN_METEO",
    )


_DAILY_TEMP_CACHE: dict[tuple[float, float, str, int], tuple[float, list[DailyTemperatureRecord]]] = {}
_DAILY_TEMP_CACHE_TTL_S: float = 3600.0


@retry(**_RETRY_POLICY)
def fetch_daily_temperatures(
    lat: float,
    lon: float,
    start_date: date,
    forecast_days: int = 7,
    today: date | None = None,
) -> list[DailyTemperatureRecord]:
    """
    Daily Tmax/Tmin from `start_date` through the forecast, for growing-degree-day accumulation.

    Two sources, merged by date:
      - archive  (start_date .. yesterday): reanalysis, available to about 1-5 days ago.
      - forecast (7 past days .. forecast end): fills the archive's latency gap and the future.
    Archive values win where both exist. Days are local (Asia/Kolkata) calendar days, since
    a crop's "day" is the farmer's day.

    The forecast API's past_days only reaches about 60 days back, which is shorter than a
    kharif or rabi season, so the archive is required for anything sown earlier.
    """
    today = today or date.today()
    cache_key = (round(lat, 4), round(lon, 4), start_date.isoformat(), forecast_days)
    if cache_key in _DAILY_TEMP_CACHE:
        cached_time, cached_records = _DAILY_TEMP_CACHE[cache_key]
        if (time.time() - cached_time) < _DAILY_TEMP_CACHE_TTL_S:
            return cached_records

    records: dict[date, DailyTemperatureRecord] = {}

    def _ingest(payload: dict[str, Any], source: str) -> None:
        daily = payload.get("daily", {})
        times = daily.get("time", [])
        rain = daily.get("precipitation_sum", [None] * len(times))
        for d, hi, lo, pr in zip(times,
                                 daily.get("temperature_2m_max", []),
                                 daily.get("temperature_2m_min", []),
                                 rain):
            if hi is None or lo is None:
                continue
            day = date.fromisoformat(d)
            if source == "forecast" and day in records:
                continue  # archive already has it
            records[day] = DailyTemperatureRecord(
                date=day, tmax_c=hi, tmin_c=lo, precipitation_mm=pr, source=source)

    yesterday = today - timedelta(days=1)
    if start_date <= yesterday:
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start_date.isoformat(), "end_date": yesterday.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum", "timezone": "Asia/Kolkata",
        }
        resp = requests.get(ARCHIVE_URL, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        _ingest(resp.json(), "archive")

    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum", "timezone": "Asia/Kolkata",
        "past_days": 7, "forecast_days": min(max(forecast_days, 1), 16),
    }
    resp = requests.get(f"{BASE_URL}/forecast", params=params, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    _ingest(resp.json(), "forecast")

    result = [records[d] for d in sorted(records) if d >= start_date]
    _DAILY_TEMP_CACHE[cache_key] = (time.time(), result)
    return result


@retry(**_RETRY_POLICY)
def fetch_daily_climate_history(
    lat: float,
    lon: float,
    start: date,
    end: date,
) -> list[DailyTemperatureRecord]:
    """
    Daily rainfall and temperature extremes from the Open-Meteo archive (ERA5 reanalysis),
    for building climate normals. A 30-year request is about 0.3 MB and takes ~2 s.
    """
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=(5.0, 120.0))
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    out: list[DailyTemperatureRecord] = []
    for d, hi, lo, pr in zip(daily.get("time", []), daily.get("temperature_2m_max", []),
                             daily.get("temperature_2m_min", []), daily.get("precipitation_sum", [])):
        out.append(DailyTemperatureRecord(date=date.fromisoformat(d), tmax_c=hi, tmin_c=lo,
                                          precipitation_mm=pr, source="archive"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_idx(lst: list[Any], idx: int) -> Any:
    """Return lst[idx] or None if out of bounds."""
    try:
        return lst[idx]
    except IndexError:
        return None
