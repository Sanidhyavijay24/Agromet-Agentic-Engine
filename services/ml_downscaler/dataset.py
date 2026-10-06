"""
@file dataset.py
@description Dataset extraction and feature engineering pipeline for ML microclimate downscaler.
@module services/ml_downscaler
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    class _LoguruCompatLogger:
        def __init__(self, name: str) -> None:
            self._logger = logging.getLogger(name)
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
                self._logger.addHandler(handler)
                self._logger.setLevel(logging.INFO)

        def _format_msg(self, msg: str, *args: Any) -> str:
            if args:
                try:
                    return msg.format(*args)
                except Exception:
                    return f"{msg} {args}"
            return msg

        def info(self, msg: str, *args: Any) -> None:
            self._logger.info(self._format_msg(msg, *args))

        def warning(self, msg: str, *args: Any) -> None:
            self._logger.warning(self._format_msg(msg, *args))

        def error(self, msg: str, *args: Any) -> None:
            self._logger.error(self._format_msg(msg, *args))

        def success(self, msg: str, *args: Any) -> None:
            self._logger.info(self._format_msg(msg, *args))

    logger = _LoguruCompatLogger("ml_downscaler")  # type: ignore[assignment]

import numpy as np
import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Target pilot evaluation domain: Jaipur Rural & Chaksu, Rajasthan
DEFAULT_BOUNDING_BOX: dict[str, float] = {
    "north": 27.2,
    "south": 26.5,
    "west": 75.5,
    "east": 76.2,
}

OPEN_METEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT_SECONDS: int = 30


def generate_grid_coordinates(
    north: float = 27.2,
    south: float = 26.5,
    west: float = 75.5,
    east: float = 76.2,
    step_deg: float = 0.14,
) -> list[dict[str, Any]]:
    """
    Generate a spatial lattice of coordinate points within the bounding box.

    Args:
        north: Northern latitude limit.
        south: Southern latitude limit.
        west: Western longitude limit.
        east: Eastern longitude limit.
        step_deg: Grid step in decimal degrees (~0.14 deg ~ 15 km spacing).

    Returns:
        List of dicts with lat, lon, point_id.
    """
    lats = np.arange(south, north + (step_deg / 2), step_deg)
    lons = np.arange(west, east + (step_deg / 2), step_deg)

    coords: list[dict[str, Any]] = []
    idx = 1
    for lat in lats:
        for lon in lons:
            coords.append(
                {
                    "point_id": f"GRID_{idx:03d}",
                    "latitude": round(float(lat), 4),
                    "longitude": round(float(lon), 4),
                }
            )
            idx += 1

    logger.info(
        "Generated spatial grid of {} points across [N:{}, S:{}, W:{}, E:{}]",
        len(coords),
        north,
        south,
        west,
        east,
    )
    return coords


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.RequestException, TimeoutError)),
    reraise=True,
)
def fetch_historical_point_archive(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    point_id: str = "POINT_001",
) -> pd.DataFrame:
    """
    Fetch hourly historical reanalysis data for a single geographic coordinate.

    Args:
        latitude: WGS-84 latitude.
        longitude: WGS-84 longitude.
        start_date: ISO date string (YYYY-MM-DD).
        end_date: ISO date string (YYYY-MM-DD).
        point_id: Unique identifier for this spatial point.

    Returns:
        Pandas DataFrame containing hourly meteorological records with elevation.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "precipitation",
            "wind_speed_10m",
            "direct_normal_irradiance",
            "shortwave_radiation_instant",
            "soil_temperature_0_to_7cm",
            "soil_moisture_0_to_7cm",
            "et0_fao_evapotranspiration",
        ],
        "timezone": "UTC",
    }

    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()

    elevation_m = float(data.get("elevation", 0.0))
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    if not times:
        raise ValueError(f"Empty timeseries returned for ({latitude}, {longitude})")

    df = pd.DataFrame(
        {
            "point_id": point_id,
            "latitude": latitude,
            "longitude": longitude,
            "elevation_m": elevation_m,
            "timestamp": pd.to_datetime(times, utc=True),
            "temperature_2m_c": hourly.get("temperature_2m", []),
            "relative_humidity_2m_pct": hourly.get("relative_humidity_2m", []),
            "surface_pressure_hpa": hourly.get("surface_pressure", []),
            "precipitation_mm": hourly.get("precipitation", []),
            "wind_speed_10m_kmh": hourly.get("wind_speed_10m", []),
            "solar_radiation_w_m2": hourly.get("direct_normal_irradiance", []),
            "shortwave_radiation_w_m2": hourly.get("shortwave_radiation_instant", []),
            "soil_temperature_0_to_7cm_c": hourly.get("soil_temperature_0_to_7cm", []),
            "soil_moisture_0_to_7cm_m3m3": hourly.get("soil_moisture_0_to_7cm", []),
            "et0_evapotranspiration_mm": hourly.get("et0_fao_evapotranspiration", []),
        }
    )
    return df


def engineer_downscaling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain baseline, residual anomaly target, and physical covariates.

    Features Engineered:
    - delta_elevation_m: Elevation relative to domain mean.
    - theoretical_lapse_delta_c: Delta elevation * -0.0065 C/m lapse rate.
    - baseline_temp_c: Spatial domain mean temperature at each hourly timestamp.
    - residual_anomaly_c: Ground-truth target (T_local - T_baseline).
    - hour_sin / hour_cos: Diurnal solar cycle encodings.
    - doy_sin / doy_cos: Seasonal cycle encodings.

    Args:
        df: Combined hourly DataFrame across all spatial points.

    Returns:
        Enriched DataFrame with engineered features ready for XGBoost.
    """
    logger.info("Computing domain baselines and microclimate residual targets...")

    # 1. Compute spatial domain mean elevation
    mean_elevation = float(df["elevation_m"].mean())
    df["delta_elevation_m"] = df["elevation_m"] - mean_elevation
    # Environmental standard lapse rate ~ -6.5 C per 1000m (-0.0065 C/m)
    df["theoretical_lapse_delta_c"] = df["delta_elevation_m"] * -0.0065

    # 2. Compute Leave-One-Out (LOO) domain baseline (coarse NWP regional mean excluding current point)
    ts_sum = df.groupby("timestamp")["temperature_2m_c"].transform("sum")
    ts_count = df.groupby("timestamp")["temperature_2m_c"].transform("count")
    df["baseline_temp_c"] = np.where(
        ts_count > 1,
        (ts_sum - df["temperature_2m_c"]) / (ts_count - 1),
        df["temperature_2m_c"],
    )

    # 3. Target Residual Anomaly: R = T_local - T_baseline
    df["residual_anomaly_c"] = df["temperature_2m_c"] - df["baseline_temp_c"]

    # 4. Temporal Cyclical Features
    timestamps = pd.DatetimeIndex(df["timestamp"])
    hours = timestamps.hour.to_numpy()
    days_of_year = timestamps.dayofyear.to_numpy()

    df["hour"] = hours
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    df["day_of_year"] = days_of_year
    df["doy_sin"] = np.sin(2 * np.pi * days_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * days_of_year / 365.25)

    # 5. Physics-Guided Feature Interactions
    # A. Vapor Pressure Deficit (VPD in kPa) using Tetens equation
    t_c = df["baseline_temp_c"]
    rh_pct = df["relative_humidity_2m_pct"].clip(lower=1.0, upper=100.0)
    es_kpa = 0.61078 * np.exp((17.27 * t_c) / (t_c + 237.3))
    ea_kpa = es_kpa * (rh_pct / 100.0)
    df["vapor_pressure_deficit_kpa"] = np.maximum(0.0, es_kpa - ea_kpa)

    # B. Nocturnal Inversion Index (valley cold-air pooling under calm wind at night)
    valley_depth_m = np.maximum(0.0, -df["delta_elevation_m"])
    wind_speed = df["wind_speed_10m_kmh"].clip(lower=0.5)
    solar_fraction = (df["solar_radiation_w_m2"] / 1000.0).clip(lower=0.0, upper=1.0)
    night_weight = 1.0 - solar_fraction
    df["nocturnal_inversion_index"] = (valley_depth_m / wind_speed) * night_weight

    # C. Solar Heating Interaction (solar irradiance scaled by diurnal solar angle)
    solar_rad = df["solar_radiation_w_m2"].clip(lower=0.0)
    diurnal_factor = np.maximum(0.0, -df["hour_cos"])
    df["solar_heating_interaction"] = (solar_rad / 1000.0) * diurnal_factor

    # D. Thermal Inertia / 3-hour temperature rate of change (per point)
    df = df.sort_values(["point_id", "timestamp"]).reset_index(drop=True)
    df["thermal_inertia_lag_3h"] = df.groupby("point_id")["baseline_temp_c"].diff(3).fillna(0.0)

    # 6. Clean / validate missing values
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    cleaned_len = len(df)
    if initial_len != cleaned_len:
        logger.warning(
            "Dropped {} rows containing null values ({} remaining)",
            initial_len - cleaned_len,
            cleaned_len,
        )

    return df


def extract_downscaling_dataset(
    coords: list[dict[str, Any]],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Extract and assemble historical timeseries for all coordinates in the lattice.

    Args:
        coords: List of coordinate dicts with point_id, latitude, longitude.
        start_date: ISO date (YYYY-MM-DD).
        end_date: ISO date (YYYY-MM-DD).

    Returns:
        Complete feature-engineered pandas DataFrame.
    """
    frames: list[pd.DataFrame] = []
    total = len(coords)

    for i, c in enumerate(coords, 1):
        logger.info(
            "[{}/{}] Fetching point {} (Lat: {}, Lon: {})...",
            i,
            total,
            c["point_id"],
            c["latitude"],
            c["longitude"],
        )
        point_df = fetch_historical_point_archive(
            latitude=c["latitude"],
            longitude=c["longitude"],
            start_date=start_date,
            end_date=end_date,
            point_id=c["point_id"],
        )
        frames.append(point_df)

    combined_df = pd.concat(frames, ignore_index=True)
    featured_df = engineer_downscaling_features(combined_df)

    logger.success(
        "Successfully extracted and engineered dataset: {} total rows, {} features",
        len(featured_df),
        len(featured_df.columns),
    )
    return featured_df
