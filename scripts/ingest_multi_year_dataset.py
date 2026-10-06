"""
@file ingest_multi_year_dataset.py
@description Scalable 10-year historical reanalysis ingestion pipeline with multi-target schemas, hydrological memory features, checkpointing, and verification.
@module scripts
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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

    logger = _LoguruCompatLogger("ingest_multi_year")  # type: ignore[assignment]

import numpy as np
import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from services.ml_downscaler.dataset import (
    DEFAULT_BOUNDING_BOX,
    OPEN_METEO_ARCHIVE_URL,
    generate_grid_coordinates,
)

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
CHECKPOINTS_DIR = RAW_DATA_DIR / "checkpoints"


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.5, min=2, max=20),
    retry=retry_if_exception_type((requests.RequestException, TimeoutError)),
    reraise=True,
)
def fetch_multiyear_point_archive(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    point_id: str = "POINT_001",
) -> pd.DataFrame:
    """
    Fetch multi-year hourly historical reanalysis data for a spatial point from Open-Meteo archive.
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
            "wind_gusts_10m",
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
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    elevation_m = float(data.get("elevation", 0.0))
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    if not times:
        raise ValueError(f"Empty timeseries returned for {point_id} ({latitude}, {longitude})")

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
            "wind_gusts_10m_kmh": hourly.get("wind_gusts_10m", hourly.get("wind_speed_10m", [])),
            "solar_radiation_w_m2": hourly.get("direct_normal_irradiance", []),
            "shortwave_radiation_w_m2": hourly.get("shortwave_radiation_instant", []),
            "soil_temperature_0_to_7cm_c": hourly.get("soil_temperature_0_to_7cm", []),
            "soil_moisture_0_to_7cm_m3m3": hourly.get("soil_moisture_0_to_7cm", []),
            "et0_evapotranspiration_mm": hourly.get("et0_fao_evapotranspiration", []),
        }
    )
    return df


def engineer_multitarget_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute multi-target residuals, topographic covariates, boundary layer physics,
    and antecedent hydrological memory features.
    """
    logger.info("Computing multi-target baselines, residuals, and physics features...")
    df = df.sort_values(["point_id", "timestamp"]).reset_index(drop=True)

    # 1. Spatial Domain Mean Elevation & Standard Theoretical Lapse
    mean_elevation = float(df["elevation_m"].mean())
    df["delta_elevation_m"] = df["elevation_m"] - mean_elevation
    df["theoretical_lapse_delta_c"] = df["delta_elevation_m"] * -0.0065

    # 2. Tetens Equation for Vapor Pressure Deficit (VPD in kPa)
    #
    # TARGET-LEAKAGE FIX (audit finding F22). This block used to compute the model FEATURE
    # `vapor_pressure_deficit_kpa` from temperature_2m_c -- the point's own temperature,
    # i.e. the quantity being predicted. The target is R = T_local - baseline, and VPD is a
    # function of (T_local, RH); with RH and the baseline also given as features, the model
    # could recover the answer by inverting Tetens. Evidence: shuffling VPD raised test RMSE
    # by 8.1 C (about the annual temperature spread), and the champion scored only +21-32%
    # once VPD was computed honestly, versus +42-48% for a model retrained on honest features.
    #
    # Two quantities are therefore kept apart:
    #   vpd_local_kpa               from the LOCAL temperature. Used ONLY to build the VPD
    #                               downscaling target (residual_vpd_kpa). Never a feature.
    #   vapor_pressure_deficit_kpa  the model FEATURE, from the regional BASELINE temperature,
    #                               computed after step 3 -- the same definition dataset.py,
    #                               train.py and inference.py use.
    rh_pct = df["relative_humidity_2m_pct"].clip(lower=1.0, upper=100.0)
    t_local = df["temperature_2m_c"]
    es_local = 0.61078 * np.exp((17.27 * t_local) / (t_local + 237.3))
    df["vpd_local_kpa"] = np.maximum(0.0, es_local - es_local * (rh_pct / 100.0))

    # 3. Domain Regional Baselines (Leave-One-Out LOO proxy aggregated across spatial points at each timestamp)
    for col, baseline_col in [
        ("temperature_2m_c", "baseline_temp_c"),
        ("soil_moisture_0_to_7cm_m3m3", "baseline_soil_moisture_m3m3"),
        ("relative_humidity_2m_pct", "baseline_relative_humidity_pct"),
        ("wind_speed_10m_kmh", "baseline_wind_speed_kmh"),
        ("vpd_local_kpa", "baseline_vpd_kpa"),
    ]:
        ts_sum = df.groupby("timestamp")[col].transform("sum")
        ts_count = df.groupby("timestamp")[col].transform("count")
        df[baseline_col] = np.where(
            ts_count > 1,
            (ts_sum - df[col]) / (ts_count - 1),
            df[col],
        )

    # 3b. The VPD model FEATURE, from the regional baseline temperature (see step 2).
    t_base = df["baseline_temp_c"]
    es_base = 0.61078 * np.exp((17.27 * t_base) / (t_base + 237.3))
    df["vapor_pressure_deficit_kpa"] = np.maximum(0.0, es_base - es_base * (rh_pct / 100.0))

    # 4. Multi-Target Residuals (R = Local - Baseline)
    df["residual_anomaly_c"] = df["temperature_2m_c"] - df["baseline_temp_c"]
    df["residual_soil_moisture_m3m3"] = df["soil_moisture_0_to_7cm_m3m3"] - df["baseline_soil_moisture_m3m3"]
    df["residual_vpd_kpa"] = df["vpd_local_kpa"] - df["baseline_vpd_kpa"]
    df["residual_wind_speed_kmh"] = df["wind_speed_10m_kmh"] - df["baseline_wind_speed_kmh"]

    # 5. Temporal Cyclical Features
    timestamps = pd.DatetimeIndex(df["timestamp"])
    hours = timestamps.hour.to_numpy()
    days_of_year = timestamps.dayofyear.to_numpy()

    df["hour"] = hours
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    df["day_of_year"] = days_of_year
    df["doy_sin"] = np.sin(2 * np.pi * days_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * days_of_year / 365.25)

    # 6. Topographic Position Index (TPI), Slope & Aspect
    coords = df[["point_id", "latitude", "longitude", "elevation_m"]].drop_duplicates().set_index("point_id")
    point_tpi, point_slope, point_aspect_sin, point_aspect_cos = {}, {}, {}, {}
    for pid, row in coords.iterrows():
        lat, lon, elev = row["latitude"], row["longitude"], row["elevation_m"]
        dists = np.sqrt((coords["latitude"] - lat) ** 2 + (coords["longitude"] - lon) ** 2)
        neighbors = coords[(dists > 0) & (dists <= 0.28)]
        if len(neighbors) > 0:
            mean_neighbor_elev = float(neighbors["elevation_m"].mean())
            tpi = float(elev - mean_neighbor_elev)
            d_lat = neighbors["latitude"].values - lat
            d_lon = neighbors["longitude"].values - lon
            d_elev = neighbors["elevation_m"].values - elev
            grad_ns = float(np.mean(d_elev / np.maximum(1e-5, np.abs(d_lat * 111000.0)) * np.sign(d_lat)))
            grad_ew = float(np.mean(d_elev / np.maximum(1e-5, np.abs(d_lon * 100000.0)) * np.sign(d_lon)))
            slope_deg = float(np.degrees(np.arctan(np.sqrt(grad_ns**2 + grad_ew**2))))
            aspect_rad = float(np.arctan2(grad_ns, grad_ew))
            aspect_s = float(np.sin(aspect_rad))
            aspect_c = float(np.cos(aspect_rad))
        else:
            tpi, slope_deg, aspect_s, aspect_c = 0.0, 0.5, 0.0, 1.0
        point_tpi[pid] = tpi
        point_slope[pid] = slope_deg
        point_aspect_sin[pid] = aspect_s
        point_aspect_cos[pid] = aspect_c

    df["topographic_position_index"] = df["point_id"].map(point_tpi).fillna(0.0)
    df["slope_magnitude_deg"] = df["point_id"].map(point_slope).fillna(0.5)
    df["aspect_sin"] = df["point_id"].map(point_aspect_sin).fillna(0.0)
    df["aspect_cos"] = df["point_id"].map(point_aspect_cos).fillna(1.0)

    # 7. Solar Insolation & Boundary Layer Coupling
    aspect_alignment = df["aspect_sin"] * df["hour_sin"] + df["aspect_cos"] * df["hour_cos"]
    slope_factor = np.sin(np.radians(df["slope_magnitude_deg"]))
    solar_rad_norm = df["solar_radiation_w_m2"] / 1000.0
    df["sloped_solar_insolation"] = solar_rad_norm * (1.0 + slope_factor * aspect_alignment)

    # Nocturnal inversion & Solar heating interaction
    valley_depth_m = np.maximum(0.0, -df["delta_elevation_m"])
    wind_speed = df["wind_speed_10m_kmh"].clip(lower=0.5)
    solar_fraction = (df["solar_radiation_w_m2"] / 1000.0).clip(lower=0.0, upper=1.0)
    night_weight = 1.0 - solar_fraction
    df["nocturnal_inversion_index"] = (valley_depth_m / wind_speed) * night_weight

    diurnal_factor = np.maximum(0.0, -df["hour_cos"])
    df["solar_heating_interaction"] = (df["solar_radiation_w_m2"] / 1000.0) * diurnal_factor

    # Thermal gradient & Latent cooling
    df["soil_air_thermal_gradient"] = df["soil_temperature_0_to_7cm_c"] - df["baseline_temp_c"]
    df["latent_cooling_potential"] = df["et0_evapotranspiration_mm"] * df["vapor_pressure_deficit_kpa"]

    # Thermal inertia lag (3h per point)
    df = df.sort_values(["point_id", "timestamp"]).reset_index(drop=True)
    df["thermal_inertia_lag_3h"] = df.groupby("point_id")["baseline_temp_c"].diff(3).fillna(0.0)

    # 8. Hydrological & Memory Features (Per-point rolling timeseries)
    logger.info("Computing antecedent hydrological memory features per spatial point...")
    df_point_groups: list[pd.DataFrame] = []
    for pid, group in df.groupby("point_id"):
        grp = group.sort_values("timestamp").copy()
        # A. Antecedent Precipitation (API 7-day = 168 hours rolling sum)
        grp["antecedent_precipitation_7d_mm"] = grp["precipitation_mm"].rolling(168, min_periods=1).sum()
        # B. Soil moisture lag (24 hours prior)
        grp["soil_moisture_lag_24h_m3m3"] = grp["soil_moisture_0_to_7cm_m3m3"].shift(24).bfill()
        # C. Cumulative Growing Degree Days (GDD base 10°C)
        daily_temp = grp["temperature_2m_c"]
        gdd_increment = np.maximum(0.0, (daily_temp - 10.0) / 24.0)
        # Cumulative GDD within each calendar year
        grp["year"] = grp["timestamp"].dt.year
        grp["accumulated_gdd_c"] = grp.groupby("year")[gdd_increment.name if hasattr(gdd_increment, 'name') else "temp_inc"].transform("cumsum") if "year" in grp.columns else gdd_increment.cumsum()
        grp = grp.drop(columns=["year"], errors="ignore")
        df_point_groups.append(grp)

    df = pd.concat(df_point_groups, ignore_index=True)

    # 9. Extended Physics Interaction Terms (Option A)
    # Evaporative demand
    df["evaporative_demand"] = df["et0_evapotranspiration_mm"] * df["vapor_pressure_deficit_kpa"] * (df["solar_radiation_w_m2"] / 1000.0)
    # Infiltration runoff index
    df["infiltration_potential"] = df["precipitation_mm"] / (df["slope_magnitude_deg"] + 1.0)
    # Valley dew potential
    tpi_neg = np.maximum(0.0, -df["topographic_position_index"])
    df["valley_dew_potential"] = tpi_neg * (df["relative_humidity_2m_pct"] / 100.0)
    # Ridge wind acceleration
    tpi_pos = np.maximum(0.0, df["topographic_position_index"])
    df["ridge_wind_acceleration"] = tpi_pos * (df["wind_speed_10m_kmh"] / 10.0)

    # Fill any remaining border NaNs
    df = df.bfill().ffill().reset_index(drop=True)
    return df


def ingest_multiyear_dataset(
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Ingest multi-year dataset across all lattice grid points with checkpointing.
    """
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    coords = generate_grid_coordinates(
        north=DEFAULT_BOUNDING_BOX["north"],
        south=DEFAULT_BOUNDING_BOX["south"],
        west=DEFAULT_BOUNDING_BOX["west"],
        east=DEFAULT_BOUNDING_BOX["east"],
        step_deg=0.14,
    )

    total_points = len(coords)
    logger.info(
        "Starting multi-year extraction for {} points ({} to {})...",
        total_points,
        start_date,
        end_date,
    )

    point_dfs: list[pd.DataFrame] = []
    t_start = time.perf_counter()

    for idx, c in enumerate(coords, 1):
        pid = c["point_id"]
        chk_file = CHECKPOINTS_DIR / f"{pid}_{start_date}_{end_date}.parquet"

        if chk_file.exists() and not force_refresh:
            logger.info("[{}/{}] Loading cached checkpoint for {} from {}", idx, total_points, pid, chk_file.name)
            pt_df = pd.read_parquet(chk_file)
        else:
            logger.info("[{}/{}] Fetching {} (Lat: {}, Lon: {})...", idx, total_points, pid, c["latitude"], c["longitude"])
            pt_df = fetch_multiyear_point_archive(
                latitude=c["latitude"],
                longitude=c["longitude"],
                start_date=start_date,
                end_date=end_date,
                point_id=pid,
            )
            # Save point checkpoint
            pt_df.to_parquet(chk_file, index=False)
            # Polite pause for API rate limit
            time.sleep(0.3)

        point_dfs.append(pt_df)

    raw_combined = pd.concat(point_dfs, ignore_index=True)
    logger.success("Raw timeseries extracted: {} rows across {} points", len(raw_combined), total_points)

    # Feature Engineering
    featured_df = engineer_multitarget_features(raw_combined)
    elapsed = time.perf_counter() - t_start

    # Validation Checks
    null_count = int(featured_df.isnull().sum().sum())
    assert null_count == 0, f"Validation failure: Found {null_count} nulls in featured dataset!"
    assert len(featured_df) > 1_000_000, f"Expected > 1M rows for multi-year dataset, got {len(featured_df)}"

    # Save to Parquet
    out_parquet = RAW_DATA_DIR / "jaipur_10yr_training_data.parquet"
    featured_df.to_parquet(out_parquet, index=False, engine="pyarrow", compression="snappy")
    parquet_size_mb = os.path.getsize(out_parquet) / (1024 * 1024)
    logger.success("Saved compressed Parquet to {} ({:.2f} MB)", out_parquet, parquet_size_mb)

    # Write Manifest
    manifest = {
        "dataset_name": "VATA_Jaipur_MultiYear_Microclimate_Downscaling",
        "temporal_range": {"start_date": start_date, "end_date": end_date, "total_years": 10},
        "spatial_domain": DEFAULT_BOUNDING_BOX,
        "total_spatial_points": total_points,
        "total_hourly_rows": len(featured_df),
        "total_features": len(featured_df.columns),
        "null_values": null_count,
        "extraction_duration_sec": round(elapsed, 2),
        "parquet_size_mb": round(parquet_size_mb, 2),
        "targets": [
            "residual_anomaly_c",
            "residual_soil_moisture_m3m3",
            "residual_vpd_kpa",
            "residual_wind_speed_kmh",
        ],
        "columns": list(featured_df.columns),
    }
    manifest_path = RAW_DATA_DIR / "dataset_10yr_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.success("Wrote dataset manifest to {}", manifest_path)

    return featured_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-year downscaling dataset ingestion.")
    parser.add_argument("--start-date", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="Force re-download and bypass checkpoints")
    args = parser.parse_args()

    logger.info("=================================================================")
    logger.info("  SIH 26074: VATA 10-YEAR MULTI-TARGET DATASET INGESTION ENGINE  ")
    logger.info("=================================================================")
    ingest_multiyear_dataset(
        start_date=args.start_date,
        end_date=args.end_date,
        force_refresh=args.force,
    )


if __name__ == "__main__":
    main()
