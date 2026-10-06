"""
@file ingest_acz_fleet_data.py
@description Phase 1 Ingestion Engine: Automated 10-Year (2015–2024) Hourly Dataset Ingestion
             and Leak-Free Feature Engineering for Pan-India 15 ICAR Agro-Climatic Zones (ACZs).
@module scripts
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

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

    logger = _LoguruCompatLogger("ingest_acz_fleet")  # type: ignore[assignment]

import numpy as np
import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from services.ml_downscaler.dem_source import extract_terrain
from services.ml_downscaler.zone_router import (
    ACZ_CATALOG,
    AgroClimaticZone,
    generate_zone_spatial_grid,
    get_zone_by_id,
)

OPEN_METEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
RAW_ZONES_DIR: Path = ROOT_DIR / "data" / "raw" / "zones"
CHECKPOINTS_DIR: Path = ROOT_DIR / "data" / "raw" / "checkpoints"


def fetch_zone_point_archive(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    point_id: str,
    zone_id: str,
    max_retries: int = 6,
) -> pd.DataFrame:
    """Fetch multi-year hourly historical data for a spatial point from Open-Meteo Archive API with polite 429 cooldown."""
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

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=90)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "60"))
                wait_sec = max(retry_after, 60 * attempt)
                logger.warning(
                    f"[{zone_id} | {point_id}] ⚠️ 429 Rate Limit hit. Pausing for {wait_sec}s before retry ({attempt}/{max_retries})..."
                )
                time.sleep(wait_sec)
                continue

            response.raise_for_status()
            data = response.json()
            break
        except (requests.RequestException, TimeoutError) as e:
            if attempt == max_retries:
                raise
            wait_sec = 5 * (2 ** (attempt - 1))
            logger.warning(f"[{zone_id} | {point_id}] Network error: {e}. Retrying in {wait_sec}s ({attempt}/{max_retries})...")
            time.sleep(wait_sec)
    else:
        raise RuntimeError(f"Failed to fetch data for {point_id} after {max_retries} attempts.")

    elevation_m = float(data.get("elevation", 0.0))
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    if not times:
        raise ValueError(f"Empty timeseries returned for {point_id} in {zone_id} ({latitude}, {longitude})")

    df = pd.DataFrame(
        {
            "point_id": point_id,
            "zone_id": zone_id,
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


def compute_leak_free_zone_features(
    df: pd.DataFrame,
    zone: AgroClimaticZone,
    dem_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute domain Leave-One-Out (LOO) baselines, leak-free VPD, cyclical features,
    and 30m continuous DEM terrain covariates.
    """
    logger.info(f"[{zone['zone_id']}] Computing multi-target baselines, residuals, and physics features...")
    df = df.sort_values(["point_id", "timestamp"]).reset_index(drop=True)

    # 1. Spatial Domain Mean Elevation & Standard Theoretical Lapse
    mean_elevation = float(df["elevation_m"].mean())
    df["delta_elevation_m"] = df["elevation_m"] - mean_elevation
    df["theoretical_lapse_delta_c"] = df["delta_elevation_m"] * -0.0065

    # 2. Local VPD for residual target construction ONLY (Never used as an input feature)
    rh_pct = df["relative_humidity_2m_pct"].clip(lower=1.0, upper=100.0)
    t_local = df["temperature_2m_c"]
    es_local = 0.61078 * np.exp((17.27 * t_local) / (t_local + 237.3))
    df["vpd_local_kpa"] = np.maximum(0.0, es_local - es_local * (rh_pct / 100.0))

    # 3. Domain Regional Baselines (Leave-One-Out LOO proxy across stations at each timestamp)
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

    # 3b. Leak-Free Model Feature: Vapor Pressure Deficit computed strictly from BASELINE temperature
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

    # 6. Continuous 30m DEM Morphological Covariates
    coords = df[["point_id", "latitude", "longitude", "elevation_m"]].drop_duplicates().set_index("point_id")
    point_tpi, point_slope, point_aspect_sin, point_aspect_cos = {}, {}, {}, {}

    for pid, row in coords.iterrows():
        lat, lon, elev = float(row["latitude"]), float(row["longitude"]), float(row["elevation_m"])
        t_sample = extract_terrain(lat, lon, dem_dir=dem_dir)

        if t_sample["slope_magnitude_deg"] is not None:
            point_tpi[pid] = float(t_sample.get("tpi_2000m") or 0.0)
            point_slope[pid] = float(t_sample.get("slope_magnitude_deg") or 0.5)
            point_aspect_sin[pid] = float(t_sample.get("aspect_sin") or 0.0)
            point_aspect_cos[pid] = float(t_sample.get("aspect_cos") or 1.0)
        else:
            # Mathematical lattice-gradient fallback if no raw DEM tile covers this zone yet
            dists = np.sqrt((coords["latitude"] - lat) ** 2 + (coords["longitude"] - lon) ** 2)
            neighbors = coords[(dists > 0) & (dists <= 0.45)]
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

    # 7. Derived Solar Insolation & Boundary Layer Interaction Features
    slope_factor = np.sin(np.radians(df["slope_magnitude_deg"]))
    alignment = df["aspect_sin"] * df["hour_sin"] + df["aspect_cos"] * df["hour_cos"]
    df["sloped_solar_insolation"] = (df["solar_radiation_w_m2"] / 1000.0) * (1.0 + slope_factor * alignment)
    df["soil_air_thermal_gradient"] = df["soil_temperature_0_to_7cm_c"] - df["baseline_temp_c"]
    df["latent_cooling_potential"] = df["et0_evapotranspiration_mm"] * df["soil_moisture_0_to_7cm_m3m3"]
    df["nocturnal_inversion_index"] = np.where(
        (df["hour"] >= 20) | (df["hour"] <= 6),
        df["baseline_temp_c"] - df["soil_temperature_0_to_7cm_c"],
        0.0,
    )

    return df


def ingest_zone_fleet_dataset(
    zone_id: str,
    start_year: int = 2015,
    end_year: int = 2024,
    points_count: int = 36,
    dem_dir: Optional[Path] = None,
    delay_sec: float = 2.5,
    overwrite: bool = False,
) -> Path:
    """Execute end-to-end multi-year ingestion and feature engineering for an ACZ zone."""
    zone = get_zone_by_id(zone_id)
    if not zone:
        raise ValueError(f"Unknown ACZ Zone: {zone_id}")

    RAW_ZONES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    output_parquet = RAW_ZONES_DIR / f"acz_{zone['numeric_id']:02d}_10yr_training_data.parquet"
    if output_parquet.exists() and not overwrite:
        logger.info(f"[{zone['zone_id']}] Clean parquet already exists: {output_parquet.name} (use --overwrite to re-ingest)")
        return output_parquet

    logger.info("=" * 72)
    logger.info(f"STARTING PHASE 1 INGESTION: {zone['zone_id']} - {zone['name']}")
    logger.info(f"Focus Domain : {zone['focus_domain']} (Bounding Box: {zone['bounding_box']})")
    logger.info(f"Temporal Span: {start_year}-01-01 to {end_year}-12-31 (10 Full Years)")
    logger.info(f"Spatial Grid : {points_count} Representative Spatial Stations")
    logger.info("=" * 72)

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    points = generate_zone_spatial_grid(zone["zone_id"], points_count=points_count)
    point_dfs: List[pd.DataFrame] = []

    for i, pt in enumerate(points, 1):
        pid = pt["point_id"]
        lat, lon = pt["latitude"], pt["longitude"]
        chk_file = CHECKPOINTS_DIR / f"{zone['zone_id']}_{pid}_{start_year}_{end_year}.parquet"

        if chk_file.exists() and not overwrite:
            logger.info(f"[{i:02d}/{points_count:02d}] Resuming from checkpoint: {chk_file.name}")
            pdf = pd.read_parquet(chk_file)
        else:
            logger.info(f"[{i:02d}/{points_count:02d}] Fetching {pid} ({lat:.4f}°N, {lon:.4f}°E) from Open-Meteo Archive...")
            t0 = time.time()
            pdf = fetch_zone_point_archive(
                latitude=lat,
                longitude=lon,
                start_date=start_date,
                end_date=end_date,
                point_id=pid,
                zone_id=zone["zone_id"],
            )
            pdf.to_parquet(chk_file, index=False)
            logger.info(f"       Fetched {len(pdf):,} records in {time.time()-t0:.2f}s (saved to {chk_file.name})")
            time.sleep(delay_sec)  # Polite rate-limit delay between requests

        point_dfs.append(pdf)

    # Combine all points for this zone
    logger.info(f"[{zone['zone_id']}] Combining {len(point_dfs)} spatial stations...")
    zone_df = pd.concat(point_dfs, ignore_index=True)

    # Engineer leak-free multi-target features
    clean_zone_df = compute_leak_free_zone_features(zone_df, zone=zone, dem_dir=dem_dir)

    # Save final parquet
    clean_zone_df.to_parquet(output_parquet, index=False)
    logger.success(f"[{zone['zone_id']}] Successfully wrote {len(clean_zone_df):,} leak-free records to {output_parquet}")

    return output_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Pan-India 15 ACZ Dataset Ingestion Engine")
    parser.add_argument("--zone", type=str, default=None, help="Specific zone ID (e.g. ACZ_01, ACZ_14, 1)")
    parser.add_argument("--all-zones", action="store_true", help="Ingest all 15 ACZ zones sequentially")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year (default: 2015)")
    parser.add_argument("--end-year", type=int, default=2024, help="End year (default: 2024)")
    parser.add_argument("--points-per-zone", type=int, default=36, help="Spatial stations per zone (default: 36)")
    parser.add_argument("--delay", type=float, default=2.5, help="Polite delay in seconds between API requests (default: 2.5)")
    parser.add_argument("--dem-dir", type=str, default=None, help="Directory containing DEM .hgt or .tif files")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing checkpoints and parquets")
    parser.add_argument("--dry-run", action="store_true", help="Print grid points and plan without fetching")

    args = parser.parse_args()

    target_zones: List[str] = []
    if args.all_zones:
        target_zones = list(ACZ_CATALOG.keys())
    elif args.zone:
        target_zones = [args.zone]
    else:
        # Default to ACZ_01 (Western Himalayan) as the immediate next target for Pan-India scaling
        target_zones = ["ACZ_01"]

    dem_path = Path(args.dem_dir) if args.dem_dir else None

    logger.info("=================================================================")
    logger.info("   PAN-INDIA 15 ACZ DATASET INGESTION ENGINE (PHASE 1)")
    logger.info(f"   Target Zones ({len(target_zones)}): {', '.join(target_zones)}")
    logger.info(f"   Temporal Span: {args.start_year} to {args.end_year}")
    logger.info(f"   Stations / Zone: {args.points_per_zone}")
    logger.info("=================================================================")

    if args.dry_run:
        for zid in target_zones:
            zone = get_zone_by_id(zid)
            if not zone:
                logger.error(f"Unknown zone: {zid}")
                continue
            grid = generate_zone_spatial_grid(zone["zone_id"], points_count=args.points_per_zone)
            logger.info(f"\n[DRY RUN] {zone['zone_id']} ({zone['name']}) - Focus: {zone['focus_domain']}")
            logger.info(f"Bounding Box: {zone['bounding_box']}")
            logger.info(f"Stations ({len(grid)}):")
            for pt in grid[:5]:
                logger.info(f"  - {pt['point_id']}: {pt['latitude']}°N, {pt['longitude']}°E")
            if len(grid) > 5:
                logger.info(f"  ... and {len(grid)-5} more stations.")
        return

    for zid in target_zones:
        try:
            ingest_zone_fleet_dataset(
                zone_id=zid,
                start_year=args.start_year,
                end_year=args.end_year,
                points_count=args.points_per_zone,
                dem_dir=dem_path,
                delay_sec=args.delay,
                overwrite=args.overwrite,
            )
        except Exception as e:
            logger.error(f"Failed ingestion for {zid}: {e}")
            if not args.all_zones:
                raise


if __name__ == "__main__":
    main()
