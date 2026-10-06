"""
@file downscaler_tool.py
@description Tool for executing 1km physics-guided ML downscaling on hourly forecasts.
@module services/agent/tools
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
from loguru import logger

from services.api.schemas import GeoLocation
from services.ingestion.open_meteo_client import fetch_baseline_forecast
from services.ml_downscaler.inference import downscale_point_forecast
from services.ml_downscaler.zone_router import route_coordinates_to_zone


def run_1km_downscaler(
    latitude: float,
    longitude: float,
    forecast_hours: int = 48,
    district: str = "Jaipur"
) -> Dict[str, Any]:
    """
    Execute 1km physics-guided downscaling for given coordinates.
    Returns hourly downscaled metrics and domain anomaly deltas.
    """
    start_time = time.perf_counter()
    res = route_coordinates_to_zone(latitude, longitude)
    zone_id = res[0]["zone_id"] if isinstance(res, tuple) else (res.get("zone_id") if isinstance(res, dict) else str(res))
    
    # Ingest baseline forecast
    baseline = fetch_baseline_forecast(lat=latitude, lon=longitude, district=district)
    hourly_points = baseline.hourly[:forecast_hours]
    
    loc = GeoLocation(
        latitude=latitude,
        longitude=longitude,
        district=district,
        elevation_m=getattr(baseline.location, "elevation_m", 400.0) or 400.0
    )
    
    downscaled_series: List[Dict[str, Any]] = []
    total_delta_t = 0.0
    
    for pt in hourly_points:
        t_base = float(pt.temperature_2m_c if pt.temperature_2m_c is not None else 25.0)
        rh = float(pt.relative_humidity_2m_pct if pt.relative_humidity_2m_pct is not None else 50.0)
        wind = float(pt.wind_speed_10m_kmh if pt.wind_speed_10m_kmh is not None else 10.0)
        soil_t = float(pt.soil_temperature_0_to_7cm_c if pt.soil_temperature_0_to_7cm_c is not None else (t_base - 1.0))
        soil_m = float(pt.soil_moisture_0_to_7cm_m3m3 if pt.soil_moisture_0_to_7cm_m3m3 is not None else 0.20)
        solar = float(pt.solar_radiation_w_m2 if pt.solar_radiation_w_m2 is not None else 0.0)
        pop = float(pt.precipitation_probability_pct if pt.precipitation_probability_pct is not None else 0.0)
        
        covariates = {
            "relative_humidity_2m": rh,
            "wind_speed_10m": wind,
            "solar_radiation_w_m2": solar,
            "surface_pressure_hpa": float(pt.surface_pressure_hpa if pt.surface_pressure_hpa is not None else 980.0),
            "cloud_cover_pct": 0.0,
            "soil_temperature_0_to_7cm": soil_t,
            "soil_moisture_0_to_7cm": soil_m,
            "et0_fao_evapotranspiration": float(pt.et0_evapotranspiration_mm if pt.et0_evapotranspiration_mm is not None else 0.25),
            "delta_elevation_m": 0.0
        }
        
        downscaled_pt = downscale_point_forecast(
            panchayat_id="KADERA_001",
            location=loc,
            baseline_point=pt,
            covariates=covariates
        )
        
        val = float(downscaled_pt.downscaled_value if downscaled_pt.downscaled_value is not None else t_base)
        delta_t = val - t_base
        total_delta_t += delta_t
        
        downscaled_series.append({
            "timestamp": pt.time.isoformat() if hasattr(pt.time, "isoformat") else str(pt.time),
            "baseline_temp_c": round(t_base, 2),
            "downscaled_temp_1km_c": round(val, 2),
            "delta_t_c": round(delta_t, 2),
            "relative_humidity_pct": round(rh, 1),
            "wind_speed_kmh": round(wind, 1),
            "solar_radiation_w_m2": round(solar, 1),
            "soil_moisture_m3m3": round(soil_m, 3),
            "rain_probability_pct": round(pop, 1)
        })
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    mean_delta_t = total_delta_t / len(downscaled_series) if downscaled_series else 0.0
    
    return {
        "zone_id": zone_id,
        "latitude": latitude,
        "longitude": longitude,
        "hours_evaluated": len(downscaled_series),
        "mean_delta_t_c": round(mean_delta_t, 3),
        "execution_time_ms": round(elapsed_ms, 2),
        "hourly_series": downscaled_series
    }
