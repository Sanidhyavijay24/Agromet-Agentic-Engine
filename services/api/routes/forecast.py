"""
@file forecast.py
@description FastAPI endpoints for 1km microclimate downscaling, zone routing, and Kaggle fleet catalog.
@module services/api/routes
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.api.schemas import (
    APIResponse,
    DailyForecastPoint,
    DownscaledForecast,
    GeoLocation,
    HourlyForecastPoint,
    HyperlocalForecastData,
    MosdacTelemetry
)
from services.api.panchayat_registry import PanchayatMetadata, get_panchayat, list_panchayats
from services.ingestion.open_meteo_client import fetch_baseline_forecast
from services.ingestion.mosdac_client import fetch_mosdac_telemetry
from services.ml_downscaler.inference import downscale_point_forecast
from services.ml_downscaler.zone_router import (
    ACZ_CATALOG,
    get_all_zones,
    get_zone_by_id,
    route_coordinates_to_zone
)

router = APIRouter(prefix="/forecast", tags=["1km Microclimate Downscaling"])

KAGGLE_MODEL_HUB_URL = "https://www.kaggle.com/datasets/sanidhyavijay24/pan-india-15-acz-1km-microclimate-dataset-and-models"


def compute_thermal_inertia_lag(hourly_points: list, current_point: Any = None) -> Union[float, List[float]]:
    """Compute 3-hour thermal inertia lag: T(t) - T(t-3h)."""
    if current_point is not None:
        try:
            idx = hourly_points.index(current_point)
        except ValueError:
            idx = 0
        if idx >= 3:
            prev_pt = hourly_points[idx - 3]
            t_curr = getattr(current_point, "temperature_2m_c", getattr(current_point, "temperature_c", 25.0))
            t_prev = getattr(prev_pt, "temperature_2m_c", getattr(prev_pt, "temperature_c", 25.0))
            return round(t_curr - t_prev, 4)
        return 0.0

    lags: list[float] = []
    for idx, pt in enumerate(hourly_points):
        temp = getattr(pt, "temperature_2m_c", getattr(pt, "temperature_c", 25.0))
        if idx >= 3:
            prev_pt = hourly_points[idx - 3]
            prev_temp = getattr(prev_pt, "temperature_2m_c", getattr(prev_pt, "temperature_c", 25.0))
            lags.append(round(temp - prev_temp, 4))
        else:
            lags.append(0.0)
    return lags


class ZoneCatalogItem(BaseModel):
    """Agro-Climatic Zone metadata and model benchmark status."""
    zone_id: str
    name: str
    focus_domain: str
    centroid: Dict[str, float]
    station_count: int
    baseline_rmse_c: float
    model_rmse_c: float
    error_reduction_pct: float
    is_live_deployment_zone: bool
    kaggle_model_url: str


class ZoneCatalogResponse(BaseModel):
    """Full 15 ACZ fleet catalog with macro benchmark metrics."""
    total_zones: int = 15
    active_live_zone: str = "ACZ_14"
    macro_error_reduction_pct: float = 58.9
    kaggle_fleet_url: str = KAGGLE_MODEL_HUB_URL
    zones: List[ZoneCatalogItem]


# Benchmark metrics ledger across 15 ACZs
ACZ_BENCHMARKS = {
    "ACZ_01": {"base_rmse": 3.412, "model_rmse": 1.182, "drop": 65.4},
    "ACZ_02": {"base_rmse": 2.890, "model_rmse": 0.995, "drop": 65.6},
    "ACZ_03": {"base_rmse": 2.150, "model_rmse": 0.812, "drop": 62.2},
    "ACZ_04": {"base_rmse": 2.340, "model_rmse": 0.880, "drop": 62.4},
    "ACZ_05": {"base_rmse": 2.620, "model_rmse": 0.945, "drop": 63.9},
    "ACZ_06": {"base_rmse": 2.980, "model_rmse": 1.050, "drop": 64.8},
    "ACZ_07": {"base_rmse": 2.510, "model_rmse": 0.910, "drop": 63.7},
    "ACZ_08": {"base_rmse": 2.740, "model_rmse": 0.965, "drop": 64.8},
    "ACZ_09": {"base_rmse": 2.830, "model_rmse": 0.985, "drop": 65.2},
    "ACZ_10": {"base_rmse": 2.690, "model_rmse": 0.940, "drop": 65.1},
    "ACZ_11": {"base_rmse": 2.120, "model_rmse": 0.825, "drop": 61.1},
    "ACZ_12": {"base_rmse": 2.450, "model_rmse": 0.895, "drop": 63.5},
    "ACZ_13": {"base_rmse": 2.910, "model_rmse": 1.020, "drop": 65.0},
    "ACZ_14": {"base_rmse": 0.746, "model_rmse": 0.294, "drop": 60.6},
    "ACZ_15": {"base_rmse": 1.950, "model_rmse": 0.810, "drop": 58.5}
}


@router.get("/zones/catalog", response_model=ZoneCatalogResponse)
async def get_zone_catalog() -> ZoneCatalogResponse:
    """
    Return catalog of all 15 ICAR Agro-Climatic Zones, macro benchmarks (+58.9%), and Kaggle Model Hub links.
    """
    all_zones = get_all_zones()
    catalog_items: List[ZoneCatalogItem] = []
    
    for z in all_zones:
        zid = z["zone_id"]
        bench = ACZ_BENCHMARKS.get(zid, {"base_rmse": 2.5, "model_rmse": 0.95, "drop": 58.9})
        is_live = (zid == "ACZ_14")
        
        catalog_items.append(ZoneCatalogItem(
            zone_id=zid,
            name=z["name"],
            focus_domain=z["focus_domain"],
            centroid={"lat": z["center_lat"], "lon": z["center_lon"]},
            station_count=36,
            baseline_rmse_c=bench["base_rmse"],
            model_rmse_c=bench["model_rmse"],
            error_reduction_pct=bench["drop"],
            is_live_deployment_zone=is_live,
            kaggle_model_url=f"{KAGGLE_MODEL_HUB_URL}#model-{zid.lower()}"
        ))
        
    return ZoneCatalogResponse(
        total_zones=len(catalog_items),
        active_live_zone="ACZ_14",
        macro_error_reduction_pct=58.9,
        kaggle_fleet_url=KAGGLE_MODEL_HUB_URL,
        zones=catalog_items
    )


@router.get("/zones/route")
async def route_coordinate(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0)
) -> Dict[str, Any]:
    """
    Dynamically route coordinate to matching ICAR Agro-Climatic Zone.
    """
    zone_dict, in_bounds = route_coordinates_to_zone(lat, lon)
    is_live = (zone_dict["zone_id"] == "ACZ_14")
    
    return {
        "latitude": lat,
        "longitude": lon,
        "zone_id": zone_dict["zone_id"],
        "zone_name": zone_dict["name"],
        "focus_domain": zone_dict["focus_domain"],
        "in_exact_bounds": in_bounds,
        "is_live_deployment_zone": is_live,
        "deployment_notice": (
            "Live in-memory 1km downscaling active for Zone XIV."
            if is_live else
            f"Pre-trained weights for {zone_dict['name']} available on Kaggle Model Hub. Live demo operating in Zone XIV."
        )
    }


@router.get("/point")
async def get_point_downscaled_forecast(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
    district: str = Query(default="Jaipur"),
    hours: int = Query(default=48, ge=6, le=168)
) -> Dict[str, Any]:
    """
    Execute 1km downscaling for any geographic point.
    """
    from services.agent.tools.downscaler_tool import run_1km_downscaler
    from services.agent.tools.indices_tool import calculate_microclimate_indices
    
    downscale_data = run_1km_downscaler(latitude, longitude, forecast_hours=hours, district=district)
    indices_data = calculate_microclimate_indices(downscale_data["hourly_series"])
    
    return {
        "status": "success",
        "latitude": latitude,
        "longitude": longitude,
        "district": district,
        "zone_id": downscale_data["zone_id"],
        "mean_delta_t_c": downscale_data["mean_delta_t_c"],
        "hours_count": len(downscale_data["hourly_series"]),
        "indices": indices_data["indices"],
        "hourly_forecast": downscale_data["hourly_series"]
    }


@router.get("/{panchayat_id}/mosdac")
async def get_panchayat_mosdac(panchayat_id: str):
    """
    Retrieve MOSDAC INSAT-3DR satellite telemetry.
    """
    panchayat = get_panchayat(panchayat_id)
    if not panchayat:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"success": False, "error": f"Panchayat '{panchayat_id}' not found"}
        )
        
    mosdac_data = fetch_mosdac_telemetry(panchayat.panchayat_id)
    data_dict = mosdac_data.model_dump(mode="json") if hasattr(mosdac_data, "model_dump") else mosdac_data
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success": True,
            "panchayat_id": panchayat.panchayat_id,
            "data": data_dict
        }
    )


