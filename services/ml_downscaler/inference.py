"""
@file inference.py
@description Hyperlocal microclimate downscaling inference engine emitting verified Pydantic payloads.
@module services/ml_downscaler
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import pickle
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
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

    logger = _LoguruCompatLogger("inference_downscaler")  # type: ignore[assignment]

import numpy as np
import pandas as pd

from services.api.schemas import (
    BaselineForecast,
    DownscaledForecast,
    DownscaledForecastBatch,
    GeoLocation,
    HourlyForecastPoint,
)

try:
    import joblib
    LOAD_FN = joblib.load
except ImportError:
    def LOAD_FN(path: Path | str) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

from services.ml_downscaler.zone_router import (
    get_zone_artifact_path,
    route_coordinates_to_zone,
)

DEFAULT_MODEL_PATH: Path = Path(__file__).resolve().parent / "artifacts" / "zones" / "residual_model_acz_14.joblib"
_ZONE_MODEL_CACHE: dict[str, dict[str, Any]] = {}


def load_downscaler_model(
    model_path: Path | str | None = None,
    zone_id: str | None = None,
) -> dict[str, Any]:
    """
    Load serialized model payload using a per-zone in-memory cache.

    Args:
        model_path: Optional custom path to model artifact.
        zone_id: Optional Agro-Climatic Zone ID (e.g. 'ACZ_01', 'ACZ_14').

    Returns:
        Dict containing model, features_list, metrics, and domain_mean_elevation_m.
    """
    if model_path:
        path = Path(model_path)
        cache_key = str(path.resolve())
    elif zone_id:
        path = get_zone_artifact_path(zone_id)
        if not path.exists():
            path = DEFAULT_MODEL_PATH
        cache_key = str(path.resolve())
    else:
        path = DEFAULT_MODEL_PATH
        cache_key = "default"

    if cache_key in _ZONE_MODEL_CACHE:
        return _ZONE_MODEL_CACHE[cache_key]

    if not path.exists():
        # Fallback to default model path if available
        if path != DEFAULT_MODEL_PATH and DEFAULT_MODEL_PATH.exists():
            path = DEFAULT_MODEL_PATH
        else:
            raise FileNotFoundError(
                f"Trained model artifact not found at {path}. "
                "Run 'python scripts/train_all_acz_models.py' or 'python services/ml_downscaler/train.py' to generate it."
            )

    logger.info("Loading downscaler model artifact from {}", path)
    payload = LOAD_FN(path)
    # Models trained on GPU keep device=cuda. Serving inputs are CPU arrays, and predicting
    # from a worker thread (FastAPI runs sync endpoints in a thread pool) with a CUDA booster
    # crashes the whole process on Windows, so always serve on CPU.
    model = payload.get("model") if isinstance(payload, dict) else None
    if hasattr(model, "set_params") and hasattr(model, "get_xgb_params"):
        model.set_params(device="cpu")

    _ZONE_MODEL_CACHE[cache_key] = payload
    return payload


def build_inference_feature_vector(
    location: GeoLocation,
    point: HourlyForecastPoint,
    domain_mean_elevation: float = 380.0,
    covariates: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Construct aligned 1-row feature vector from forecast and location telemetry.

    Args:
        location: Target panchayat GeoLocation (contains elevation_m).
        point: Baseline forecast hourly point.
        domain_mean_elevation: Regional reference elevation in metres.
        covariates: Optional additional sensor or satellite covariates.

    Returns:
        1-row pandas DataFrame matching FEATURES_LIST.
    """
    elevation_m = location.elevation_m if location.elevation_m is not None else domain_mean_elevation
    delta_elevation_m = elevation_m - domain_mean_elevation
    theoretical_lapse_delta_c = delta_elevation_m * -0.0065

    # Base weather
    baseline_temp_c = point.temperature_2m_c if point.temperature_2m_c is not None else 28.0
    relative_humidity = point.relative_humidity_2m_pct if point.relative_humidity_2m_pct is not None else 65.0
    surface_pressure = point.surface_pressure_hpa if point.surface_pressure_hpa is not None else 960.0
    wind_speed = point.wind_speed_10m_kmh if point.wind_speed_10m_kmh is not None else 10.0
    precipitation = point.precipitation_mm if point.precipitation_mm is not None else 0.0

    covs = covariates or {}
    solar_radiation = covs.get(
        "solar_radiation_w_m2",
        point.solar_radiation_w_m2 if getattr(point, "solar_radiation_w_m2", None) is not None else 0.0,
    )
    shortwave_radiation = covs.get(
        "shortwave_radiation_w_m2",
        point.shortwave_radiation_w_m2 if getattr(point, "shortwave_radiation_w_m2", None) is not None else 0.0,
    )
    soil_temp = covs.get(
        "soil_temperature_0_to_7cm_c",
        point.soil_temperature_0_to_7cm_c if getattr(point, "soil_temperature_0_to_7cm_c", None) is not None else baseline_temp_c,
    )
    soil_moist = covs.get(
        "soil_moisture_0_to_7cm_m3m3",
        point.soil_moisture_0_to_7cm_m3m3 if getattr(point, "soil_moisture_0_to_7cm_m3m3", None) is not None else 0.25,
    )
    et0 = covs.get(
        "et0_evapotranspiration_mm",
        point.et0_evapotranspiration_mm if getattr(point, "et0_evapotranspiration_mm", None) is not None else 0.1,
    )

    # Cyclical temporal
    ts = point.time
    hour = ts.hour
    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)
    doy = ts.timetuple().tm_yday
    doy_sin = math.sin(2 * math.pi * doy / 365.25)
    doy_cos = math.cos(2 * math.pi * doy / 365.25)

    # Physics-guided interactions
    # A. Vapor Pressure Deficit (VPD in kPa)
    rh_clipped = max(1.0, min(100.0, relative_humidity))
    es_kpa = 0.61078 * math.exp((17.27 * baseline_temp_c) / (baseline_temp_c + 237.3))
    ea_kpa = es_kpa * (rh_clipped / 100.0)
    vpd_kpa = max(0.0, es_kpa - ea_kpa)

    # B. Nocturnal Inversion Index
    valley_depth = max(0.0, -delta_elevation_m)
    wind_effective = max(0.5, wind_speed)
    solar_frac = max(0.0, min(1.0, solar_radiation / 1000.0))
    nocturnal_inversion = (valley_depth / wind_effective) * (1.0 - solar_frac)

    # C. Solar Heating Interaction
    diurnal_factor = max(0.0, -hour_cos)
    solar_heating = (max(0.0, solar_radiation) / 1000.0) * diurnal_factor

    # D. Thermal Inertia Lag Proxy
    thermal_lag = covs.get("thermal_inertia_lag_3h", 0.0)

    # E. Terrain Morphology (TPI, Slope, Aspect)
    # Use verified continuous DEM terrain reader with covariate overrides if supplied
    from services.ml_downscaler.dem_source import extract_terrain

    terrain_res = extract_terrain(location.latitude, location.longitude)
    tpi = covs.get(
        "topographic_position_index",
        terrain_res.get("tpi_2000m") if terrain_res and terrain_res.get("tpi_2000m") is not None else 0.0,
    )
    slope_deg = covs.get(
        "slope_magnitude_deg",
        terrain_res.get("slope_magnitude_deg") if terrain_res and terrain_res.get("slope_magnitude_deg") is not None else 0.5,
    )
    aspect_sin = covs.get(
        "aspect_sin",
        terrain_res.get("aspect_sin") if terrain_res and terrain_res.get("aspect_sin") is not None else 0.0,
    )
    aspect_cos = covs.get(
        "aspect_cos",
        terrain_res.get("aspect_cos") if terrain_res and terrain_res.get("aspect_cos") is not None else 1.0,
    )

    # F. Sloped Solar Insolation
    aspect_alignment = aspect_sin * hour_sin + aspect_cos * hour_cos
    slope_factor = math.sin(math.radians(slope_deg))
    sloped_solar = (solar_radiation / 1000.0) * (1.0 + slope_factor * aspect_alignment)

    # G. Soil-to-Air Thermal Gradient
    soil_air_thermal_gradient = soil_temp - baseline_temp_c

    # H. Latent Evaporative Cooling Potential
    latent_cooling = et0 * vpd_kpa

    feat_dict = {
        # Topographical & Elevation Relief
        "delta_elevation_m": delta_elevation_m,
        "theoretical_lapse_delta_c": theoretical_lapse_delta_c,
        "elevation_m": elevation_m,
        "topographic_position_index": tpi,
        "slope_magnitude_deg": slope_deg,
        "aspect_sin": aspect_sin,
        "aspect_cos": aspect_cos,
        # Atmospheric Baseline
        "baseline_temp_c": baseline_temp_c,
        "relative_humidity_2m_pct": relative_humidity,
        "surface_pressure_hpa": surface_pressure,
        "wind_speed_10m_kmh": wind_speed,
        "precipitation_mm": precipitation,
        # Solar Radiation & Agro
        "solar_radiation_w_m2": solar_radiation,
        "shortwave_radiation_w_m2": shortwave_radiation,
        "soil_temperature_0_to_7cm_c": soil_temp,
        "soil_moisture_0_to_7cm_m3m3": soil_moist,
        "et0_evapotranspiration_mm": et0,
        # Physics-Guided Interactions
        "vapor_pressure_deficit_kpa": vpd_kpa,
        "nocturnal_inversion_index": nocturnal_inversion,
        "solar_heating_interaction": solar_heating,
        "sloped_solar_insolation": sloped_solar,
        "thermal_inertia_lag_3h": thermal_lag,
        "soil_air_thermal_gradient": soil_air_thermal_gradient,
        "latent_cooling_potential": latent_cooling,
        # Cyclical Diurnal & Seasonal
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
    }
    return pd.DataFrame([feat_dict])


def downscale_point_forecast(
    panchayat_id: str,
    location: GeoLocation,
    baseline_point: HourlyForecastPoint,
    covariates: dict[str, float] | None = None,
    model_path: Path | str | None = None,
) -> DownscaledForecast:
    """
    Downscale a single hourly weather point for a specific panchayat location.

    Args:
        panchayat_id: Unique panchayat code or name (e.g. "CHAKSU_001").
        location: Resolved GeoLocation context with elevation.
        baseline_point: Coarse baseline HourlyForecastPoint.
        covariates: Optional extra covariates (solar, soil, ET0).
        model_path: Optional path override for model artifact.

    Returns:
        Validated DownscaledForecast Pydantic instance.
    """
    if model_path is None and location.latitude is not None and location.longitude is not None:
        zone_info, _ = route_coordinates_to_zone(location.latitude, location.longitude)
        payload = load_downscaler_model(zone_id=zone_info["zone_id"])
    else:
        payload = load_downscaler_model(model_path)

    model = payload["model"]
    domain_mean_elevation = float(payload.get("domain_mean_elevation_m", 380.0))
    features_list = payload["features_list"]

    X = build_inference_feature_vector(
        location=location,
        point=baseline_point,
        domain_mean_elevation=domain_mean_elevation,
        covariates=covariates,
    )
    X = X[features_list]

    # Predict residual anomaly R
    pred_residual = float(model.predict(X)[0])

    # Physical meteorological guardrails: clamp anomaly to [-12Â°C, +12Â°C]
    clamped_residual = max(-12.0, min(12.0, pred_residual))

    baseline_val = float(baseline_point.temperature_2m_c or 28.0)
    downscaled_val = round(baseline_val + clamped_residual, 2)

    # Compute physical confidence score based on residual magnitude
    # Lower residual variance -> higher confidence (clamped to [0.70, 0.99])
    conf_score = max(0.70, min(0.99, 1.0 - (abs(clamped_residual) / 30.0)))

    covariates_used = {
        "elevation_m": float(location.elevation_m or domain_mean_elevation),
        "delta_elevation_m": float(X["delta_elevation_m"].iloc[0]),
        "predicted_residual_c": round(clamped_residual, 3),
        "solar_radiation_w_m2": float(X["solar_radiation_w_m2"].iloc[0]),
        "soil_temperature_0_to_7cm_c": float(X["soil_temperature_0_to_7cm_c"].iloc[0]),
        "soil_moisture_0_to_7cm_m3m3": float(X["soil_moisture_0_to_7cm_m3m3"].iloc[0]),
        "et0_evapotranspiration_mm": float(X["et0_evapotranspiration_mm"].iloc[0]),
        "thermal_inertia_lag_3h": float(X["thermal_inertia_lag_3h"].iloc[0]),
    }
    if covariates:
        covariates_used.update(covariates)

    return DownscaledForecast(
        panchayat_id=panchayat_id,
        timestamp=baseline_point.time,
        target_variable="temperature_2m_c",
        baseline_value=round(baseline_val, 2),
        downscaled_value=downscaled_val,
        confidence_score=round(conf_score, 3),
        covariates_used=covariates_used,
    )


def downscale_forecast_batch(
    panchayat_id: str,
    location: GeoLocation,
    baseline: BaselineForecast,
    covariates: dict[str, float] | None = None,
    model_path: Path | str | None = None,
) -> DownscaledForecastBatch:
    """
    Downscale all hourly points in a 7-day BaselineForecast for a panchayat.
    Vectorized across all time steps for sub-20ms batch inference speed.

    Args:
        panchayat_id: Unique panchayat code or name.
        location: Resolved GeoLocation context.
        baseline: BaselineForecast container with hourly points.
        covariates: Optional extra covariates map.
        model_path: Optional model artifact path override.

    Returns:
        Validated DownscaledForecastBatch Pydantic instance.
    """
    if not baseline.hourly:
        return DownscaledForecastBatch(
            panchayat_id=panchayat_id,
            location=location,
            forecasts=[],
            generated_at=datetime.now(timezone.utc),
        )

    # 1. Resolve Zone Model once for the entire batch
    if model_path is None and location.latitude is not None and location.longitude is not None:
        zone_info, _ = route_coordinates_to_zone(location.latitude, location.longitude)
        payload = load_downscaler_model(zone_id=zone_info["zone_id"])
    else:
        payload = load_downscaler_model(model_path)

    model = payload["model"]
    domain_mean_elevation = float(payload.get("domain_mean_elevation_m", 380.0))
    features_list = payload["features_list"]

    # 2. Extract terrain features once for the location
    from services.ml_downscaler.dem_source import extract_terrain

    terrain_res = extract_terrain(location.latitude, location.longitude)
    tpi_base = terrain_res.get("tpi_2000m") if terrain_res and terrain_res.get("tpi_2000m") is not None else 0.0
    slope_base = terrain_res.get("slope_magnitude_deg") if terrain_res and terrain_res.get("slope_magnitude_deg") is not None else 0.5
    aspect_sin_base = terrain_res.get("aspect_sin") if terrain_res and terrain_res.get("aspect_sin") is not None else 0.0
    aspect_cos_base = terrain_res.get("aspect_cos") if terrain_res and terrain_res.get("aspect_cos") is not None else 1.0

    elevation_m = float(location.elevation_m if location.elevation_m is not None else domain_mean_elevation)
    delta_elevation_m = elevation_m - domain_mean_elevation
    theoretical_lapse_delta_c = -(delta_elevation_m / 100.0) * 0.65
    valley_depth = max(0.0, -delta_elevation_m)

    # 3. Build vectorized feature rows
    rows: list[dict[str, float]] = []
    base_temps: list[float] = []
    times: list[datetime] = []

    for idx, pt in enumerate(baseline.hourly):
        times.append(pt.time)
        covs = dict(covariates or {})

        # Compute 3-hour lag
        if "thermal_inertia_lag_3h" not in covs:
            if idx >= 3 and baseline.hourly[idx - 3].temperature_2m_c is not None and pt.temperature_2m_c is not None:
                thermal_lag = float(pt.temperature_2m_c - baseline.hourly[idx - 3].temperature_2m_c)
            elif idx > 0 and baseline.hourly[0].temperature_2m_c is not None and pt.temperature_2m_c is not None:
                thermal_lag = float(pt.temperature_2m_c - baseline.hourly[0].temperature_2m_c)
            else:
                thermal_lag = 0.0
        else:
            thermal_lag = covs["thermal_inertia_lag_3h"]

        # Time harmonic encodings
        t_utc = pt.time if pt.time.tzinfo else pt.time.replace(tzinfo=timezone.utc)
        hour_angle = 2.0 * math.pi * (t_utc.hour + t_utc.minute / 60.0) / 24.0
        doy_angle = 2.0 * math.pi * t_utc.timetuple().tm_yday / 365.25
        hour_sin = math.sin(hour_angle)
        hour_cos = math.cos(hour_angle)
        doy_sin = math.sin(doy_angle)
        doy_cos = math.cos(doy_angle)

        baseline_temp_c = float(pt.temperature_2m_c if pt.temperature_2m_c is not None else 28.0)
        base_temps.append(baseline_temp_c)
        relative_humidity = float(pt.relative_humidity_2m_pct if pt.relative_humidity_2m_pct is not None else 60.0)
        surface_pressure = float(pt.surface_pressure_hpa if pt.surface_pressure_hpa is not None else 980.0)
        wind_speed = float(pt.wind_speed_10m_kmh if pt.wind_speed_10m_kmh is not None else 10.0)
        precipitation = float(pt.precipitation_mm if pt.precipitation_mm is not None else 0.0)

        solar_radiation = float(pt.solar_radiation_w_m2 or covs.get("solar_radiation_w_m2", 0.0))
        shortwave_radiation = float(pt.shortwave_radiation_w_m2 or covs.get("shortwave_radiation_w_m2", solar_radiation))
        soil_temp = float(pt.soil_temperature_0_to_7cm_c or covs.get("soil_temperature_0_to_7cm_c", baseline_temp_c))
        soil_moist = float(pt.soil_moisture_0_to_7cm_m3m3 or covs.get("soil_moisture_0_to_7cm_m3m3", 0.20))
        et0 = float(pt.et0_evapotranspiration_mm or covs.get("et0_evapotranspiration_mm", 0.25))

        # Atmospheric and terrain physics
        rh_clipped = max(5.0, min(100.0, relative_humidity))
        es_kpa = 0.61078 * math.exp((17.27 * baseline_temp_c) / (baseline_temp_c + 237.3))
        ea_kpa = es_kpa * (rh_clipped / 100.0)
        vpd_kpa = max(0.0, es_kpa - ea_kpa)

        wind_effective = max(0.5, wind_speed)
        solar_frac = max(0.0, min(1.0, solar_radiation / 1000.0))
        nocturnal_inversion = (valley_depth / wind_effective) * (1.0 - solar_frac)

        diurnal_factor = max(0.0, -hour_cos)
        solar_heating = (max(0.0, solar_radiation) / 1000.0) * diurnal_factor

        tpi = covs.get("topographic_position_index", tpi_base)
        slope_deg = covs.get("slope_magnitude_deg", slope_base)
        aspect_sin = covs.get("aspect_sin", aspect_sin_base)
        aspect_cos = covs.get("aspect_cos", aspect_cos_base)

        aspect_alignment = aspect_sin * hour_sin + aspect_cos * hour_cos
        slope_factor = math.sin(math.radians(slope_deg))
        sloped_solar = (solar_radiation / 1000.0) * (1.0 + slope_factor * aspect_alignment)

        soil_air_thermal_gradient = soil_temp - baseline_temp_c
        latent_cooling = et0 * vpd_kpa

        feat_dict = {
            "delta_elevation_m": delta_elevation_m,
            "theoretical_lapse_delta_c": theoretical_lapse_delta_c,
            "elevation_m": elevation_m,
            "topographic_position_index": tpi,
            "slope_magnitude_deg": slope_deg,
            "aspect_sin": aspect_sin,
            "aspect_cos": aspect_cos,
            "baseline_temp_c": baseline_temp_c,
            "relative_humidity_2m_pct": relative_humidity,
            "surface_pressure_hpa": surface_pressure,
            "wind_speed_10m_kmh": wind_speed,
            "precipitation_mm": precipitation,
            "solar_radiation_w_m2": solar_radiation,
            "shortwave_radiation_w_m2": shortwave_radiation,
            "soil_temperature_0_to_7cm_c": soil_temp,
            "soil_moisture_0_to_7cm_m3m3": soil_moist,
            "et0_evapotranspiration_mm": et0,
            "vapor_pressure_deficit_kpa": vpd_kpa,
            "nocturnal_inversion_index": nocturnal_inversion,
            "solar_heating_interaction": solar_heating,
            "sloped_solar_insolation": sloped_solar,
            "thermal_inertia_lag_3h": thermal_lag,
            "soil_air_thermal_gradient": soil_air_thermal_gradient,
            "latent_cooling_potential": latent_cooling,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "doy_sin": doy_sin,
            "doy_cos": doy_cos,
        }
        rows.append(feat_dict)

    # 4. Predict batch residuals in a single vectorized matrix call
    X_batch = pd.DataFrame(rows)[features_list]
    pred_residuals = model.predict(X_batch)

    forecasts: list[DownscaledForecast] = []
    for idx, (t, base_val, pred_res, r_dict) in enumerate(zip(times, base_temps, pred_residuals, rows)):
        clamped_res = max(-12.0, min(12.0, float(pred_res)))
        downscaled_val = round(base_val + clamped_res, 2)
        conf_score = max(0.70, min(0.99, 1.0 - (abs(clamped_res) / 30.0)))

        covs_used = {
            "elevation_m": elevation_m,
            "delta_elevation_m": delta_elevation_m,
            "predicted_residual_c": round(clamped_res, 3),
            "solar_radiation_w_m2": r_dict["solar_radiation_w_m2"],
            "soil_temperature_0_to_7cm_c": r_dict["soil_temperature_0_to_7cm_c"],
            "soil_moisture_0_to_7cm_m3m3": r_dict["soil_moisture_0_to_7cm_m3m3"],
            "et0_evapotranspiration_mm": r_dict["et0_evapotranspiration_mm"],
            "thermal_inertia_lag_3h": r_dict["thermal_inertia_lag_3h"],
        }
        forecasts.append(
            DownscaledForecast(
                panchayat_id=panchayat_id,
                timestamp=t,
                target_variable="temperature_2m_c",
                baseline_value=round(base_val, 2),
                downscaled_value=downscaled_val,
                confidence_score=round(conf_score, 3),
                covariates_used=covs_used,
            )
        )

    return DownscaledForecastBatch(
        panchayat_id=panchayat_id,
        location=location,
        forecasts=forecasts,
        generated_at=datetime.now(timezone.utc),
    )


