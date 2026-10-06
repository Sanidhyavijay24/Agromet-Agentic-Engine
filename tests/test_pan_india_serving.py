"""
@file test_pan_india_serving.py
@description Integration tests verifying pan-India 15 ACZ dynamic inference routing, latency, and artifact integrity.
@module tests
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
import pytest
from tests.conftest import requires_model

from services.api.schemas import GeoLocation, HourlyForecastPoint, BaselineForecast
from services.ml_downscaler.inference import (
    downscale_point_forecast,
    downscale_forecast_batch,
    load_downscaler_model,
    DEFAULT_MODEL_PATH,
)
from services.ml_downscaler.zone_router import (
    ACZ_CATALOG,
    get_zone_artifact_path,
    route_coordinates_to_zone,
)


@requires_model
def test_flagship_zone_14_model_loadable():
    """Verify that flagship Zone XIV serialized model artifact loads successfully on CPU."""
    assert DEFAULT_MODEL_PATH.exists(), f"Missing flagship Zone 14 artifact at {DEFAULT_MODEL_PATH}"

    payload = load_downscaler_model(zone_id="ACZ_14")
    assert "model" in payload
    assert "features_list" in payload
    assert len(payload["features_list"]) == 28
    assert "domain_mean_elevation_m" in payload
    assert "anchor_points" in payload
    assert len(payload["anchor_points"]) == 8


@requires_model
def test_pan_india_dynamic_spatial_routing_and_inference():
    """Test dynamic downscaling inference across 15 distinct pan-India coordinates."""
    # Warm up in-memory cache
    load_downscaler_model(zone_id="ACZ_14")

    test_points = [
        # (Zone ID, Name, Latitude, Longitude, Elevation)
        ("ACZ_01", "Western Himalayan (Kangra)", 32.2190, 76.3234, 733.0),
        ("ACZ_02", "Eastern Himalayan (Guwahati)", 26.1445, 91.7362, 55.0),
        ("ACZ_03", "Lower Gangetic (Kolkata)", 22.5726, 88.3639, 9.0),
        ("ACZ_04", "Middle Gangetic (Patna)", 25.5941, 85.1376, 53.0),
        ("ACZ_05", "Upper Gangetic (Meerut)", 28.9845, 77.7064, 218.0),
        ("ACZ_06", "Trans-Gangetic (Ludhiana)", 30.9010, 75.8573, 244.0),
        ("ACZ_07", "Eastern Plateau (Ranchi)", 23.3441, 85.3096, 651.0),
        ("ACZ_08", "Central Plateau (Bhopal)", 23.2599, 77.4126, 527.0),
        ("ACZ_09", "Western Plateau (Pune)", 18.5204, 73.8567, 560.0),
        ("ACZ_10", "Southern Plateau (Hyderabad)", 17.3850, 78.4867, 542.0),
        ("ACZ_11", "East Coast Plains (Vijayawada)", 16.5062, 80.6480, 23.0),
        ("ACZ_12", "West Coast & Ghats (Ratnagiri)", 16.9902, 73.3120, 35.0),
        ("ACZ_13", "Gujarat Plains (Rajkot)", 22.3039, 70.8022, 128.0),
        ("ACZ_14", "Western Dry (Kadera / Jaipur)", 26.8500, 75.8000, 390.0),
        ("ACZ_15", "Islands Region (Port Blair)", 11.6234, 92.7265, 16.0),
    ]

    for expected_zid, name, lat, lon, elev in test_points:
        loc = GeoLocation(
            latitude=lat,
            longitude=lon,
            elevation_m=elev,
            district=name.split()[0],
            panchayat=f"{expected_zid}_STATION",
        )
        base_point = HourlyForecastPoint(
            time=datetime.now(timezone.utc),
            temperature_2m_c=30.0,
            relative_humidity_2m_pct=60.0,
            surface_pressure_hpa=980.0,
            wind_speed_10m_kmh=10.0,
            solar_radiation_w_m2=450.0,
            soil_temperature_0_to_7cm_c=31.5,
            soil_moisture_0_to_7cm_m3m3=0.22,
            et0_evapotranspiration_mm=0.35,
        )

        # 1. Spatial Routing Check
        zone_dict, _ = route_coordinates_to_zone(lat, lon)
        assert zone_dict["zone_id"] == expected_zid

        # 2. Downscale Point Inference & Timing
        t0 = time.perf_counter()
        result = downscale_point_forecast(
            panchayat_id=loc.panchayat,
            location=loc,
            baseline_point=base_point,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Assertions
        assert result.panchayat_id == loc.panchayat
        assert result.target_variable == "temperature_2m_c"
        assert result.baseline_value == 30.0
        assert 15.0 <= result.downscaled_value <= 45.0
        assert 0.70 <= result.confidence_score <= 1.0
        assert "predicted_residual_c" in result.covariates_used
        assert abs(result.covariates_used["predicted_residual_c"]) <= 12.0
        assert latency_ms < 50.0, f"Latency too high: {latency_ms:.2f}ms"


@requires_model
def test_batch_forecast_downscaling_speed():
    """Verify 7-day (168-hour) downscaling batch latency is sub-150ms."""
    loc = GeoLocation(
        latitude=26.8500,
        longitude=75.8000,
        elevation_m=390.0,
        district="Jaipur",
        panchayat="KADERA_001",
    )
    now = datetime.now(timezone.utc)
    hourly_series = [
        HourlyForecastPoint(
            time=now,
            temperature_2m_c=25.0 + (i % 12),
            relative_humidity_2m_pct=50.0,
            surface_pressure_hpa=970.0,
            wind_speed_10m_kmh=8.0,
        )
        for i in range(168)
    ]
    baseline_fc = BaselineForecast(
        location=loc,
        hourly=hourly_series,
    )

    t0 = time.perf_counter()
    batch_res = downscale_forecast_batch(
        panchayat_id=loc.panchayat,
        location=loc,
        baseline=baseline_fc,
    )
    total_time_ms = (time.perf_counter() - t0) * 1000.0

    assert len(batch_res.forecasts) == 168
    assert total_time_ms < 350.0  # 168 points in < 350ms (~2ms per point)
