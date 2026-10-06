"""
@file test_downscaler.py
@description Unit tests for ML downscaler training, physical guardrails, and inference schemas.
@module tests
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.api.schemas import (
    BaselineForecast,
    DownscaledForecast,
    DownscaledForecastBatch,
    GeoLocation,
    HourlyForecastPoint,
)
from services.ml_downscaler.inference import (
    build_inference_feature_vector,
    downscale_forecast_batch,
    downscale_point_forecast,
    load_downscaler_model,
)
from services.ml_downscaler.train import (
    FEATURES_LIST,
    spatial_block_split,
    spatial_three_way_split,
    train_residual_model,
)


@pytest.fixture(scope="session")
def sample_dataset() -> pd.DataFrame:
    """Generate deterministic synthetic dataset for fast test execution."""
    np.random.seed(42)
    n_rows = 500
    timestamps = pd.date_range("2024-01-01", periods=n_rows, freq="h", tz="UTC")
    
    points = [f"GRID_{i:03d}" for i in range(1, 11)]
    records = []
    
    for i in range(n_rows):
        pt = points[i % len(points)]
        elev = 320.0 + (i % len(points)) * 15.0
        base_t = 25.0 + 5.0 * np.sin(i / 12.0)
        actual_t = base_t + (elev - 380.0) * -0.0065 + np.random.normal(0, 0.5)
        
        records.append(
            {
                "point_id": pt,
                "latitude": 26.5 + (i % 5) * 0.1,
                "longitude": 75.5 + (i % 5) * 0.1,
                "elevation_m": elev,
                "timestamp": timestamps[i],
                "temperature_2m_c": actual_t,
                "relative_humidity_2m_pct": 60 + int(10 * np.sin(i / 24.0)),
                "surface_pressure_hpa": 960.0,
                "precipitation_mm": 0.0,
                "wind_speed_10m_kmh": 10.0,
                "solar_radiation_w_m2": max(0.0, 800.0 * np.sin((i % 24) * np.pi / 24.0)),
                "shortwave_radiation_w_m2": max(0.0, 750.0 * np.sin((i % 24) * np.pi / 24.0)),
                "soil_temperature_0_to_7cm_c": actual_t + 1.0,
                "soil_moisture_0_to_7cm_m3m3": 0.25,
                "et0_evapotranspiration_mm": 0.2,
                "delta_elevation_m": elev - 380.0,
                "theoretical_lapse_delta_c": (elev - 380.0) * -0.0065,
                "baseline_temp_c": base_t,
                "residual_anomaly_c": actual_t - base_t,
                "hour": timestamps[i].hour,
                "hour_sin": np.sin(2 * np.pi * timestamps[i].hour / 24.0),
                "hour_cos": np.cos(2 * np.pi * timestamps[i].hour / 24.0),
                "day_of_year": timestamps[i].dayofyear,
                "doy_sin": np.sin(2 * np.pi * timestamps[i].dayofyear / 365.25),
                "doy_cos": np.cos(2 * np.pi * timestamps[i].dayofyear / 365.25),
            }
        )
    df = pd.DataFrame(records)
    from services.ml_downscaler.train import ensure_interaction_features
    return ensure_interaction_features(df)


def test_spatial_block_split(sample_dataset: pd.DataFrame) -> None:
    """Verify that train and test sets have zero spatial point overlap."""
    train_df, test_df, train_pts, test_pts = spatial_block_split(sample_dataset, test_fraction=0.20)
    
    assert len(train_pts) > 0
    assert len(test_pts) > 0
    # No overlap
    assert set(train_pts).isdisjoint(set(test_pts))
    assert len(train_df) + len(test_df) == len(sample_dataset)


def test_spatial_three_way_split(sample_dataset: pd.DataFrame) -> None:
    """Verify that train, validation, and test sets are mutually disjoint and cover all rows."""
    train_df, val_df, test_df, train_pts, val_pts, test_pts = spatial_three_way_split(
        sample_dataset, val_fraction=0.10, test_fraction=0.20
    )
    
    assert len(train_pts) > 0
    assert len(val_pts) > 0
    assert len(test_pts) > 0
    # Mutually disjoint
    assert set(train_pts).isdisjoint(set(val_pts))
    assert set(train_pts).isdisjoint(set(test_pts))
    assert set(val_pts).isdisjoint(set(test_pts))
    assert len(train_df) + len(val_df) + len(test_df) == len(sample_dataset)


def test_feature_vector_alignment() -> None:
    """Verify inference feature vector produces exactly the required columns."""
    loc = GeoLocation(latitude=26.9, longitude=75.8, elevation_m=420.0, district="Jaipur", panchayat="Chaksu")
    pt = HourlyForecastPoint(
        time=datetime(2024, 6, 1, 12, 0),
        temperature_2m_c=38.0,
        relative_humidity_2m_pct=40.0,
        surface_pressure_hpa=955.0,
        wind_speed_10m_kmh=15.0,
        precipitation_mm=0.0,
    )
    
    feat_df = build_inference_feature_vector(loc, pt, domain_mean_elevation=380.0)
    
    for f in FEATURES_LIST:
        assert f in feat_df.columns, f"Missing feature: {f}"
    assert len(feat_df) == 1


def test_model_training_and_quality_gate(sample_dataset: pd.DataFrame, tmp_path: Path) -> None:
    """Verify that model training executes and satisfies RMSE quality gate."""
    model, metrics = train_residual_model(df=sample_dataset, save_artifact=False)
    
    assert metrics["downscaled_rmse_c"] < metrics["baseline_rmse_c"]
    assert metrics["downscaled_mae_c"] < metrics["baseline_mae_c"]
    assert 0.0 <= metrics["downscaled_r2"] <= 1.0


def test_inference_physical_guardrails() -> None:
    """Ensure downscaling respects physical bounds (|R| <= 12°C) and returns valid Pydantic model."""
    loc = GeoLocation(latitude=26.9, longitude=75.8, elevation_m=650.0, district="Jaipur", panchayat="Chaksu")
    pt = HourlyForecastPoint(
        time=datetime(2024, 7, 10, 15, 0),
        temperature_2m_c=35.0,
        relative_humidity_2m_pct=50.0,
        surface_pressure_hpa=950.0,
        wind_speed_10m_kmh=10.0,
        precipitation_mm=0.0,
    )
    
    result = downscale_point_forecast("CHAKSU_TEST", loc, pt)
    
    assert isinstance(result, DownscaledForecast)
    assert result.panchayat_id == "CHAKSU_TEST"
    assert result.target_variable == "temperature_2m_c"
    assert 0.0 <= result.confidence_score <= 1.0
    
    # Residual anomaly must be strictly within [-12, +12]
    residual = result.downscaled_value - result.baseline_value
    assert -12.0 <= residual <= 12.0


def test_batch_downscaling() -> None:
    """Verify batch downscaling processes all hourly points correctly."""
    loc = GeoLocation(latitude=26.9, longitude=75.8, elevation_m=400.0, district="Jaipur", panchayat="Chaksu")
    pts = [
        HourlyForecastPoint(time=datetime(2024, 7, 1, h, 0), temperature_2m_c=25.0 + h * 0.5)
        for h in range(5)
    ]
    baseline = BaselineForecast(location=loc, hourly=pts)
    
    batch = downscale_forecast_batch("CHAKSU_BATCH", loc, baseline)
    
    assert isinstance(batch, DownscaledForecastBatch)
    assert len(batch.forecasts) == 5
    for f in batch.forecasts:
        assert isinstance(f, DownscaledForecast)


def test_spatial_cross_validation(sample_dataset: pd.DataFrame) -> None:
    """Verify that 3-fold spatial cross-validation runs cleanly and computes fold statistics."""
    from services.ml_downscaler.train import run_spatial_cross_validation
    cv_res = run_spatial_cross_validation(df=sample_dataset, n_splits=3, use_gpu=False)
    
    assert len(cv_res["fold_results"]) == 3
    assert cv_res["mean_downscaled_rmse"] > 0.0
    assert -100.0 <= cv_res["mean_rmse_improvement_pct"] <= 100.0


def test_inference_covariates_propagation() -> None:
    """Verify that solar, soil, and ET0 covariates propagate correctly to feature vector and result."""
    loc = GeoLocation(latitude=26.9, longitude=75.8, elevation_m=420.0, district="Jaipur", panchayat="Chaksu")
    pt = HourlyForecastPoint(
        time=datetime(2024, 6, 1, 12, 0),
        temperature_2m_c=38.0,
        relative_humidity_2m_pct=40.0,
        surface_pressure_hpa=955.0,
        wind_speed_10m_kmh=15.0,
        precipitation_mm=0.0,
        solar_radiation_w_m2=750.0,
        shortwave_radiation_w_m2=700.0,
        soil_temperature_0_to_7cm_c=42.0,
        soil_moisture_0_to_7cm_m3m3=0.18,
        et0_evapotranspiration_mm=0.45,
    )
    
    feat_df = build_inference_feature_vector(loc, pt, domain_mean_elevation=380.0)
    assert feat_df["solar_radiation_w_m2"].iloc[0] == 750.0
    assert feat_df["shortwave_radiation_w_m2"].iloc[0] == 700.0
    assert feat_df["soil_temperature_0_to_7cm_c"].iloc[0] == 42.0
    assert feat_df["soil_moisture_0_to_7cm_m3m3"].iloc[0] == 0.18
    assert feat_df["et0_evapotranspiration_mm"].iloc[0] == 0.45
    assert feat_df["soil_air_thermal_gradient"].iloc[0] == 4.0  # 42.0 - 38.0
    
    result = downscale_point_forecast("KADERA_COV", loc, pt)
    assert result.covariates_used["solar_radiation_w_m2"] == 750.0
    assert result.covariates_used["soil_temperature_0_to_7cm_c"] == 42.0
    assert result.covariates_used["soil_moisture_0_to_7cm_m3m3"] == 0.18
    assert result.covariates_used["et0_evapotranspiration_mm"] == 0.45


def test_leave_one_out_baseline_calculation() -> None:
    """Verify that Leave-One-Out baseline excludes the current point from regional mean."""
    from services.ml_downscaler.dataset import engineer_downscaling_features
    
    # Create test data with 3 points at 1 timestamp: temps = 20, 30, 40
    raw_df = pd.DataFrame([
        {
            "point_id": "P1",
            "latitude": 26.5,
            "longitude": 75.5,
            "elevation_m": 300.0,
            "timestamp": "2024-05-01 12:00:00",
            "temperature_2m_c": 20.0,
            "relative_humidity_2m_pct": 50.0,
            "surface_pressure_hpa": 960.0,
            "precipitation_mm": 0.0,
            "wind_speed_10m_kmh": 10.0,
            "solar_radiation_w_m2": 800.0,
        },
        {
            "point_id": "P2",
            "latitude": 26.6,
            "longitude": 75.6,
            "elevation_m": 350.0,
            "timestamp": "2024-05-01 12:00:00",
            "temperature_2m_c": 30.0,
            "relative_humidity_2m_pct": 50.0,
            "surface_pressure_hpa": 960.0,
            "precipitation_mm": 0.0,
            "wind_speed_10m_kmh": 10.0,
            "solar_radiation_w_m2": 800.0,
        },
        {
            "point_id": "P3",
            "latitude": 26.7,
            "longitude": 75.7,
            "elevation_m": 400.0,
            "timestamp": "2024-05-01 12:00:00",
            "temperature_2m_c": 40.0,
            "relative_humidity_2m_pct": 50.0,
            "surface_pressure_hpa": 960.0,
            "precipitation_mm": 0.0,
            "wind_speed_10m_kmh": 10.0,
            "solar_radiation_w_m2": 800.0,
        },
    ])
    
    res = engineer_downscaling_features(raw_df)
    p1_base = res[res["point_id"] == "P1"]["baseline_temp_c"].iloc[0]
    p2_base = res[res["point_id"] == "P2"]["baseline_temp_c"].iloc[0]
    p3_base = res[res["point_id"] == "P3"]["baseline_temp_c"].iloc[0]
    
    # LOO baseline for P1 (20) should be mean(30, 40) = 35.0
    # LOO baseline for P2 (30) should be mean(20, 40) = 30.0
    # LOO baseline for P3 (40) should be mean(20, 30) = 25.0
    assert p1_base == 35.0
    assert p2_base == 30.0
    assert p3_base == 25.0
    
    # Residuals: R = T_local - T_baseline
    assert res[res["point_id"] == "P1"]["residual_anomaly_c"].iloc[0] == -15.0  # 20 - 35
    assert res[res["point_id"] == "P2"]["residual_anomaly_c"].iloc[0] == 0.0    # 30 - 30
    assert res[res["point_id"] == "P3"]["residual_anomaly_c"].iloc[0] == 15.0   # 40 - 25


def test_thermal_inertia_lag_and_batch_propagation() -> None:
    """Verify 3-hour thermal inertia lag calculation and propagation in forecast batches."""
    from services.api.routes.forecast import compute_thermal_inertia_lag
    
    times = pd.date_range("2024-06-01 00:00:00", periods=6, freq="h", tz="UTC")
    temps = [24.0, 25.0, 27.0, 30.0, 34.0, 37.0]  # Warming morning curve
    
    hourly_pts = [
        HourlyForecastPoint(
            time=t,
            temperature_2m_c=temp,
            relative_humidity_2m_pct=50.0,
            surface_pressure_hpa=960.0,
            wind_speed_10m_kmh=10.0,
            solar_radiation_w_m2=500.0,
            shortwave_radiation_w_m2=500.0,
        )
        for t, temp in zip(times, temps)
    ]
    
    # Check compute_thermal_inertia_lag helper directly
    # At t=0 (24°C), lag is 0.0
    assert compute_thermal_inertia_lag(hourly_pts, hourly_pts[0]) == 0.0
    # At t=3 (30°C vs t=0 24°C), lag is 30 - 24 = 6.0°C
    assert compute_thermal_inertia_lag(hourly_pts, hourly_pts[3]) == 6.0
    # At t=4 (34°C vs t=1 25°C), lag is 34 - 25 = 9.0°C
    assert compute_thermal_inertia_lag(hourly_pts, hourly_pts[4]) == 9.0
    
    # Test batch downscaling dynamically calculates and populates lag
    loc = GeoLocation(latitude=26.9, longitude=75.8, elevation_m=400.0, district="Jaipur", panchayat="Kadera")
    baseline_fc = BaselineForecast(
        location=loc,
        hourly=hourly_pts,
        daily_max_temp=[37.0],
        daily_min_temp=[24.0],
        daily_rain_sum=[0.0],
        fetched_at=datetime.now(),
        source="TEST",
    )
    
    batch = downscale_forecast_batch("KADERA_001", loc, baseline_fc)
    assert len(batch.forecasts) == 6
    assert batch.forecasts[3].covariates_used["thermal_inertia_lag_3h"] == 6.0
    assert batch.forecasts[4].covariates_used["thermal_inertia_lag_3h"] == 9.0
    assert batch.forecasts[3].covariates_used["solar_radiation_w_m2"] == 500.0



