"""
@file test_multiyear_dataset.py
@description Rigorous QA and integrity tests for the 10-year multi-target downscaling dataset.
@module tests
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PARQUET_PATH = RAW_DATA_DIR / "jaipur_10yr_training_data.parquet"
MANIFEST_PATH = RAW_DATA_DIR / "dataset_10yr_manifest.json"


@pytest.fixture(scope="module")
def manifest_data() -> dict:
    """Load dataset manifest."""
    assert MANIFEST_PATH.exists(), f"Manifest file missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def dataset_df() -> pd.DataFrame:
    """Load 10-year parquet dataset."""
    assert PARQUET_PATH.exists(), f"Parquet dataset missing: {PARQUET_PATH}"
    return pd.read_parquet(PARQUET_PATH)


def test_manifest_metadata(manifest_data: dict) -> None:
    """Verify manifest records 10 years, 36 spatial points, and >3M rows."""
    assert manifest_data["temporal_range"]["total_years"] == 10
    assert manifest_data["temporal_range"]["start_date"] == "2015-01-01"
    assert manifest_data["temporal_range"]["end_date"] == "2024-12-31"
    assert manifest_data["total_spatial_points"] == 36
    assert manifest_data["total_hourly_rows"] == 3_156_192
    assert manifest_data["null_values"] == 0
    assert len(manifest_data["targets"]) == 4


def test_dataset_dimensions_and_zero_nulls(dataset_df: pd.DataFrame) -> None:
    """Verify exact row counts, point cardinality, and complete absence of NaNs."""
    assert len(dataset_df) == 3_156_192
    assert dataset_df["point_id"].nunique() == 36
    assert dataset_df.isnull().sum().sum() == 0


def test_multi_target_residuals_integrity(dataset_df: pd.DataFrame) -> None:
    """Verify all 4 target residual columns exist with near-zero spatial mean and expected variation."""
    target_cols = [
        "residual_anomaly_c",
        "residual_soil_moisture_m3m3",
        "residual_vpd_kpa",
        "residual_wind_speed_kmh",
    ]
    for col in target_cols:
        assert col in dataset_df.columns, f"Target column missing: {col}"
        mean_val = float(dataset_df[col].mean())
        std_val = float(dataset_df[col].std())
        # Domain residuals must center around 0.0
        assert abs(mean_val) < 0.05, f"Residual {col} mean ({mean_val}) is not centered near 0.0"
        assert std_val > 0.001, f"Residual {col} has near-zero variance ({std_val})"


def test_hydrological_memory_features(dataset_df: pd.DataFrame) -> None:
    """Verify hydrological memory and antecedent features are valid and non-negative."""
    # Antecedent Precipitation Index
    assert "antecedent_precipitation_7d_mm" in dataset_df.columns
    assert (dataset_df["antecedent_precipitation_7d_mm"] >= 0.0).all()

    # Soil Moisture Lag
    assert "soil_moisture_lag_24h_m3m3" in dataset_df.columns
    assert (dataset_df["soil_moisture_lag_24h_m3m3"] >= 0.0).all()
    assert (dataset_df["soil_moisture_lag_24h_m3m3"] <= 0.60).all()

    # Growing Degree Days
    assert "accumulated_gdd_c" in dataset_df.columns
    assert (dataset_df["accumulated_gdd_c"] >= 0.0).all()

    # Option A physical interaction terms
    assert "evaporative_demand" in dataset_df.columns
    assert "infiltration_potential" in dataset_df.columns
    assert "valley_dew_potential" in dataset_df.columns
    assert "ridge_wind_acceleration" in dataset_df.columns
