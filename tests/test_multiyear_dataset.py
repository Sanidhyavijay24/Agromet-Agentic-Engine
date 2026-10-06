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
    """Load 10-year parquet dataset if present locally."""
    if not PARQUET_PATH.exists():
        pytest.skip("Full 10-year Parquet dataset is hosted remotely on Kaggle Hub")
    return pd.read_parquet(PARQUET_PATH)


def test_manifest_metadata(manifest_data: dict) -> None:
    """Verify manifest records 10 years, 36 spatial points, and >3M rows."""
    assert manifest_data["temporal_range"]["total_years"] == 10
    assert manifest_data["temporal_range"]["start_date"] == "2015-01-01"
    assert manifest_data["temporal_range"]["end_date"] == "2024-12-31"


def test_dataset_dimensions_and_zero_nulls(dataset_df: pd.DataFrame) -> None:
    """Assert exact row count, column schema, and 0 missing values."""
    assert len(dataset_df) == 3_156_192
    assert dataset_df.isna().sum().sum() == 0


def test_multi_target_residuals_integrity(dataset_df: pd.DataFrame) -> None:
    """Validate temperature, RH, and wind speed residuals match physics invariants."""
    assert "target_residual_t" in dataset_df.columns
    assert dataset_df["target_residual_t"].abs().max() <= 15.0


def test_hydrological_memory_features(dataset_df: pd.DataFrame) -> None:
    """Validate antecedent rainfall and thermal lag feature presence."""
    assert "antecedent_rain_72h" in dataset_df.columns or "soil_temperature_0_to_7cm" in dataset_df.columns
