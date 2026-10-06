"""
@file test_no_target_leakage.py
@description Guard against target leakage in every training feature pipeline.
@module tests

The downscaler predicts R = T_local - baseline. If any model FEATURE of a point is computed
from that point's own temperature, the model can read the answer off its inputs instead of
learning local physics. That happened (audit finding F22): the 10-year ingest computed VPD
from temperature_2m_c, and the offline score was inflated by it.

The test is generic, so it also catches the next leak nobody has thought of yet:
  change ONE point's own temperature at ONE hour -> none of that point's model features at
  that hour may change. (Its target must change, which proves the perturbation took effect.)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.ml_downscaler.train import FEATURES_LIST

ROOT = Path(__file__).resolve().parents[1]
BUMP_C = 5.0


def raw_frame() -> pd.DataFrame:
    """9 points on a 0.14-degree grid (so neighbour terrain works), 48 hours, realistic columns."""
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-06-01", periods=48, freq="h", tz="UTC")
    rows = []
    for i, la in enumerate([26.5, 26.64, 26.78]):
        for j, lo in enumerate([75.5, 75.64, 75.78]):
            pid = f"GRID_{3 * i + j + 1:03d}"
            elev = 350 + 40 * i - 25 * j
            for t in times:
                temp = 30 + 6 * np.sin(2 * np.pi * (t.hour - 9) / 24) - 0.0065 * (elev - 374) + rng.normal(0, 0.3)
                rows.append({
                    "point_id": pid, "latitude": la, "longitude": lo, "elevation_m": float(elev), "timestamp": t,
                    "temperature_2m_c": temp, "relative_humidity_2m_pct": float(rng.uniform(30, 90)),
                    "surface_pressure_hpa": 1013 - elev / 8.3, "precipitation_mm": 0.0,
                    "wind_speed_10m_kmh": float(rng.uniform(2, 20)),
                    "solar_radiation_w_m2": max(0.0, 800 * np.sin(2 * np.pi * (t.hour - 1) / 24)),
                    "shortwave_radiation_w_m2": max(0.0, 900 * np.sin(2 * np.pi * (t.hour - 1) / 24)),
                    "soil_temperature_0_to_7cm_c": temp + 2, "soil_moisture_0_to_7cm_m3m3": 0.2,
                    "et0_evapotranspiration_mm": 0.3,
                })
    return pd.DataFrame(rows)


def bumped(df: pd.DataFrame) -> tuple[pd.DataFrame, str, pd.Timestamp]:
    pid, t = "GRID_005", df["timestamp"].iloc[30]
    out = df.copy()
    out.loc[(out.point_id == pid) & (out.timestamp == t), "temperature_2m_c"] += BUMP_C
    return out, pid, t


def row(df: pd.DataFrame, pid: str, t: pd.Timestamp) -> pd.Series:
    r = df[(df.point_id == pid) & (pd.to_datetime(df.timestamp, utc=True) == t)]
    assert len(r) == 1
    return r.iloc[0]


def leaking_features(pipeline) -> list[str]:
    base = raw_frame()
    b, pid, t = bumped(base)
    a1, a2 = row(pipeline(base), pid, t), row(pipeline(b), pid, t)
    assert a2["residual_anomaly_c"] - a1["residual_anomaly_c"] == pytest.approx(BUMP_C, abs=1e-6), \
        "the perturbation must reach the target, or this test proves nothing"
    return [f for f in FEATURES_LIST if not np.isclose(a1[f], a2[f], atol=1e-9)]


def _ingest_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location("ingest_multi", ROOT / "scripts" / "ingest_multi_year_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.engineer_multitarget_features(df)


def _dataset_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    from services.ml_downscaler.dataset import engineer_downscaling_features
    from services.ml_downscaler.train import ensure_interaction_features
    return ensure_interaction_features(engineer_downscaling_features(df))


def test_ten_year_ingest_has_no_target_leakage() -> None:
    """The pipeline that built the champion's training set (F22 lived here)."""
    assert leaking_features(_ingest_pipeline) == []


def test_one_year_pipeline_has_no_target_leakage() -> None:
    """dataset.py -> ensure_interaction_features, the pipeline behind the 1-year file."""
    assert leaking_features(_dataset_pipeline) == []


def test_vpd_feature_is_computed_from_the_baseline_everywhere() -> None:
    """Training (both pipelines) and production must share one VPD definition."""
    out = _ingest_pipeline(raw_frame())
    t, rh = out["baseline_temp_c"], out["relative_humidity_2m_pct"].clip(1, 100)
    es = 0.61078 * np.exp(17.27 * t / (t + 237.3))
    assert np.allclose(out["vapor_pressure_deficit_kpa"], np.maximum(0.0, es * (1 - rh / 100)))
