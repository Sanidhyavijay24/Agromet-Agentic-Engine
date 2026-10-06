"""
@file _common.py
@description Shared loading, labelling and scoring for the Track B audit scripts, so every
             script defines "RMSE gain" and "residual R2" identically.
@module scripts/audit
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUT_DIR = ROOT_DIR / "data" / "processed" / "audit"
IST_OFFSET_H = 5.5
# Open-Meteo's archive default is ERA5-Land before 2017 and ECMWF IFS HRES from 2017 on
# (verified year by year in b0_source_skew.py). Production is served by ECMWF IFS HRES.
IFS_ERA_START_YEAR = 2017

ELEVATION_GROUP = ["elevation_m", "delta_elevation_m", "theoretical_lapse_delta_c"]
TERRAIN_GROUP = ["topographic_position_index", "slope_magnitude_deg", "aspect_sin", "aspect_cos",
                 "sloped_solar_insolation"]
SOIL_GROUP = ["soil_temperature_0_to_7cm_c", "soil_moisture_0_to_7cm_m3m3", "soil_air_thermal_gradient"]
SOLAR_GROUP = ["solar_radiation_w_m2", "shortwave_radiation_w_m2", "solar_heating_interaction",
               "sloped_solar_insolation", "nocturnal_inversion_index"]


# ─────────────────────────────────────────────────────────────────────────────
#  loading
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(path: str | None = None) -> tuple[pd.DataFrame, str]:
    """Load the training parquet (10-year preferred) with every engineered feature present."""
    from services.ml_downscaler.train import ensure_interaction_features

    raw = ROOT_DIR / "data" / "raw"
    candidates = [Path(path)] if path else [
        raw / "jaipur_10yr_training_data_clean.parquet",
        raw / "jaipur_10yr_training_data_vpdfix.parquet",
        raw / "jaipur_10yr_training_data.parquet",
        raw / "jaipur_1yr_training_data.parquet",
    ]
    for p in candidates:
        p = p if p.is_absolute() else ROOT_DIR / p
        if p.exists():
            if "1yr" in p.name:
                print(f"! using {p.name}: the champion was trained on the 10-year file, so numbers")
                print("  will NOT match its stored metrics. Get the 10-year parquet from Lead ML.")
            t0 = time.perf_counter()
            df = pd.read_parquet(p)
            df = ensure_interaction_features(df)
            print(f"  loaded {p.name}: {len(df):,} rows, {df['point_id'].nunique()} points "
                  f"in {time.perf_counter() - t0:.1f}s")
            return label(df), p.name
    raise SystemExit(f"No training parquet found (looked for: {', '.join(str(c) for c in candidates)})")


def load_artifact() -> dict[str, Any]:
    from services.ml_downscaler.inference import load_downscaler_model
    return load_downscaler_model()


def label(df: pd.DataFrame) -> pd.DataFrame:
    """Add IST hour, month, IMD season, year and source era, for breakdowns."""
    ts = pd.DatetimeIndex(df["timestamp"])
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    ist = ts.tz_convert("Asia/Kolkata")
    df = df.copy()
    df["ist_hour"] = ist.hour
    df["ist_date"] = ist.date
    df["month"] = ist.month
    df["year"] = ist.year
    df["imd_season"] = pd.Categorical.from_codes(
        np.select([df["month"].isin([1, 2]), df["month"].isin([3, 4, 5]),
                   df["month"].between(6, 9)], [0, 1, 2], default=3),
        categories=["Winter", "Pre-monsoon", "SW monsoon", "Post-monsoon"])
    df["source_era"] = np.where(df["year"] >= IFS_ERA_START_YEAR, "ECMWF IFS", "ERA5-Land")
    df["daytime"] = df["ist_hour"].between(7, 18)
    return df


def split_by_points(df: pd.DataFrame, payload: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """The artifact's own train / val / test points, as recorded in its metrics."""
    m = payload.get("metrics", {})
    out = {}
    for name in ("train", "val", "test"):
        pts = m.get(f"{name}_points")
        if pts:
            out[name] = df[df["point_id"].isin(pts)]
    if "test" not in out:
        raise SystemExit("The artifact does not record its test points; cannot reproduce its split.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  scoring
# ─────────────────────────────────────────────────────────────────────────────


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum(np.square(y - pred)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def score(actual: np.ndarray, baseline: np.ndarray, pred_residual: np.ndarray) -> dict[str, float]:
    """
    Every metric the audit reports, from one place.

    temperature R2  -- what the project headlines (0.9986). Dominated by the diurnal and
                       seasonal cycle that the BASELINE already carries; nearly any model
                       scores ~0.99 on it.
    residual R2     -- the share of the local deviation (actual - baseline) the model explains.
                       This is the honest measure of what the downscaler contributes.
    """
    actual, baseline, pred_residual = (np.asarray(a, float) for a in (actual, baseline, pred_residual))
    true_res = actual - baseline
    down = baseline + pred_residual
    b_rmse, d_rmse = _rmse(actual - baseline), _rmse(actual - down)
    b_mae = float(np.mean(np.abs(actual - baseline))) if actual.size else float("nan")
    d_mae = float(np.mean(np.abs(actual - down))) if actual.size else float("nan")
    return {
        "rows": int(actual.size),
        "baseline_rmse": b_rmse,
        "downscaled_rmse": d_rmse,
        "rmse_gain_pct": 100.0 * (b_rmse - d_rmse) / b_rmse if b_rmse > 0 else float("nan"),
        "baseline_mae": b_mae,
        "downscaled_mae": d_mae,
        "mae_gain_pct": 100.0 * (b_mae - d_mae) / b_mae if b_mae > 0 else float("nan"),
        "temperature_r2": _r2(actual, down),
        "residual_r2": _r2(true_res, pred_residual),
        "residual_rmse": _rmse(true_res - pred_residual),
        # Mean (downscaled - actual). Per point or per band this is more telling than residual
        # R2: a single point's residual is mostly a constant offset with little variance, so
        # R2 there can be strongly negative even when the model gets the offset right.
        "bias": float(np.mean(down - actual)) if actual.size else float("nan"),
    }


def predict(model: Any, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.asarray(model.predict(df[features]), float)


def score_frame(df: pd.DataFrame, pred_residual: np.ndarray) -> dict[str, float]:
    return score(df["temperature_2m_c"].to_numpy(), df["baseline_temp_c"].to_numpy(), pred_residual)


# ─────────────────────────────────────────────────────────────────────────────
#  training (used only by training_experiments.py)
# ─────────────────────────────────────────────────────────────────────────────


def train_xgb(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    target: str = "residual_anomaly_c",
    n_estimators: int = 1200,
    max_depth: int = 8,
    learning_rate: float = 0.03,
    early_stopping_rounds: int = 50,
    gpu: bool = False,
    seed: int = 42,
) -> tuple[Any, dict[str, Any]]:
    """Train with the champion's hyperparameters unless told otherwise."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.85, colsample_bytree=0.85, early_stopping_rounds=early_stopping_rounds,
        tree_method="hist", device="cuda" if gpu else "cpu", random_state=seed, verbosity=0,
    )
    t0 = time.perf_counter()
    model.fit(train[features], train[target], eval_set=[(val[features], val[target])], verbose=False)
    history = model.evals_result().get("validation_0", {}).get("rmse", [])
    return model, {
        "seconds": time.perf_counter() - t0,
        "best_iteration": int(getattr(model, "best_iteration", n_estimators - 1)),
        "n_estimators": n_estimators,
        "val_rmse_curve": history,
        "hit_the_cap": int(getattr(model, "best_iteration", n_estimators - 1)) >= n_estimators - 1,
    }


def fmt_pct(x: float) -> str:
    return "   n/a" if x != x else f"{x:+6.1f}%"


def write_json(name: str, payload: dict[str, Any]) -> Path:
    import json

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
