"""
@file run_all_benchmarks.py
@description Live benchmark evaluation harness reproducing every single model architecture on the 2024 Spatial Holdout Split.
@module scripts
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.ml_downscaler.train import (
    load_training_dataset,
    spatial_three_way_split,
    FEATURES_LIST,
    TARGET_COLUMN,
)


def evaluate_model(
    name: str,
    desc: str,
    y_true_actual: np.ndarray,
    t_baseline: np.ndarray,
    pred_residuals: np.ndarray,
    train_time_sec: float,
    num_features: int,
) -> dict[str, Any]:
    pred_temp = t_baseline + pred_residuals
    b_mae = float(mean_absolute_error(y_true_actual, t_baseline))
    b_rmse = float(np.sqrt(mean_squared_error(y_true_actual, t_baseline)))
    b_r2 = float(r2_score(y_true_actual, t_baseline))

    m_mae = float(mean_absolute_error(y_true_actual, pred_temp))
    m_rmse = float(np.sqrt(mean_squared_error(y_true_actual, pred_temp)))
    m_r2 = float(r2_score(y_true_actual, pred_temp))

    rmse_drop_pct = ((b_rmse - m_rmse) / b_rmse) * 100.0
    mae_drop_pct = ((b_mae - m_mae) / b_mae) * 100.0

    return {
        "name": name,
        "desc": desc,
        "features": num_features,
        "mae_c": m_mae,
        "rmse_c": m_rmse,
        "r2": m_r2,
        "rmse_drop_pct": rmse_drop_pct,
        "mae_drop_pct": mae_drop_pct,
        "train_time_sec": train_time_sec,
    }


def main() -> None:
    print("================================================================================")
    print("    SIH 26074: MASTER MULTI-MODEL BENCHMARK AUDIT (SPATIAL HOLDOUT 2024)       ")
    print("================================================================================")

    df = load_training_dataset()
    train_df, val_df, test_df, train_pts, val_pts, test_pts = spatial_three_way_split(df)

    t_actual = test_df["temperature_2m_c"].to_numpy()
    t_base = test_df["baseline_temp_c"].to_numpy()
    delta_z = test_df["delta_elevation_m"].to_numpy()

    # Feature subsets
    f17 = [
        "delta_elevation_m", "theoretical_lapse_delta_c", "elevation_m",
        "baseline_temp_c", "relative_humidity_2m_pct", "surface_pressure_hpa",
        "wind_speed_10m_kmh", "precipitation_mm", "solar_radiation_w_m2",
        "shortwave_radiation_w_m2", "soil_temperature_0_to_7cm_c",
        "soil_moisture_0_to_7cm_m3m3", "et0_evapotranspiration_mm",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos"
    ]
    f21 = f17 + [
        "vapor_pressure_deficit_kpa", "nocturnal_inversion_index",
        "solar_heating_interaction", "thermal_inertia_lag_3h"
    ]
    f28 = FEATURES_LIST

    results = []

    # 1. Baseline
    b_mae = float(mean_absolute_error(t_actual, t_base))
    b_rmse = float(np.sqrt(mean_squared_error(t_actual, t_base)))
    b_r2 = float(r2_score(t_actual, t_base))
    results.append({
        "name": "Coarse Baseline",
        "desc": "Spatial Mean Baseline (No ML)",
        "features": 0,
        "mae_c": b_mae,
        "rmse_c": b_rmse,
        "r2": b_r2,
        "rmse_drop_pct": 0.0,
        "mae_drop_pct": 0.0,
        "train_time_sec": 0.0,
    })

    # 2. Exp 04: Standard Lapse Rate
    t0 = time.perf_counter()
    pred_res_04 = -0.0065 * delta_z
    t_lapse = time.perf_counter() - t0
    results.append(evaluate_model("Exp 04: Standard Lapse", "Standard Environmental Lapse (-6.5°C/km)", t_actual, t_base, pred_res_04, t_lapse, 1))

    # 3. Exp 04b: Fitted 1D Lapse Model
    lr = LinearRegression()
    t0 = time.perf_counter()
    lr.fit(train_df[["delta_elevation_m"]], train_df["residual_anomaly_c"])
    t_lr = time.perf_counter() - t0
    pred_res_04b = lr.predict(test_df[["delta_elevation_m"]])
    results.append(evaluate_model("Exp 04b: 1D Linear Lapse", "Fitted 1D Linear Regression", t_actual, t_base, pred_res_04b, t_lr, 1))

    # 4. Exp 05: Random Forest (300 trees, depth 14)
    rf = RandomForestRegressor(n_estimators=300, max_depth=14, max_features="sqrt", n_jobs=-1, random_state=42)
    t0 = time.perf_counter()
    rf.fit(train_df[f21], train_df["residual_anomaly_c"])
    t_rf = time.perf_counter() - t0
    pred_res_05 = rf.predict(test_df[f21])
    results.append(evaluate_model("Exp 05: Random Forest", "Random Forest (300 trees, depth 14)", t_actual, t_base, pred_res_05, t_rf, 21))

    # 5. Exp 01: Baseline XGBoost (300 trees, depth 6, lr 0.05)
    xgb_01 = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, tree_method="hist", random_state=42)
    t0 = time.perf_counter()
    xgb_01.fit(train_df[f17], train_df["residual_anomaly_c"])
    t_xgb_01 = time.perf_counter() - t0
    pred_res_01 = xgb_01.predict(test_df[f17])
    results.append(evaluate_model("Exp 01: Baseline XGBoost", "Depth GBDT (300 trees, depth 6, lr 0.05)", t_actual, t_base, pred_res_01, t_xgb_01, 17))

    # 6. Exp 02: Deepened XGBoost (1200 trees, depth 8, lr 0.03, ES 50 on Val Set)
    xgb_02 = xgb.XGBRegressor(n_estimators=1200, max_depth=8, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, early_stopping_rounds=50, tree_method="hist", random_state=42)
    t0 = time.perf_counter()
    xgb_02.fit(train_df[f21], train_df["residual_anomaly_c"], eval_set=[(train_df[f21], train_df["residual_anomaly_c"]), (val_df[f21], val_df["residual_anomaly_c"])], verbose=False)
    t_xgb_02 = time.perf_counter() - t0
    pred_res_02 = xgb_02.predict(test_df[f21])
    results.append(evaluate_model("Exp 02: Deepened XGBoost", "Depth GBDT (1200 trees, depth 8, lr 0.03)", t_actual, t_base, pred_res_02, t_xgb_02, 21))

    # 7. Exp 07: CatBoost (1200 trees, depth 8, lr 0.03)
    cb_07 = CatBoostRegressor(iterations=1200, depth=8, learning_rate=0.03, random_seed=42, verbose=0)
    t0 = time.perf_counter()
    cb_07.fit(train_df[f21], train_df["residual_anomaly_c"])
    t_cb_07 = time.perf_counter() - t0
    pred_res_07 = cb_07.predict(test_df[f21])
    results.append(evaluate_model("Exp 07: CatBoost", "Symmetric GBDT (1200 trees, depth 8)", t_actual, t_base, pred_res_07, t_cb_07, 21))

    # 8. Exp 06: LightGBM (1200 trees, num_leaves 63, lr 0.03)
    lgb_06 = lgb.LGBMRegressor(n_estimators=1200, num_leaves=63, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbose=-1)
    t0 = time.perf_counter()
    lgb_06.fit(train_df[f21], train_df["residual_anomaly_c"])
    t_lgb_06 = time.perf_counter() - t0
    pred_res_06 = lgb_06.predict(test_df[f21])
    results.append(evaluate_model("Exp 06: LightGBM", "Leaf-wise GBDT (1200 trees, leaves 63)", t_actual, t_base, pred_res_06, t_lgb_06, 21))

    # 9. Exp 08b: Track 2 LightGBM (1200 trees, num_leaves 63, lr 0.03, 28 features)
    lgb_08b = lgb.LGBMRegressor(n_estimators=1200, num_leaves=63, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbose=-1)
    t0 = time.perf_counter()
    lgb_08b.fit(train_df[f28], train_df["residual_anomaly_c"])
    t_lgb_08b = time.perf_counter() - t0
    pred_res_08b = lgb_08b.predict(test_df[f28])
    results.append(evaluate_model("Exp 08b: Track 2 LightGBM", "Leaf-wise GBDT (1200 trees, 28 feat)", t_actual, t_base, pred_res_08b, t_lgb_08b, 28))

    # 10. Exp 08: Track 2 Deepened XGBoost (Champion, 1200 trees, depth 8, 28 features, ES 50 on Val Set)
    xgb_08 = xgb.XGBRegressor(n_estimators=1200, max_depth=8, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, early_stopping_rounds=50, tree_method="hist", random_state=42)
    t0 = time.perf_counter()
    xgb_08.fit(train_df[f28], train_df["residual_anomaly_c"], eval_set=[(train_df[f28], train_df["residual_anomaly_c"]), (val_df[f28], val_df["residual_anomaly_c"])], verbose=False)
    t_xgb_08 = time.perf_counter() - t0
    pred_res_08 = xgb_08.predict(test_df[f28])
    results.append(evaluate_model("Exp 08: Track 2 XGBoost (🏆)", "Depth GBDT (1200 trees, 28 feat)", t_actual, t_base, pred_res_08, t_xgb_08, 28))

    print("\n" + "=" * 115)
    print(f"{'Experiment':<28} | {'Features':<8} | {'Test MAE':<10} | {'Test RMSE':<10} | {'Test R²':<8} | {'RMSE Drop':<10} | {'Train Time':<10}")
    print("-" * 115)
    for r in results:
        drop_str = f"+{r['rmse_drop_pct']:.1f}%" if r['rmse_drop_pct'] >= 0 else f"{r['rmse_drop_pct']:.1f}%"
        print(f"{r['name']:<28} | {r['features']:<8} | {r['mae_c']:.3f}°C     | {r['rmse_c']:.3f}°C     | {r['r2']:.4f}   | {drop_str:<10} | {r['train_time_sec']:.2f}s")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
