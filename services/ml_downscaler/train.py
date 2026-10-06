"""
@file train.py
@description Model training pipeline for the XGBoost microclimate residual downscaler with Spatial K-Fold CV & physics features.
@module services/ml_downscaler
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import sys
import time
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

    logger = _LoguruCompatLogger("train_downscaler")  # type: ignore[assignment]

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    def mean_squared_error(y_true: Any, y_pred: Any) -> float:
        return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

    def r2_score(y_true: Any, y_pred: Any) -> float:
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        return float(1.0 - (ss_res / max(float(ss_tot), 1e-12)))

import xgboost as xgb

try:
    import joblib
    SAVE_FN = joblib.dump
    LOAD_FN = joblib.load
except ImportError:
    def SAVE_FN(obj: Any, path: Path | str) -> None:
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def LOAD_FN(path: Path | str) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

from services.ml_downscaler.dataset import engineer_downscaling_features

# Expanded 28-Feature Physics Schema (Track 2 Verified Champion)
FEATURES_LIST: list[str] = [
    # Topographical & Elevation Relief
    "delta_elevation_m",
    "theoretical_lapse_delta_c",
    "elevation_m",
    "topographic_position_index",
    "slope_magnitude_deg",
    "aspect_sin",
    "aspect_cos",
    # Atmospheric Baseline
    "baseline_temp_c",
    "relative_humidity_2m_pct",
    "surface_pressure_hpa",
    "wind_speed_10m_kmh",
    "precipitation_mm",
    # Solar Radiation & Agro
    "solar_radiation_w_m2",
    "shortwave_radiation_w_m2",
    "soil_temperature_0_to_7cm_c",
    "soil_moisture_0_to_7cm_m3m3",
    "et0_evapotranspiration_mm",
    # Physics-Guided Interactions
    "vapor_pressure_deficit_kpa",
    "nocturnal_inversion_index",
    "solar_heating_interaction",
    "sloped_solar_insolation",
    "thermal_inertia_lag_3h",
    "soil_air_thermal_gradient",
    "latent_cooling_potential",
    # Cyclical Diurnal & Seasonal
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

TARGET_COLUMN: str = "residual_anomaly_c"
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH: Path = ARTIFACTS_DIR / "residual_model.joblib"


def ensure_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all physics-guided interaction and terrain morphology features exist in dataframe."""
    missing = [f for f in FEATURES_LIST if f not in df.columns]
    if missing:
        logger.info("Computing missing interaction & terrain features: {}", missing)
        t_c = df["baseline_temp_c"] if "baseline_temp_c" in df.columns else df["temperature_2m_c"]
        rh_pct = df["relative_humidity_2m_pct"].clip(lower=1.0, upper=100.0)
        
        # A. Vapor Pressure Deficit (VPD in kPa) using Tetens equation
        if "vapor_pressure_deficit_kpa" not in df.columns:
            es_kpa = 0.61078 * np.exp((17.27 * t_c) / (t_c + 237.3))
            ea_kpa = es_kpa * (rh_pct / 100.0)
            df["vapor_pressure_deficit_kpa"] = np.maximum(0.0, es_kpa - ea_kpa)

        # B. Nocturnal Inversion Index
        if "nocturnal_inversion_index" not in df.columns:
            valley_depth_m = np.maximum(0.0, -df["delta_elevation_m"])
            wind_speed = df["wind_speed_10m_kmh"].clip(lower=0.5)
            solar_fraction = (df["solar_radiation_w_m2"] / 1000.0).clip(lower=0.0, upper=1.0)
            night_weight = 1.0 - solar_fraction
            df["nocturnal_inversion_index"] = (valley_depth_m / wind_speed) * night_weight

        # C. Solar Heating Interaction
        if "solar_heating_interaction" not in df.columns:
            solar_rad = df["solar_radiation_w_m2"].clip(lower=0.0)
            diurnal_factor = np.maximum(0.0, -df["hour_cos"])
            df["solar_heating_interaction"] = (solar_rad / 1000.0) * diurnal_factor

        # D. Thermal Inertia Lag
        if "thermal_inertia_lag_3h" not in df.columns:
            if "timestamp" in df.columns and "baseline_temp_c" in df.columns:
                baseline_agg = df.groupby("timestamp")["baseline_temp_c"].mean().reset_index()
                sorted_times = baseline_agg.sort_values("timestamp")
                sorted_times["thermal_inertia_lag_3h"] = sorted_times["baseline_temp_c"].diff(3).fillna(0.0)
                df = df.merge(sorted_times[["timestamp", "thermal_inertia_lag_3h"]], on="timestamp", how="left")
            else:
                df["thermal_inertia_lag_3h"] = 0.0

        # E. Topographic Position Index (TPI) and Spatial Gradients
        if any(f not in df.columns for f in ["topographic_position_index", "slope_magnitude_deg", "aspect_sin", "aspect_cos"]):
            if "point_id" in df.columns and "latitude" in df.columns and "longitude" in df.columns:
                coords = df[["point_id", "latitude", "longitude", "elevation_m"]].drop_duplicates().set_index("point_id")
                point_tpi, point_slope, point_aspect_sin, point_aspect_cos = {}, {}, {}, {}
                for pid, row in coords.iterrows():
                    lat, lon, elev = row["latitude"], row["longitude"], row["elevation_m"]
                    dists = np.sqrt((coords["latitude"] - lat) ** 2 + (coords["longitude"] - lon) ** 2)
                    neighbors = coords[(dists > 0) & (dists <= 0.28)]
                    if len(neighbors) > 0:
                        mean_neighbor_elev = float(neighbors["elevation_m"].mean())
                        tpi = float(elev - mean_neighbor_elev)
                        d_lat = neighbors["latitude"].values - lat
                        d_lon = neighbors["longitude"].values - lon
                        d_elev = neighbors["elevation_m"].values - elev
                        grad_ns = float(np.mean(d_elev / np.maximum(1e-5, np.abs(d_lat * 111000.0)) * np.sign(d_lat)))
                        grad_ew = float(np.mean(d_elev / np.maximum(1e-5, np.abs(d_lon * 100000.0)) * np.sign(d_lon)))
                        slope_deg = float(np.degrees(np.arctan(np.sqrt(grad_ns ** 2 + grad_ew ** 2))))
                        aspect_rad = float(np.arctan2(grad_ns, grad_ew))
                        aspect_s = float(np.sin(aspect_rad))
                        aspect_c = float(np.cos(aspect_rad))
                    else:
                        tpi, slope_deg, aspect_s, aspect_c = 0.0, 0.5, 0.0, 1.0
                    point_tpi[pid] = tpi
                    point_slope[pid] = slope_deg
                    point_aspect_sin[pid] = aspect_s
                    point_aspect_cos[pid] = aspect_c
                
                df["topographic_position_index"] = df["point_id"].map(point_tpi).fillna(0.0)
                df["slope_magnitude_deg"] = df["point_id"].map(point_slope).fillna(0.5)
                df["aspect_sin"] = df["point_id"].map(point_aspect_sin).fillna(0.0)
                df["aspect_cos"] = df["point_id"].map(point_aspect_cos).fillna(1.0)
            else:
                df["topographic_position_index"] = 0.0
                df["slope_magnitude_deg"] = 0.5
                df["aspect_sin"] = 0.0
                df["aspect_cos"] = 1.0

        # F. Sloped Solar Insolation
        if "sloped_solar_insolation" not in df.columns:
            aspect_alignment = (df["aspect_sin"] * df["hour_sin"] + df["aspect_cos"] * df["hour_cos"])
            slope_factor = np.sin(np.radians(df["slope_magnitude_deg"]))
            solar_rad_norm = df["solar_radiation_w_m2"] / 1000.0
            df["sloped_solar_insolation"] = solar_rad_norm * (1.0 + slope_factor * aspect_alignment)

        # G. Soil-to-Air Thermal Gradient
        if "soil_air_thermal_gradient" not in df.columns:
            soil_t = df["soil_temperature_0_to_7cm_c"] if "soil_temperature_0_to_7cm_c" in df.columns else t_c
            df["soil_air_thermal_gradient"] = soil_t - t_c

        # H. Latent Evaporative Cooling Potential
        if "latent_cooling_potential" not in df.columns:
            et0 = df["et0_evapotranspiration_mm"] if "et0_evapotranspiration_mm" in df.columns else 0.20
            vpd = df["vapor_pressure_deficit_kpa"] if "vapor_pressure_deficit_kpa" in df.columns else 0.0
            df["latent_cooling_potential"] = et0 * vpd

    return df.fillna(0.0)


def load_training_dataset(data_dir: Path | None = None, dataset_name: str | None = None) -> pd.DataFrame:
    """
    Load the historical training dataset (Clean 10-Year Parquet preferred, 1-Year fallback).

    Args:
        data_dir: Optional custom directory path containing raw datasets.
        dataset_name: Optional explicit parquet filename to load.

    Returns:
        Loaded and feature-enriched pandas DataFrame.
    """
    raw_dir = data_dir or (ROOT_DIR / "data" / "raw")
    if dataset_name:
        target_path = raw_dir / dataset_name if not Path(dataset_name).is_absolute() else Path(dataset_name)
        if target_path.exists():
            logger.info("Loading specified training dataset: {}", target_path)
            return ensure_interaction_features(pd.read_parquet(target_path))

    clean_10yr = raw_dir / "jaipur_10yr_training_data_clean.parquet"
    vpdfix_10yr = raw_dir / "jaipur_10yr_training_data_vpdfix.parquet"
    parquet_10yr = raw_dir / "jaipur_10yr_training_data.parquet"
    parquet_1yr = raw_dir / "jaipur_1yr_training_data.parquet"
    csv_1yr = raw_dir / "jaipur_1yr_training_data.csv"

    if clean_10yr.exists():
        logger.info("Loading clean leak-free 10-year training dataset from Parquet: {}", clean_10yr)
        df = pd.read_parquet(clean_10yr)
    elif vpdfix_10yr.exists():
        logger.info("Loading VPD-fixed 10-year training dataset from Parquet: {}", vpdfix_10yr)
        df = pd.read_parquet(vpdfix_10yr)
    elif parquet_10yr.exists():
        logger.info("Loading 10-year training dataset from Parquet: {}", parquet_10yr)
        df = pd.read_parquet(parquet_10yr)
    elif parquet_1yr.exists():
        logger.info("Loading 1-year training dataset from Parquet: {}", parquet_1yr)
        df = pd.read_parquet(parquet_1yr)
    elif csv_1yr.exists():
        logger.info("Loading training dataset from CSV: {}", csv_1yr)
        df = pd.read_csv(csv_1yr)
    else:
        raise FileNotFoundError(
            f"No training data found in {raw_dir}. "
            "Run 'python scripts/ingest_multi_year_dataset.py' or 'python scripts/download_training_data.py' first."
        )

    return ensure_interaction_features(df)



def spatial_three_way_split(
    df: pd.DataFrame,
    val_fraction: float = 0.10,
    test_fraction: float = 0.20,
    seed: int = 42,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    """
    Perform a strict 3-way Spatial Block Holdout Split across grid points.

    - Train set: used for model weight optimization.
    - Validation set: used for early stopping monitoring (prevents data leakage).
    - Test set: strictly unseen spatial holdout for final unbiased evaluation metrics.

    Args:
        df: Complete dataset.
        val_fraction: Fraction of spatial locations to hold out for early stopping validation.
        test_fraction: Fraction of spatial locations to hold out for final test evaluation.
        seed: Random seed for deterministic spatial point selection.
        random_state: Alias for seed.

    Returns:
        (train_df, val_df, test_df, train_point_ids, val_point_ids, test_point_ids)
    """
    effective_seed = random_state if random_state is not None else seed
    unique_points = sorted(df["point_id"].unique())
    num_points = len(unique_points)
    num_test = max(1, int(round(num_points * test_fraction)))
    num_val = max(1, int(round(num_points * val_fraction)))

    # Ensure train set has at least 1 point
    while num_test + num_val >= num_points and (num_test > 1 or num_val > 1):
        if num_val > 1:
            num_val -= 1
        elif num_test > 1:
            num_test -= 1
        else:
            break

    rng = np.random.RandomState(effective_seed)
    shuffled = rng.permutation(unique_points)

    test_points = sorted(shuffled[:num_test].tolist())
    val_points = sorted(shuffled[num_test : num_test + num_val].tolist())
    train_points = sorted(shuffled[num_test + num_val :].tolist())

    train_df = df[df["point_id"].isin(train_points)].reset_index(drop=True)
    val_df = df[df["point_id"].isin(val_points)].reset_index(drop=True)
    test_df = df[df["point_id"].isin(test_points)].reset_index(drop=True)

    logger.info(
        "Spatial 3-Way Split: {} Total Points -> {} Train ({} rows), {} Val ({} rows), {} Test ({} rows)",
        num_points,
        len(train_points),
        len(train_df),
        len(val_points),
        len(val_df),
        len(test_points),
        len(test_df),
    )
    return train_df, val_df, test_df, train_points, val_points, test_points


def spatial_block_split(
    df: pd.DataFrame,
    test_fraction: float = 0.22,
    seed: int = 42,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """
    Perform Spatial Block Holdout Split across the grid points for all 12 months.

    Args:
        df: Complete 1-year dataset.
        test_fraction: Fraction of spatial locations to hold out for testing.
        seed: Random seed for deterministic spatial point selection.
        random_state: Alias for seed.

    Returns:
        (train_df, test_df, train_point_ids, test_point_ids)
    """
    effective_seed = random_state if random_state is not None else seed
    unique_points = sorted(df["point_id"].unique())
    num_points = len(unique_points)
    num_test = max(1, int(round(num_points * test_fraction)))

    np.random.seed(effective_seed)
    test_indices = np.linspace(0, num_points - 1, num_test, dtype=int)
    test_points = [unique_points[i] for i in test_indices]
    train_points = [p for p in unique_points if p not in test_points]

    train_df = df[df["point_id"].isin(train_points)].reset_index(drop=True)
    test_df = df[df["point_id"].isin(test_points)].reset_index(drop=True)

    logger.info(
        "Spatial Block Split: {} Total Points -> {} Train Points ({} rows), {} Test Points ({} rows)",
        num_points,
        len(train_points),
        len(train_df),
        len(test_points),
        len(test_df),
    )
    return train_df, test_df, train_points, test_points


def run_spatial_cross_validation(
    df: pd.DataFrame | None = None,
    n_splits: int = 5,
    use_gpu: bool = False,
) -> dict[str, Any]:
    """
    Execute Spatial Group K-Fold Cross-Validation across all grid points.

    Args:
        df: Optional pre-loaded DataFrame.
        n_splits: Number of spatial folds (default: 5).
        use_gpu: Whether to enable CUDA acceleration.

    Returns:
        Dictionary of fold-by-fold and aggregated cross-validation metrics.
    """
    if df is None:
        df = load_training_dataset()
    else:
        df = ensure_interaction_features(df)

    unique_points = np.array(sorted(df["point_id"].unique()))
    np.random.seed(42)
    shuffled_points = np.random.permutation(unique_points)
    folds = np.array_split(shuffled_points, n_splits)

    fold_results: list[dict[str, float]] = []
    device = "cuda" if use_gpu else "cpu"

    logger.info("=================================================================")
    logger.info("  STARTING {}-FOLD SPATIAL CROSS-VALIDATION (ALL 2024 SEASONS)  ", n_splits)
    logger.info("=================================================================")

    for fold_idx, val_points in enumerate(folds, 1):
        val_set = set(val_points)
        train_df = df[~df["point_id"].isin(val_set)].reset_index(drop=True)
        val_df = df[df["point_id"].isin(val_set)].reset_index(drop=True)

        X_train, y_train = train_df[FEATURES_LIST], train_df[TARGET_COLUMN]
        X_val, y_val = val_df[FEATURES_LIST], val_df[TARGET_COLUMN]

        t_actual = val_df["temperature_2m_c"].to_numpy()
        t_base = val_df["baseline_temp_c"].to_numpy()

        model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            early_stopping_rounds=50,
            tree_method="hist",
            device=device,
            random_state=42 + fold_idx,
            verbosity=0,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        pred_res = model.predict(X_val)
        pred_down = t_base + pred_res

        b_mae = float(mean_absolute_error(t_actual, t_base))
        b_rmse = float(np.sqrt(mean_squared_error(t_actual, t_base)))
        d_mae = float(mean_absolute_error(t_actual, pred_down))
        d_rmse = float(np.sqrt(mean_squared_error(t_actual, pred_down)))
        d_r2 = float(r2_score(t_actual, pred_down))
        rmse_gain = ((b_rmse - d_rmse) / b_rmse) * 100.0

        fold_results.append(
            {
                "fold": fold_idx,
                "val_points_count": len(val_points),
                "val_rows": len(val_df),
                "best_iteration": int(model.best_iteration if hasattr(model, "best_iteration") else 1000),
                "baseline_mae": b_mae,
                "baseline_rmse": b_rmse,
                "downscaled_mae": d_mae,
                "downscaled_rmse": d_rmse,
                "downscaled_r2": d_r2,
                "rmse_improvement_pct": rmse_gain,
            }
        )

        logger.info(
            "Fold {}/{} | Best Tree: {:3d} | Base RMSE: {:.3f}°C -> Down RMSE: {:.3f}°C (+{:.1f}% gain)",
            fold_idx,
            n_splits,
            fold_results[-1]["best_iteration"],
            b_rmse,
            d_rmse,
            rmse_gain,
        )

    # Aggregations
    mean_b_rmse = float(np.mean([f["baseline_rmse"] for f in fold_results]))
    mean_d_rmse = float(np.mean([f["downscaled_rmse"] for f in fold_results]))
    mean_b_mae = float(np.mean([f["baseline_mae"] for f in fold_results]))
    mean_d_mae = float(np.mean([f["downscaled_mae"] for f in fold_results]))
    mean_r2 = float(np.mean([f["downscaled_r2"] for f in fold_results]))
    mean_gain = float(np.mean([f["rmse_improvement_pct"] for f in fold_results]))
    std_gain = float(np.std([f["rmse_improvement_pct"] for f in fold_results]))

    summary = {
        "fold_results": fold_results,
        "mean_baseline_rmse": mean_b_rmse,
        "mean_downscaled_rmse": mean_d_rmse,
        "mean_baseline_mae": mean_b_mae,
        "mean_downscaled_mae": mean_d_mae,
        "mean_downscaled_r2": mean_r2,
        "mean_rmse_improvement_pct": mean_gain,
        "std_rmse_improvement_pct": std_gain,
    }

    logger.success(
        "Spatial {}-Fold CV Summary: Mean RMSE: {:.3f}°C (Mean Gain: +{:.1f}% ± {:.1f}%)",
        n_splits,
        mean_d_rmse,
        mean_gain,
        std_gain,
    )
    return summary


def train_residual_model(
    df: pd.DataFrame | None = None,
    save_artifact: bool = True,
    use_gpu: bool = False,
    n_estimators: int = 6000,
    max_depth: int = 8,
    learning_rate: float = 0.03,
    early_stopping_rounds: int = 200,
    zone_id: str = "ACZ-14",
    output_path: Path | str | None = None,
) -> tuple[xgb.XGBRegressor, dict[str, Any]]:
    """
    Train the deepened XGBoost microclimate residual downscaling model with early stopping.

    Args:
        df: Optional pre-loaded DataFrame.
        save_artifact: If True, serializes model to artifacts/residual_model.joblib.
        use_gpu: If True, explicitly requests CUDA GPU acceleration.
        n_estimators: Max decision trees to grow (default: 3600).
        max_depth: Tree depth (default: 8).
        learning_rate: Learning rate / shrinkage (default: 0.03).
        early_stopping_rounds: Early stopping patience (default: 150).
        zone_id: Zone identifier (default: "ACZ-14").

    Returns:
        (trained_model, evaluation_metrics_dict)
    """
    if df is None:
        df = load_training_dataset()
    else:
        df = ensure_interaction_features(df)

    # Compute domain elevation and anchor points for serving baseline parity
    domain_mean_elev = float(df["elevation_m"].mean())
    anchor_points: list[dict[str, float]] = []
    anchor_mean_elev = domain_mean_elev
    bbox = {
        "min_lat": float(df["latitude"].min()) if "latitude" in df.columns else 26.0,
        "max_lat": float(df["latitude"].max()) if "latitude" in df.columns else 28.0,
        "min_lon": float(df["longitude"].min()) if "longitude" in df.columns else 75.0,
        "max_lon": float(df["longitude"].max()) if "longitude" in df.columns else 77.0,
    }

    if "latitude" in df.columns and "longitude" in df.columns and "elevation_m" in df.columns:
        pts_meta = df[["point_id", "latitude", "longitude", "elevation_m"]].drop_duplicates().copy()
        pts_meta["elev_diff"] = (pts_meta["elevation_m"] - domain_mean_elev).abs()
        # Select 8 points closest to domain mean elevation to construct anchor baseline
        selected_anchors = pts_meta.sort_values("elev_diff").head(8)
        anchor_mean_elev = float(selected_anchors["elevation_m"].mean())
        for _, row in selected_anchors.iterrows():
            anchor_points.append({
                "point_id": str(row["point_id"]),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "elevation_m": float(row["elevation_m"]),
            })
        logger.info(
            "Selected 8 Anchor Points (Mean Elevation: {:.1f}m vs Domain Mean: {:.1f}m, Δ={:.2f}m)",
            anchor_mean_elev,
            domain_mean_elev,
            abs(anchor_mean_elev - domain_mean_elev),
        )

    # 1. Spatial Block Holdout Split (3-Way: Train / Validation for Early Stopping / Test for Final Evaluation)
    train_df, val_df, test_df, train_points, val_points, test_points = spatial_three_way_split(df)

    X_train = train_df[FEATURES_LIST]
    y_train = train_df[TARGET_COLUMN]

    X_val = val_df[FEATURES_LIST]
    y_val = val_df[TARGET_COLUMN]

    X_test = test_df[FEATURES_LIST]
    y_test = test_df[TARGET_COLUMN]

    # Ground truth actual temperatures
    t_actual_test = test_df["temperature_2m_c"].to_numpy()
    t_baseline_test = test_df["baseline_temp_c"].to_numpy()

    # 2. Configure XGBRegressor with Early Stopping
    tree_method = "hist"
    device = "cuda" if use_gpu else "cpu"

    logger.info(
        "Instantiating Deepened XGBRegressor (n_estimators={}, max_depth={}, lr={}, early_stopping={}, device={}, features={})",
        n_estimators,
        max_depth,
        learning_rate,
        early_stopping_rounds,
        device,
        len(FEATURES_LIST),
    )

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        early_stopping_rounds=early_stopping_rounds,
        tree_method=tree_method,
        device=device,
        random_state=42,
        verbosity=1,
    )

    # 3. Fit Model with Live Progress Logging (Validation set used strictly for early stopping)
    logger.info("Training deepened model on {} samples across all seasons...", len(X_train))
    t0 = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100,
    )
    train_time = time.perf_counter() - t0
    best_iter = getattr(model, "best_iteration", n_estimators)
    logger.success("Model converged at iteration {} in {:.2f} seconds", best_iter, train_time)

    # 4. Predict Residuals on Unseen Test Points (Never seen during training or early stopping)
    pred_residuals = model.predict(X_test)
    pred_downscaled_temp = t_baseline_test + pred_residuals

    # 5. Compute Quantitative Metrics
    baseline_mae = float(mean_absolute_error(t_actual_test, t_baseline_test))
    baseline_rmse = float(np.sqrt(mean_squared_error(t_actual_test, t_baseline_test)))
    baseline_r2 = float(r2_score(t_actual_test, t_baseline_test))

    downscaled_mae = float(mean_absolute_error(t_actual_test, pred_downscaled_temp))
    downscaled_rmse = float(np.sqrt(mean_squared_error(t_actual_test, pred_downscaled_temp)))
    downscaled_r2 = float(r2_score(t_actual_test, pred_downscaled_temp))

    mae_improvement_pct = ((baseline_mae - downscaled_mae) / baseline_mae) * 100.0
    rmse_improvement_pct = ((baseline_rmse - downscaled_rmse) / baseline_rmse) * 100.0

    residual_mae = float(mean_absolute_error(y_test, pred_residuals))
    residual_rmse = float(np.sqrt(mean_squared_error(y_test, pred_residuals)))
    residual_r2 = float(r2_score(y_test, pred_residuals))

    metrics: dict[str, Any] = {
        "zone_id": zone_id,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "train_points": train_points,
        "val_points": val_points,
        "test_points": test_points,
        "train_time_sec": train_time,
        "best_iteration": int(best_iter),
        "baseline_mae_c": baseline_mae,
        "baseline_rmse_c": baseline_rmse,
        "baseline_r2": baseline_r2,
        "downscaled_mae_c": downscaled_mae,
        "downscaled_rmse_c": downscaled_rmse,
        "downscaled_r2": downscaled_r2,
        "mae_improvement_pct": mae_improvement_pct,
        "rmse_improvement_pct": rmse_improvement_pct,
        "residual_mae_c": residual_mae,
        "residual_rmse_c": residual_rmse,
        "residual_r2": residual_r2,
        "feature_importances": dict(zip(FEATURES_LIST, model.feature_importances_.tolist())),
        "domain_mean_elevation_m": domain_mean_elev,
        "anchor_mean_elevation_m": anchor_mean_elev,
        "anchor_points": anchor_points,
        "bounding_box": bbox,
    }

    # 6. Quality Gate Assertion
    logger.info("Evaluating Quality Gate: RMSE_downscaled < RMSE_baseline")
    assert downscaled_rmse < baseline_rmse, (
        f"Quality gate failed: Downscaled RMSE ({downscaled_rmse:.4f}) >= Baseline RMSE ({baseline_rmse:.4f})"
    )
    assert downscaled_mae < baseline_mae, (
        f"Quality gate failed: Downscaled MAE ({downscaled_mae:.4f}) >= Baseline MAE ({baseline_mae:.4f})"
    )
    logger.success("Quality gate passed! Downscaler achieved {:.1f}% RMSE error reduction", rmse_improvement_pct)

    # 7. Serialize Artifacts
    if save_artifact:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        zones_dir = ARTIFACTS_DIR / "zones"
        zones_dir.mkdir(parents=True, exist_ok=True)
        
        zone_slug = zone_id.lower().replace("-", "_")
        zone_target_path = Path(output_path) if output_path else zones_dir / f"residual_model_{zone_slug}.joblib"
        
        artifact_payload = {
            "model": model,
            "features_list": FEATURES_LIST,
            "metrics": metrics,
            "domain_mean_elevation_m": domain_mean_elev,
            "anchor_mean_elevation_m": anchor_mean_elev,
            "anchor_points": anchor_points,
            "bounding_box": bbox,
            "zone_id": zone_id,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        
        SAVE_FN(artifact_payload, zone_target_path)
        logger.success("Saved Zone [{}] model artifact to {}", zone_id, zone_target_path)
        
        # Also maintain primary champion path for global default/backward compatibility
        SAVE_FN(artifact_payload, MODEL_PATH)
        logger.success("Saved primary model artifact payload to {}", MODEL_PATH)

    return model, metrics


if __name__ == "__main__":
    train_residual_model()
