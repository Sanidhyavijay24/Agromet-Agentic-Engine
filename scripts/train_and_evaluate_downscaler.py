"""
@file train_and_evaluate_downscaler.py
@description CLI script to train the deepened XGBoost downscaler with Spatial K-Fold CV & physics features.
@module scripts
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
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

    logger = _LoguruCompatLogger("train_and_eval")  # type: ignore[assignment]

def _simple_table(rows: list[list[Any]], headers: list[str]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
    
    header_str = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_strs = [" | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row)) for row in rows]
    return f"{header_str}\n{sep_str}\n" + "\n".join(row_strs)

try:
    from tabulate import tabulate
except ImportError:
    tabulate = lambda rows, headers=(), tablefmt="": _simple_table(rows, headers)  # type: ignore[assignment]

from services.api.schemas import (
    DownscaledForecast,
    GeoLocation,
    HourlyForecastPoint,
)
from services.ml_downscaler.inference import (
    downscale_point_forecast,
    load_downscaler_model,
)
from services.ml_downscaler.train import (
    load_training_dataset,
    run_spatial_cross_validation,
    train_residual_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the deepened XGBoost microclimate downscaler.")
    parser.add_argument("--use-gpu", action="store_true", help="Enable CUDA GPU acceleration.")
    parser.add_argument("--run-cv", action="store_true", help="Run full 5-Fold Spatial Cross-Validation.")
    parser.add_argument("--n-estimators", type=int, default=6000, help="Max decision trees to grow (default: 6000).")
    parser.add_argument("--early-stopping", type=int, default=200, help="Early stopping patience rounds (default: 200).")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Learning rate (default: 0.03).")
    parser.add_argument("--dataset", type=str, default=None, help="Explicit dataset parquet filename/path.")
    parser.add_argument("--zone-id", type=str, default="ACZ-14", help="Agro-Climatic Zone ID (default: ACZ-14).")
    parser.add_argument("--output-path", type=str, default=None, help="Explicit model artifact save path.")
    args = parser.parse_args()

    logger.info("=================================================================")
    logger.info("   SIH 26074: DEEPENED XGBOOST RESIDUAL DOWNSCALER (GPU/CV)      ")
    logger.info("=================================================================")

    df = load_training_dataset(dataset_name=args.dataset)

    # 1. Optional 5-Fold Spatial Cross-Validation
    if args.run_cv:
        cv_summary = run_spatial_cross_validation(df=df, n_splits=5, use_gpu=args.use_gpu)
        cv_rows = [
            [
                f"Fold {f['fold']} ({f['val_points_count']} Points, {f['val_rows']:,} rows)",
                f"{f['best_iteration']} trees",
                f"{f['baseline_rmse']:.3f}°C",
                f"{f['downscaled_rmse']:.3f}°C",
                f"+{f['rmse_improvement_pct']:.1f}%",
            ]
            for f in cv_summary["fold_results"]
        ]
        cv_rows.append([
            "MEAN ± STD ACROSS ALL FOLDS",
            "-",
            f"{cv_summary['mean_baseline_rmse']:.3f}°C",
            f"{cv_summary['mean_downscaled_rmse']:.3f}°C",
            f"+{cv_summary['mean_rmse_improvement_pct']:.1f}% ± {cv_summary['std_rmse_improvement_pct']:.1f}%",
        ])

        print("\n" + "=" * 80)
        print("          5-FOLD SPATIAL GROUP CROSS-VALIDATION RESULTS (10-YEAR DATASET)")
        print("=" * 80)
        print(tabulate(cv_rows, headers=["Spatial Fold", "Optimal Trees", "Baseline RMSE", "Downscaled RMSE", "Error Reduction (Δ)"], tablefmt="fancy_grid"))

    # 2. Train Primary Model with Early Stopping
    logger.info("Training Primary Model for Zone [{}] with {} Max Trees & Early Stopping (Patience: {})...", args.zone_id, args.n_estimators, args.early_stopping)
    model, metrics = train_residual_model(
        df=df,
        save_artifact=True,
        use_gpu=args.use_gpu,
        n_estimators=args.n_estimators,
        max_depth=8,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping,
        zone_id=args.zone_id,
        output_path=args.output_path,
    )

    # 3. Print Evaluation Comparison Table
    eval_table = [
        [
            "Mean Absolute Error (MAE)",
            f"{metrics['baseline_mae_c']:.3f}°C",
            f"{metrics['downscaled_mae_c']:.3f}°C",
            f"+{metrics['mae_improvement_pct']:.1f}% Error Drop",
        ],
        [
            "Root Mean Squared Error (RMSE)",
            f"{metrics['baseline_rmse_c']:.3f}°C",
            f"{metrics['downscaled_rmse_c']:.3f}°C",
            f"+{metrics['rmse_improvement_pct']:.1f}% Error Drop",
        ],
        [
            "Coefficient of Determination (R²)",
            f"{metrics['baseline_r2']:.4f}",
            f"{metrics['downscaled_r2']:.4f}",
            f"+{metrics['downscaled_r2'] - metrics['baseline_r2']:.4f} Variance",
        ],
        [
            "Optimal Trees (Early Stopped)",
            "-",
            f"{metrics['best_iteration']} / 1200 trees",
            f"Trained in {metrics['train_time_sec']:.2f}s",
        ],
    ]

    print("\n" + "=" * 80)
    print("      QUANTITATIVE EVALUATION ON UNSEEN SPATIAL HOLDOUT POINTS (10-YEAR HORIZON)")
    print("=" * 80)
    print(tabulate(eval_table, headers=["Metric", "Coarse Baseline (No ML)", "Downscaled XGBoost", "Improvement (Δ)"], tablefmt="fancy_grid"))

    # 4. Top Feature Importances
    sorted_features = sorted(metrics["feature_importances"].items(), key=lambda x: x[1], reverse=True)
    feat_table = [[rank, feat, f"{score * 100:.2f}%"] for rank, (feat, score) in enumerate(sorted_features[:10], 1)]

    print("\n" + "=" * 80)
    print("              TOP 10 PHYSICAL & AGRO FEATURE IMPORTANCES")
    print("=" * 80)
    print(tabulate(feat_table, headers=["Rank", "Feature Name", "Relative Importance"], tablefmt="fancy_grid"))

    # 5. Test Live Inference Pass
    logger.info("Executing sample inference test for Panchayat Chaksu...")
    test_loc = GeoLocation(
        latitude=26.9124,
        longitude=75.7873,
        elevation_m=431.0,
        district="Jaipur",
        panchayat="Chaksu",
    )
    test_point = HourlyForecastPoint(
        time=datetime(2024, 7, 15, 14, 0, 0, tzinfo=timezone.utc),
        temperature_2m_c=34.5,
        relative_humidity_2m_pct=52.0,
        surface_pressure_hpa=955.2,
        wind_speed_10m_kmh=12.4,
        precipitation_mm=0.0,
    )
    sample_covariates = {
        "solar_radiation_w_m2": 720.0,
        "soil_moisture_0_to_7cm_m3m3": 0.22,
        "et0_evapotranspiration_mm": 0.65,
    }

    result = downscale_point_forecast(
        panchayat_id="CHAKSU_001",
        location=test_loc,
        baseline_point=test_point,
        covariates=sample_covariates,
    )

    # 6. Save Processed Payload
    processed_dir = ROOT_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    sample_out_path = processed_dir / "sample_downscaled_forecast.json"

    with open(sample_out_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))

    logger.success("Saved verified DownscaledForecast payload to {}", sample_out_path)

    inf_table = [
        ["Target Panchayat", f"{test_loc.panchayat}, {test_loc.district} (Elev: {test_loc.elevation_m}m ASL)"],
        ["Forecast Timestamp", str(test_point.time)],
        ["Coarse Regional Baseline Temp", f"{result.baseline_value}°C"],
        ["Predicted Residual Anomaly (R̂)", f"{result.covariates_used.get('predicted_residual_c', 0.0):+0.2f}°C"],
        ["Hyperlocal Downscaled Temp (T̂)", f"{result.downscaled_value}°C"],
        ["Model Confidence Score", f"{result.confidence_score * 100:.1f}%"],
    ]

    print("\n" + "=" * 80)
    print("             SAMPLE HYPERLOCAL DOWNSCALED FORECAST RESULT")
    print("=" * 80)
    print(tabulate(inf_table, headers=["Parameter", "Value"], tablefmt="fancy_grid"))
    print("=" * 80 + "\n")

    logger.success("Deepened ML Downscaler training, evaluation, and inference verification completed successfully!")


if __name__ == "__main__":
    main()
