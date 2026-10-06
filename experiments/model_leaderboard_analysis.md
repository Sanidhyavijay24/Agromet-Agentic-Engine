# ML Downscaler Experiments Leaderboard & Comparative Analysis

> **Single Source of Experiment Tracking for SIH 26074**  
> *Updated after every downscaler experiment to benchmark performance, track metrics, and define the active production champion.*

---

## 1. Master Experiment Leaderboard

| Exp # | Experiment Name | Model Architecture | Features | Validation Scheme | Test MAE (°C) | Test RMSE (°C) | Test $R^2$ | RMSE Drop (Δ%) | Train Duration | Status |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **—** | **Coarse Baseline** | No ML (Spatial Mean) | 0 | Spatial Holdout (8 pts) | 0.523°C | 0.726°C | 0.9923 | 0.0% | — | Baseline |
| **04** | [Exp 04: Standard Lapse Rate](04_linear_lapse_rate_benchmark.md) | Analytical Lapse ($-6.5^\circ\text{C}/\text{km}$) | 1 | Spatial Holdout (8 pts) | 0.547°C | 0.774°C | 0.9912 | -6.7% | < 0.01s | Benchmark Floor |
| **04b** | [Exp 04b: Fitted 1D Lapse](04_linear_lapse_rate_benchmark.md) | Empirical 1D Linear Regression | 1 | Spatial Holdout (8 pts) | 0.508°C | 0.728°C | 0.9922 | -0.4% | 0.02s | Benchmark Floor |
| **05** | [Exp 05: Random Forest](05_random_forest_benchmark.md) | Bagging Ensemble (300 trees, depth 14) | 21 | Spatial Holdout (8 pts) | 0.426°C | 0.602°C | 0.9947 | +17.0% | 8.73s (16 threads) | Evaluated |
| **07** | [Exp 07: CatBoost](07_catboost_benchmark.md) | Symmetric GBDT (1199 trees, depth 8) | 21 | Spatial Holdout (8 pts) | 0.356°C | 0.498°C | 0.9964 | +31.4% | 15.31s (GPU) | Evaluated |
| **06** | [Exp 06: LightGBM (21-Feat)](06_lightgbm_benchmark.md) | Leaf-wise GBDT (1200 trees, leaves 63) | 21 | Spatial Holdout (8 pts) | 0.323°C | 0.451°C | 0.9970 | +37.8% | 4.11s (16 threads) | Evaluated |
| **01** | [Exp 01: Baseline XGBoost](01_baseline_xgboost_300trees.md) | XGBoost (300 trees, depth 6, lr 0.05) | 17 | Spatial Holdout (8 pts) | 0.391°C | 0.543°C | 0.9957 | +25.1% | 1.41s (GPU) | Superseded |
| **02** | [Exp 02: Deepened + Physics](02_deepened_xgboost_physics_features.md) | XGBoost (1200 trees, depth 8, lr 0.03, ES 50) | 21 | Spatial Holdout (8 pts) | 0.311°C | 0.439°C | 0.9972 | +39.5% | 8.03s (GPU) | Superseded |
| **03** | [Exp 03: Spatial 5-Fold CV (1-Yr)](03_spatial_5fold_cross_validation.md) | XGBoost (1000 trees, depth 8, lr 0.03, ES 50) | 21 | 5-Fold Group CV 1-Yr (All 36 pts) | 0.330°C (mean) | 0.463°C (mean) | 0.9969 (mean) | +37.6% ± 2.6% | 34.3s (5 folds) | Verified Stability |
| **08b** | [Exp 08: Track 2 LightGBM](08_track2_physics_features.md) | Leaf-wise GBDT (1200 trees, leaves 63) | 28 | Spatial Holdout (8 pts) | 0.307°C | 0.425°C | 0.9974 | +41.5% | 5.85s (16 threads) | Top Contender |
| **08** | [Exp 08: Track 2 XGBoost (1-Yr)](08_track2_physics_features.md) | Depth-wise GBDT (1200 trees, depth 8) | 28 | Spatial Holdout 1-Yr (8 pts, 70k) | 0.287°C | 0.400°C | 0.9977 | +44.9% | 8.23s (GPU) | 1-Yr Champion |
| **09b** | [Exp 09: 10-Yr LightGBM](09_10yr_dataset_scaling_experiment.md) | Leaf-wise GBDT (1200 trees, leaves 63) | 28 | Spatial Holdout 10-Yr (8 pts, 701k) | 0.231°C | 0.313°C | 0.8147 | +57.0% | 54.9s (CPU) | Evaluated |
| **09** | [Exp 09: 10-Yr XGBoost](09_10yr_dataset_scaling_experiment.md) | Depth-wise GBDT (1200 trees, depth 8) | 28 | Spatial Holdout 10-Yr (8 pts, 701k) | 0.214°C | 0.288°C | 0.8434 | +60.5% | 28.66s (GPU) | Superseded (Audit Fixes) |
| **10b**| [Exp 10b: 10-Yr Spatial 5-Fold CV](10_leak_free_audit_and_retraining_verification.md#b-5-fold-spatial-group-cross-validation-all-36-spatial-points-across-10-year-horizon) | Depth-wise GBDT (1000 trees, depth 8) | 28 | 5-Fold Group CV 10-Yr (All 36 pts, 3.15M) | — | 0.329°C (mean) | — | +58.3% ± 0.9% | 110.2s (5 folds) | Superseded (Audit Fixes) |
| **10** | [Exp 10: Leak-Free 10-Yr XGBoost](10_leak_free_audit_and_retraining_verification.md) | Depth-wise GBDT (1200 trees, depth 8) | 28 | Spatial 3-Way LOO (7 pts, 614k) | 0.218°C | 0.294°C | 0.9986 | +60.6% | 23.30s (GPU) | Superseded (VPD Audit Fix) |
| **11** | [Exp 11: 2025 Future Year Temporal Holdout](11_unseen_2025_temporal_holdout_benchmark.md) | Depth-wise GBDT (Champion) | 28 | Pure Temporal Holdout (All 36 pts, 315k) | 0.278°C | 0.365°C | 0.9976 | +55.6% | Evaluated on 2025 | Verified Temporal Robustness |
| **11b**| [Exp 11b: 2025 Spatio-Temporal Holdout](11_unseen_2025_temporal_holdout_benchmark.md#b-unseen-spatial-points-in-2025-dual-spatio-temporal-holdout-61320-rows) | Depth-wise GBDT (Champion) | 28 | Dual Spatio-Temporal (7 test pts, 61k) | 0.284°C | 0.370°C | 0.9974 | +52.1% | Evaluated on 2025 | Verified Spatio-Temporal |
| **14** | 🏆 [Exp 14: Leak-Free 10-Yr Converged Champion](14_leak_free_dem_retraining_3600_trees.md) | Depth-wise GBDT (5555 trees, depth 8) | 28 | Spatial 3-Way LOO (7 pts, 614k) | **0.319°C** | **0.446°C** | **0.9967** | **+40.2%** | **129.21s (GPU)** | 🏆 **ACTIVE CONVERGED CHAMPION** |
| **14b**| [Exp 14b: Leak-Free 10-Yr Spatial 5-Fold CV](14_leak_free_dem_retraining_3600_trees.md) | Depth-wise GBDT (1000 trees, depth 8) | 28 | 5-Fold Group CV 10-Yr (All 36 pts, 3.15M) | — | **0.533°C (mean)** | — | **+32.4% ± 1.6%** | **140.5s (5 folds)** | **Verified 10-Yr Spatial Stability** |

---

## 2. Active Champion Model Specification

- **Current Champion:** **Experiment 14 (Leak-Free Converged XGBoost with 28 Physics Features, 30m SRTM DEM, & 8 Anchor Baseline)**
- **Key Hyperparameters:**
  - `n_estimators`: 6,000 (converged at optimal step **5,555**, early stopping patience 200)
  - `max_depth`: 8
  - `learning_rate`: 0.03
  - `subsample`: 0.85
  - `colsample_bytree`: 0.85
  - `early_stopping_rounds`: 200 (monitored on disjoint 4-point Validation set)
  - `tree_method`: `hist` (CUDA GPU)
- **Data & Partitioning Scheme:**
  - Dataset: 3,156,192 hourly records across 36 spatial points (10 years: 2015–2024, zero nulls)
  - Parquet: `data/raw/jaipur_10yr_training_data_clean.parquet` (Clean VPD from baseline + continuous 30m SRTM DEM)
  - Target: Leave-One-Out (LOO) Residual Anomaly ($R = T_{\text{local}} - T_{\text{baseline\_LOO}}$)
  - Serving Baseline: 8 Anchor Points (Mean Elev: 374.8m vs Domain Mean: 373.9m, $\Delta = 0.89\text{m}$)
  - Split: 25 Train (2.19M rows) / 4 Val (350k rows) / 7 Test (614k rows)
- **Feature Set:** 28 variables (topography, TPI, slope, aspect, insolation, air-to-soil thermal gradient, latent cooling).
- **Inference Latency:** **$< 0.003\text{ ms}$** per sample on holdout test set.
- **Model Artifacts:**
  - Zone Artifact: [`services/ml_downscaler/artifacts/zones/residual_model_acz_14.joblib`](file:///c:/Codes/SIH/SIH/services/ml_downscaler/artifacts/zones/residual_model_acz_14.joblib)
  - Global Default: [`services/ml_downscaler/artifacts/residual_model.joblib`](file:///c:/Codes/SIH/SIH/services/ml_downscaler/artifacts/residual_model.joblib)

---

## 3. Key Mathematical & Physical Takeaways from Track 2

1. **Breakthrough Beyond 44% Error Reduction:** Expanding to 28 physics features achieved a record **+44.9% RMSE error reduction** ($0.726^\circ\text{C} \to 0.400^\circ\text{C}$) and **+45.1% MAE error drop** ($0.523^\circ\text{C} \to 0.287^\circ\text{C}$).
2. **Ground Sensible Heat Flux is Critical (#4 Rank, 6.09%):** `soil_air_thermal_gradient` ($T_{\text{soil\_0-7cm}} - T_{\text{baseline}}$) immediately became the 4th most important predictor across all 28 features, capturing how the ground surface directly heats or cools the 2m air column.
3. **Terrain Morphology & Solar Coupling (#7, #11, #12 Ranks):** `slope_magnitude_deg` (3.38%), `sloped_solar_insolation` (2.81%), and `aspect_cos` (2.71%) provided the spatial geometry needed to resolve microclimate temperature deltas on uneven terrain.
4. **LightGBM Performance (+41.5%):** LightGBM also surged from +37.8% $\to$ +41.5% with the new 28-feature set, confirming that the new physical features generalize across completely different GBDT algorithms.
