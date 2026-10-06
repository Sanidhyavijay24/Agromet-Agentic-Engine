# Experiment 14: Leak-Free 10-Year Retraining with Real 30m USGS SRTM DEM (5,555 Trees)

## 1. Overview & Objective
- **Experiment ID:** `EXP-14` (Converged Production Champion)
- **Objective:** Train the definitive 10-year physics-guided residual downscaler for Zone XIV (Western Dry / Rajasthan) on authentic 30m USGS SRTM DEM terrain data with expanded tree capacity (up to 6,000 estimators with early stopping) and zero target leakage.
- **Dataset:** 3,156,192 hourly records across 10 years (2015-2024), partitioned via strict spatial holdout (28 train points / 8 test points).

---

## 2. Retraining Configuration
- **Model Architecture:** `xgboost.XGBRegressor`
- **Hyperparameters:**
  - `n_estimators`: 6,000 (Early stopped at iteration **5,555**)
  - `max_depth`: 8
  - `learning_rate`: 0.03
  - `subsample`: 0.85
  - `colsample_bytree`: 0.85
  - `reg_alpha`: 0.1
  - `reg_lambda`: 1.0
  - `tree_method`: `hist` (`device='cuda'`)
- **Terrain Data:** Authentic 30m USGS SRTM Digital Elevation Model (`N26E075.hgt`, `N26E076.hgt`, `N27E075.hgt`, `N27E076.hgt`).
- **Feature Matrix:** 28 leak-free physics-guided features (including TPI, slope magnitude, aspect, solar geometry, and boundary layer stability).

---

## 3. Verified Benchmark Results

| Metric | Analytical Baseline (LOO Lapse) | ML Residual Model (Exp 14) | Verified Performance Gain |
| :--- | :---: | :---: | :---: |
| **Root Mean Squared Error (RMSE)** | 0.746 °C | **0.446 °C** | **+40.2% Error Reduction** |
| **Mean Absolute Error (MAE)** | 0.538 °C | **0.319 °C** | **+40.7% Error Reduction** |
| **R² Score** | 0.9908 | **0.9967** | **+0.0059 Explained Variance** |
| **Max Residual Error** | 4.820 °C | **2.110 °C** | **-56.2% Peak Error Suppression** |

---

## 4. Analysis of Empirical Findings
- **Why +40.2% is the True Honest Figure:**
  - Earlier prototype experiments (Exp 10) reported an inflated +60.6% error reduction due to synthetic grid elevation approximations and VPD coupling leakage.
  - In Experiment 14, using authentic 30m USGS SRTM elevation and strict spatial holdout with zero leakage, the verified RMSE drop is **+40.2%** ($0.746^\circ	ext{C} 	o 0.446^\circ	ext{C}$).
- **Production Artifact:** Serialized as `services/ml_downscaler/artifacts/zones/residual_model_acz_14.joblib` (195 MB, 5,555 trees), serving sub-millisecond live microclimate inferences.
