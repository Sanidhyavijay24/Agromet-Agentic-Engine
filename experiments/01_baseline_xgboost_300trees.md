# Experiment 01: Initial XGBoost Residual Downscaler Baseline (300 Trees)

## 1. Objective & Hypothesis
**Objective:** Establish the first functional machine learning residual downscaler baseline on the extracted 2024 hourly dataset for Jaipur / Chaksu.

**Hypothesis:** An `XGBRegressor` trained on topographical (Delta Elevation, theoretical lapse rate) and basic meteorological features will learn non-linear spatial residuals (`R = T_local - T_baseline`) and outperform coarse NWP regional averages on unseen spatial coordinates.

---

## 2. Experimental Configuration
- **Model Architecture:** `xgboost.XGBRegressor`
- **Hyperparameters:**
  - `n_estimators`: 300
  - `max_depth`: 6
  - `learning_rate`: 0.05
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `tree_method`: `hist` (CPU)
- **Features (8 features):**
  - `coarse_temperature_c`, `coarse_relative_humidity_pct`, `coarse_pressure_hpa`, `coarse_wind_speed_ms`
  - `delta_elevation_m`, `theoretical_lapse_delta_c`
  - `hour_sin`, `hour_cos`
- **Target:** `residual_temperature_c = local_temperature_c - coarse_temperature_c`
- **Split Strategy:** Spatial Holdout (80% spatial points train / 20% spatial points test, 2024 hourly data).

---

## 3. Results & Evaluation Metrics

| Metric | Coarse NWP Baseline | XGBoost Downscaled (Exp 01) | Improvement |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | 0.523 °C | **0.391 °C** | **+25.3% Error Reduction** |
| **Root Mean Squared Error (RMSE)** | 0.726 °C | **0.543 °C** | **+25.1% Error Reduction** |
| **R² Score** | 0.9880 | **0.9934** | **+0.0054 Variance Explained** |

---

## 4. Key Takeaways & Limitations
- **Confirmed Viability:** Predicting residual deviations rather than absolute temperature enables fast convergence and ensures the model acts as a physical correction layer.
- **Identified Bottlenecks:**
  1. 300 shallow trees underfit complex diurnal boundary layer transitions (e.g. nocturnal radiative cooling).
  2. The 8-feature schema lacked atmospheric physics interactions (VPD, thermal inertia, solar geometry).
