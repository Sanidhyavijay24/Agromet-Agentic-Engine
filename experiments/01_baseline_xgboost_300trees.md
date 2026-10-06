# Experiment 01: Initial XGBoost Residual Downscaler Baseline (300 Trees)

## 1. Objective & Hypothesis
**Objective:** Establish the first functional machine learning residual downscaler baseline on the extracted 2024 hourly dataset for Jaipur/Chaksu.

**Hypothesis:** An `XGBRegressor` trained on topographical ($\Delta\text{Elevation}$, theoretical lapse rate) and basic meteorological features will learn non-linear spatial residuals ($R = T_{\text{local}} - T_{\text{baseline}}$) and outperform coarse NWP regional averages on unseen spatial coordinates.

---

## 2. Experimental Configuration
- **Model Architecture:** `xgboost.XGBRegressor`
- **Hyperparameters:**
  - `n_estimators`: 300
  - `max_depth`: 6
  - `learning_rate`: 0.05
  - `subsample`: 0.80
  - `colsample_bytree`: 0.80
  - `tree_method`: `hist` (CUDA GPU)
  - `early_stopping_rounds`: None
- **Feature Set (17 Features):**
  - Topographical: `delta_elevation_m`, `theoretical_lapse_delta_c`, `elevation_m`
  - Atmospheric Baseline: `baseline_temp_c`, `relative_humidity_2m_pct`, `surface_pressure_hpa`, `wind_speed_10m_kmh`, `precipitation_mm`
  - Solar & Agro: `solar_radiation_w_m2`, `shortwave_radiation_w_m2`, `soil_temperature_0_to_7cm_c`, `soil_moisture_0_to_7cm_m3m3`, `et0_evapotranspiration_mm`
  - Cyclical Temporal: `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos`
- **Validation Split:** Spatial Block Holdout (28 Train Points / 245,952 rows vs. 8 Unseen Test Points / 70,272 rows).

---

## 3. Results & Metrics (Unseen Spatial Test Points)

| Metric | Coarse Baseline (No ML) | XGBoost Baseline (Exp 01) | Net Improvement (Δ) |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | 0.523°C | **0.391°C** | **+25.3% Error Reduction** |
| **Root Mean Squared Error (RMSE)** | 0.726°C | **0.543°C** | **+25.1% Error Reduction** |
| **Coefficient of Determination ($R^2$)** | 0.9923 | **0.9957** | **+0.0034 Variance Capture** |
| **Training Time (RTX 5050 GPU)** | — | **1.41 seconds** | — |

---

## 4. Top Feature Importances
1. `delta_elevation_m`: 24.61%
2. `theoretical_lapse_delta_c`: 11.38%
3. `precipitation_mm`: 7.46%
4. `et0_evapotranspiration_mm`: 5.04%
5. `hour_cos`: 4.88%

---

## 5. Conclusions & Limitations
- **Confirmed Viability:** Downscaling via residual regression successfully beat the coarse baseline by +25.1%.
- **Under-Convergence:** Test RMSE was still decreasing at round 299 (`0.54323`), indicating that 300 trees was insufficient to reach the global minimum.
- **Missing Physics:** Lacked higher-order vapor pressure deficit (VPD) and nocturnal valley pooling terms.
