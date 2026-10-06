# Experiment 05: Random Forest Regressor (Bagging Ensemble)

## 1. Overview & Objective
- **Experiment ID:** `EXP-05`
- **Model Architecture:** Random Forest Regressor (`n_estimators=300`, `max_depth=14`, `max_features='sqrt'`).
- **Objective:** Evaluate bagging (Bootstrap Aggregation) against gradient boosting (XGBoost) for microclimate residual modeling to evaluate variance reduction across 21 physics-guided covariates.
- **Validation Scheme:** Spatial Block Holdout Split (8 unseen test points, 70,272 hourly samples across all 2024).

---

## 2. Quantitative Benchmark Results

| Model Architecture | Train Time (s) | Inf Latency (ms) | Test MAE (°C) | Test RMSE (°C) | Test $R^2$ | RMSE Drop (Δ%) | MAE Drop (Δ%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Coarse Baseline** | — | — | 0.523°C | 0.726°C | 0.9923 | 0.0% | 0.0% |
| **Random Forest (Bagging)** | 8.73s | 0.001ms | 0.426°C | 0.602°C | 0.9947 | **+17.0%** | **+18.6%** |
| **🏆 Champion (Exp 02 Deepened XGBoost)** | **8.03s (GPU)** | **0.003ms** | **0.311°C** | **0.439°C** | **0.9972** | **+39.5%** | **+40.4%** |

---

## 3. Top Feature Importances
- **theoretical_lapse_delta_c**: 12.16%
- **delta_elevation_m**: 11.81%
- **elevation_m**: 11.65%
- **surface_pressure_hpa**: 7.40%
- **soil_moisture_0_to_7cm_m3m3**: 5.33%
- **wind_speed_10m_kmh**: 4.98%

---

## 4. Key Takeaways
1. **Bagging vs Boosting Performance:** Random Forest achieved **+17.0% RMSE error reduction**, confirming robust non-linear modeling capability.
2. **Computational Trade-off:** While Random Forest achieves strong accuracy, XGBoost benefits from gradient-directed sequential error minimization and histogram GPU acceleration (8.03s vs 8.7s CPU).
3. **Ensemble Utility:** Random Forest predictions exhibit lower correlation with tree boosters, making it an excellent candidate for Stage 2B Stacking Ensemble blending.
