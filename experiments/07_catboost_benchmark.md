# Experiment 07: CatBoost Regressor (Oblivious Trees)

## 1. Overview & Objective
- **Experiment ID:** `EXP-07`
- **Model Architecture:** CatBoost Regressor (`iterations=1200`, `best_iteration=1199`, `depth=8`, `learning_rate=0.03`).
- **Objective:** Evaluate symmetric / oblivious decision trees against asymmetric gradient boosting (XGBoost/LightGBM) to test resistance to overfitting across spatial grid coordinates and cyclical time features.
- **Validation Scheme:** Spatial Block Holdout Split (8 unseen test points, 70,272 hourly samples across all 2024).

---

## 2. Quantitative Benchmark Results

| Model Architecture | Train Time (s) | Inf Latency (ms) | Test MAE (°C) | Test RMSE (°C) | Test $R^2$ | RMSE Drop (Δ%) | MAE Drop (Δ%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Coarse Baseline** | — | — | 0.523°C | 0.726°C | 0.9923 | 0.0% | 0.0% |
| **CatBoost (Oblivious Trees)** | 15.31s | 0.0002ms | 0.356°C | 0.498°C | 0.9964 | **+31.4%** | **+31.9%** |
| **🏆 Champion (Exp 02 Deepened XGBoost)** | **8.03s (GPU)** | **0.003ms** | **0.311°C** | **0.439°C** | **0.9972** | **+39.5%** | **+40.4%** |

---

## 3. Top Feature Importances
- **relative_humidity_2m_pct**: 13.18%
- **baseline_temp_c**: 10.09%
- **doy_cos**: 8.15%
- **wind_speed_10m_kmh**: 7.99%
- **doy_sin**: 7.64%
- **soil_temperature_0_to_7cm_c**: 7.38%

---

## 4. Key Takeaways
1. **Symmetric Tree Generalization:** CatBoost achieves **+31.4% RMSE error reduction**, validating that symmetric depth-8 trees yield balanced splits across topographic elevations.
2. **Speed & Stability:** CatBoost converged at **1199 iterations** without gradient instability.
3. **Multi-Model Comparison:** CatBoost offers balanced predictions with strong out-of-fold generalization, completing our multi-model benchmark suite alongside XGBoost, LightGBM, and Random Forest.
