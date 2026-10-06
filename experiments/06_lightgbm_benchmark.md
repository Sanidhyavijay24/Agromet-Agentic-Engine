# Experiment 06: LightGBM Regressor (Leaf-wise GBDT)

## 1. Overview & Objective
- **Experiment ID:** `EXP-06`
- **Model Architecture:** LightGBM Regressor (`n_estimators=1200`, `best_iteration=1200`, `num_leaves=63`, `learning_rate=0.03`).
- **Objective:** Evaluate leaf-wise histogram tree growth against XGBoost depth-wise tree growth, analyzing training throughput, memory footprint, and test generalization on 21 physics-guided features.
- **Validation Scheme:** Spatial Block Holdout Split (8 unseen test points, 70,272 hourly samples across all 2024).

---

## 2. Quantitative Benchmark Results

| Model Architecture | Train Time (s) | Inf Latency (ms) | Test MAE (°C) | Test RMSE (°C) | Test $R^2$ | RMSE Drop (Δ%) | MAE Drop (Δ%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Coarse Baseline** | — | — | 0.523°C | 0.726°C | 0.9923 | 0.0% | 0.0% |
| **LightGBM (Leaf-wise GBDT)** | 4.11s | 0.0066ms | 0.323°C | 0.451°C | 0.9970 | **+37.8%** | **+38.2%** |
| **🏆 Champion (Exp 02 Deepened XGBoost)** | **8.03s (GPU)** | **0.003ms** | **0.311°C** | **0.439°C** | **0.9972** | **+39.5%** | **+40.4%** |

---

## 3. Top Feature Importances (Split Gain)
- **delta_elevation_m**: 19.37%
- **relative_humidity_2m_pct**: 9.01%
- **doy_sin**: 8.75%
- **baseline_temp_c**: 8.53%
- **theoretical_lapse_delta_c**: 6.65%
- **doy_cos**: 6.44%

---

## 4. Key Takeaways
1. **Histogram Efficiency:** LightGBM trains rapidly with histogram binning and leaf-wise splitting, converging at **1200 iterations**.
2. **Predictive Performance:** LightGBM achieves **+37.8% RMSE error reduction**, performing neck-and-neck with XGBoost.
3. **Inference Latency:** Ultra-low inference latency of **0.0066 ms/sample**, making it an optimal candidate for high-throughput batch downscaling.
