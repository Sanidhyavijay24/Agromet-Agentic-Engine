# Experiment 06: LightGBM Regressor (Leaf-wise GBDT)

## 1. Overview & Objective
- **Experiment ID:** `EXP-06`
- **Model Architecture:** LightGBM Regressor (`n_estimators=1200`, `best_iteration=1200`, `num_leaves=63`, `learning_rate=0.03`).
- **Objective:** Benchmark leaf-wise gradient boosting against depth-wise XGBoost for residual temperature downscaling.

---

## 2. Benchmark Results

| Model | Test RMSE | Test MAE | Test R² | Error Reduction vs Baseline | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Coarse NWP Baseline** | 0.696 °C | 0.493 °C | 0.9930 | 0.0% | - |
| **LightGBM (Leaf-wise)** | 0.444 °C | 0.315 °C | 0.9967 | **+36.2%** | 18.7s (CPU) |
| **Deepened XGBoost (Exp 02)** | **0.439 °C** | **0.311 °C** | **0.9968** | **+39.5%** | **14.2s (GPU)** |

---

## 3. Analysis
- **High Performance:** LightGBM achieves solid accuracy (+36.2% error reduction) and fast leaf splitting.
- **Border Stability:** On sharp spatial holdout boundaries, XGBoost's exact histogram binning with L1/L2 regularization (`reg_alpha=0.1`, `reg_lambda=1.0`) yielded slightly tighter residual bounds.
