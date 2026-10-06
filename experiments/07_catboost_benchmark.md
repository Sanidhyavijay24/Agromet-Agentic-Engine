# Experiment 07: CatBoost Regressor (Oblivious Trees)

## 1. Overview & Objective
- **Experiment ID:** `EXP-07`
- **Model Architecture:** CatBoost Regressor (`iterations=1200`, `best_iteration=1199`, `depth=8`, `learning_rate=0.03`).
- **Objective:** Benchmark symmetric (oblivious) decision trees against asymmetric gradient boosting for microclimate downscaling.

---

## 2. Benchmark Results

| Model | Test RMSE | Test MAE | Test R² | Error Reduction vs Baseline | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Coarse NWP Baseline** | 0.696 °C | 0.493 °C | 0.9930 | 0.0% | - |
| **CatBoost (Oblivious Trees)** | 0.449 °C | 0.320 °C | 0.9967 | **+35.5%** | 45.1s (GPU) |
| **Deepened XGBoost (Exp 02)** | **0.439 °C** | **0.311 °C** | **0.9968** | **+39.5%** | **14.2s (GPU)** |

---

## 3. Analysis
- **Symmetric Trees:** CatBoost's oblivious tree structure provides robust resistance to overfitting, but requires deeper architectures to capture localized micro-terrain elevation discontinuities.
