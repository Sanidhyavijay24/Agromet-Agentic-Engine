# Experiment 05: Random Forest Regressor (Bagging Ensemble)

## 1. Overview & Objective
- **Experiment ID:** `EXP-05`
- **Model Architecture:** Random Forest Regressor (`n_estimators=300`, `max_depth=14`, `max_features='sqrt'`).
- **Objective:** Evaluate how classical bagging tree ensembles compare to gradient boosting architectures on the identical 21-feature physics matrix under spatial holdout.

---

## 2. Benchmark Results

| Model | Test RMSE | Test MAE | Test R² | Error Reduction vs Baseline | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Coarse NWP Baseline** | 0.696 °C | 0.493 °C | 0.9930 | 0.0% | - |
| **Random Forest (300 Trees)** | 0.482 °C | 0.344 °C | 0.9966 | **+30.7%** | 82.4s (CPU) |
| **Deepened XGBoost (Exp 02)** | **0.439 °C** | **0.311 °C** | **0.9968** | **+39.5%** | **14.2s (GPU)** |

---

## 3. Analysis
- **Bagging vs Boosting:** Random Forest demonstrates good generalization (+30.7% error reduction), but independent tree averaging struggles to resolve the sharp, fine-grained residual peaks captured by iterative gradient boosting.
- **Inference Footprint:** Random forest trees at depth 14 result in an 85 MB serialized model, compared to under 12 MB for regularized XGBoost.
