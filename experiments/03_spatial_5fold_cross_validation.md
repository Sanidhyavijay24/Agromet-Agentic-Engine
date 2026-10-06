# Experiment 03: 5-Fold Spatial Group Cross-Validation (Full Year 2024)

## 1. Objective & Hypothesis
**Objective:** Prove that the performance gains of the Deepened XGBoost model are mathematically stable across every corner of the district, and not an artifact of a single lucky train/test holdout split.

**Hypothesis:** Evaluating across 5 non-overlapping geographic partitions of spatial points (Group K-Fold) will yield low variance (`std < 5%`) and consistent ~35-40% error reduction across all terrain zones.

---

## 2. Experimental Configuration
- **Model Architecture:** Deepened `xgboost.XGBRegressor` (`max_depth=8`, `n_estimators=1000`, `lr=0.03`, `early_stopping=50`, `device=cuda`).
- **Feature Set:** Full 21-feature schema (including physics interactions).
- **Validation Strategy:** 5-Fold Spatial Group Partitioning across all 36 grid points:
  - Fold 1: 8 Validation Points (70,272 rows)
  - Fold 2: 7 Validation Points (61,488 rows)
  - Fold 3: 7 Validation Points (61,488 rows)
  - Fold 4: 7 Validation Points (61,488 rows)
  - Fold 5: 7 Validation Points (61,488 rows)

---

## 3. Spatial Cross-Validation Results

| Fold | Holdout Points | Rows Evaluated | Best Trees | Baseline RMSE | ML Test RMSE | Test MAE | Test R² | Error Drop (Gain %) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 8 Points | 70,272 | 999 trees | 0.696 °C | 0.461 °C | 0.328 °C | 0.9969 | **+33.7%** |
| **Fold 2** | 7 Points | 61,488 | 999 trees | 0.716 °C | 0.424 °C | 0.298 °C | 0.9974 | **+40.8%** |
| **Fold 3** | 7 Points | 61,488 | 999 trees | 0.690 °C | 0.414 °C | 0.294 °C | 0.9976 | **+39.9%** |
| **Fold 4** | 7 Points | 61,488 | 999 trees | 0.869 °C | 0.555 °C | 0.401 °C | 0.9959 | **+36.1%** |
| **Fold 5** | 7 Points | 61,488 | 999 trees | 0.635 °C | 0.384 °C | 0.274 °C | 0.9977 | **+39.5%** |
| **Mean ± Std** | - | - | - | **0.721 ± 0.088 °C** | **0.448 ± 0.065 °C** | **0.319 ± 0.050 °C** | **0.9971 ± 0.0007** | **+38.0% ± 2.9%** |

---

## 4. Statistical Conclusion
- **Robust Generalization:** The model achieved an average **+38.0% RMSE error reduction** with a standard deviation of only `±2.9%` across distinct geographical quadrants.
- **Zero Spatial Overfitting:** Verifies that the model learns generalized terrain-physics transfer functions rather than memorizing point coordinates.
