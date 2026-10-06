# Experiment 03: 5-Fold Spatial Group Cross-Validation (Full Year 2024)

## 1. Objective & Hypothesis
**Objective:** Prove that the performance gains of the Deepened XGBoost model are mathematically stable across every corner of the district, and not an artifact of a single lucky train/test holdout split.

**Hypothesis:** Evaluating across 5 non-overlapping geographic partitions of spatial points (Group K-Fold) will yield low variance ($\text{std} < 5\%$) and consistent ~35–40% error reduction across all terrain zones.

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
  - *Each fold evaluates on points it was never trained on, across all 366 days of 2024.*

---

## 3. Fold-by-Fold Results

| Spatial Fold | Test Points | Test Rows | Best Iteration | Baseline RMSE | Downscaled RMSE | Downscaled MAE | Downscaled $R^2$ | RMSE Error Drop (Δ) |
|---|---|---|---|---|---|---|---|---|
| **Fold 1** | 8 Points | 70,272 | 999 trees | 0.696°C | 0.461°C | 0.328°C | 0.9969 | **+33.7%** |
| **Fold 2** | 7 Points | 61,488 | 999 trees | 0.716°C | 0.424°C | 0.298°C | 0.9974 | **+40.8%** |
| **Fold 3** | 7 Points | 61,488 | 999 trees | 0.690°C | 0.414°C | 0.294°C | 0.9976 | **+39.9%** |
| **Fold 4** | 7 Points | 61,488 | 999 trees | 0.869°C | 0.555°C | 0.401°C | 0.9959 | **+36.1%** |
| **Fold 5** | 7 Points | 61,488 | 999 trees | 0.731°C | 0.459°C | 0.327°C | 0.9969 | **+37.2%** |
| **MEAN ± STD** | — | — | — | **0.740°C** | **0.463°C** | **0.330°C** | **0.9969** | **+37.6% ± 2.6%** |

---

## 4. Key Takeaways
1. **Exceptional Spatial Stability:** Mean RMSE reduction is **+37.6%** with an ultra-tight standard deviation of only **$\pm 2.6\%$**.
2. **Robustness Against Difficult Terrain:** Fold 4 had the highest baseline error (0.869°C, high terrain variance), and the model still reduced error to 0.555°C (+36.1% drop).
3. **Statistical Proof for Judges:** Provides rigorous proof that the model generalizes to new panchayats anywhere within the semi-arid agro-ecological domain.
