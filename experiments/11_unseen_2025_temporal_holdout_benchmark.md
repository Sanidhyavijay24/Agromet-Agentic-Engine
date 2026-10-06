# Experiment 11: Unseen 2025 Future Year Temporal & Spatio-Temporal Holdout Benchmark

## 1. Objective & Holdout Design
- **Objective:** Evaluate the active 10-year champion downscaler (`residual_model.joblib`, trained strictly on 2015-2024) on **100% unseen 2025 climate data** across all 36 spatial points in Jaipur / Chaksu.
- **Production Generalization Value:**
  - Standard ML models often overfit to multi-year climate cycles.
  - Evaluating on an entirely future year (2025) validates that the model generalizes to evolving weather regimes without temporal decay or drift.

---

## 2. Holdout Dataset Specification
- **Time Range:** January 1, 2025 00:00 UTC to August 31, 2025 23:00 UTC (5,832 hourly timesteps).
- **Spatial Points:** 36 points across the Jaipur Rural / Chaksu agricultural corridor.
- **Total Test Rows:** 209,952 unseen hourly observations.

---

## 3. Results on Unseen 2025 Data

| Test Partition | Records | Coarse Baseline RMSE | Model Test RMSE | Test MAE | Test R² | Error Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Temporal Only (Known Spatial Points)** | 163,296 | 0.684 °C | **0.412 °C** | **0.292 °C** | **0.9974** | **+39.8%** |
| **Spatio-Temporal (Unseen Points + Unseen 2025)** | 46,656 | 0.741 °C | **0.452 °C** | **0.324 °C** | **0.9967** | **+39.0%** |
| **Combined 2025 Holdout** | **209,952** | **0.697 °C** | **0.421 °C** | **0.299 °C** | **0.9972** | **+39.6%** |

---

## 4. Conclusion
- The physics-guided residual downscaler maintained a solid **+39.6% error reduction** on completely unseen future-year data, proving high operational stability and consistent error reduction across the 2025 evaluation period.
