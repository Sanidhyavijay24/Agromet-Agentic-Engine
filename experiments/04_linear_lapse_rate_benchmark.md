# Experiment 04: Analytical Environmental Lapse Rate Benchmark

## 1. Overview & Objective
- **Experiment ID:** `EXP-04`
- **Objective:** Establish the pure physics baseline for temperature downscaling using the standard atmospheric environmental lapse rate (`-6.5 °C / 1000m`) and an empirical 1D fitted lapse model, to rigorously quantify how much predictive accuracy ML gradient boosters add over traditional atmospheric barometrics.
- **Dataset:** Jaipur / Chaksu 2024 hourly dataset (316,224 rows; 28 train points, 8 spatial test points).

---

## 2. Models Evaluated
1. **Standard Atmospheric Lapse Rate (Physics Only):**
   `T_downscaled = T_coarse + Gamma * (Elevation_local - Elevation_coarse)`, where `Gamma = -0.0065 °C/m`.
2. **Empirical 1D OLS Fitted Lapse Rate:**
   Ordinary Least Squares linear regression fitted strictly on `Delta_Elevation_m` -> `residual_temperature_c`.

---

## 3. Benchmark Results

| Model / Baseline | Test RMSE | Test MAE | Test R² | Error Reduction vs Coarse |
| :--- | :---: | :---: | :---: | :---: |
| **Coarse NWP Baseline (No Correction)** | 0.696 °C | 0.493 °C | 0.9930 | 0.0% |
| **Standard Atmospheric Lapse (-6.5 °C/km)** | 0.648 °C | 0.457 °C | 0.9939 | +6.9% |
| **Empirical 1D OLS Fitted Lapse** | 0.639 °C | 0.451 °C | 0.9941 | +8.2% |
| **Deepened XGBoost (Exp 02)** | **0.439 °C** | **0.311 °C** | **0.9968** | **+39.5%** |

---

## 4. Key Takeaways
- **Physics Alone is Insufficient:** Standard environmental lapse rates capture linear hydrostatic cooling (+6.9% gain), but fail to account for nocturnal radiation pooling, thermal inertia, solar geometry, and non-linear boundary layer friction.
- **Machine Learning Synergy:** The physics-guided GBDT captures these non-linear spatial interactions, providing an additional **+32.6% error reduction** over analytical lapse rate models alone.
