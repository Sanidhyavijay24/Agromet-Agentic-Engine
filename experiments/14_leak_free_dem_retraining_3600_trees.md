# Experiment 14: Leak-Free 10-Year Converged Retraining with Real 30m DEM (5,555 Trees)

> **Experiment ID:** Exp 14 (Converged Champion)  
> **Date:** 2026-09-27  
> **Status:** Fully Converged & Verified (Early Stopped at step 5555/6000)  
> **Focus Domain:** ACZ-14 (Western Dry Region — Jaipur / Chaksu / Alwar, Rajasthan)  
> **Production Champion Artifact:** [`services/ml_downscaler/artifacts/residual_model.joblib`](file:///c:/Codes/SIH/SIH/services/ml_downscaler/artifacts/residual_model.joblib)

---

## 1. Executive Summary & Audit Pre-flight Resolution

This experiment establishes the definitive production champion for ACZ-14 following the complete resolution of all four audit findings:

1. **VPD Target Leakage Eliminated (F22):** Recomputed all vapor pressure deficit and latent cooling potential features strictly from `baseline_temp_c` in [`data/raw/jaipur_10yr_training_data_clean.parquet`](file:///c:/Codes/SIH/SIH/data/raw/jaipur_10yr_training_data_clean.parquet). Verified with zero leakage via [`tests/test_no_target_leakage.py`](file:///c:/Codes/SIH/SIH/tests/test_no_target_leakage.py).
2. **Authentic 30m USGS SRTM DEM Features (F17 / F3):** Analytical slope and aspect approximations replaced with genuine Horn’s $3\times3$ kernel slope, true north-azimuth aspect components (`aspect_sin`, `aspect_cos`), and multi-scale 2,000m Topographic Position Index (TPI) from authentic 1-arc-second `.hgt` rasters.
3. **Serving Baseline Parity (F5):** Embedded the **8-point anchor contract** into the model artifact. The selected anchor set achieves a mean elevation of **374.8 m** against the true domain mean of **373.9 m** ($\Delta = 0.89\text{ m}$), eliminating the runtime forecast displacement.
4. **Natural Convergence Achieved (F8):** Lifted `n_estimators` to **6,000 trees** (with `early_stopping_rounds = 200`). The model reached its global loss minimum at iteration **5,555**, achieving a verified **+40.2% RMSE error drop** and **+40.2% MAE error drop** on 613,704 unseen spatial holdout records.

---

## 2. Dataset & Partitioning Specification

- **Clean Dataset Path:** `data/raw/jaipur_10yr_training_data_clean.parquet` (3,156,192 rows, 52 columns, 36 spatial points).
- **Temporal Span:** 2015-01-01 00:00 UTC to 2024-12-31 23:00 UTC (10 full calendar years, 0 nulls).
- **Spatial 3-Way Partitioning:**
  - **Train Set (25 points):** 2,191,800 rows ($69.4\%$) — Used for gradient boosting tree construction.
  - **Validation Set (4 points):** 350,688 rows ($11.1\%$) — Monitored strictly for early stopping loss.
  - **Test Set (7 points):** 613,704 rows ($19.4\%$) — Untouched spatial holdout for honest metric reporting.

---

## 3. Quantitative Evaluation on Unseen Spatial Holdout (10-Year Horizon)

```
================================================================================
      QUANTITATIVE EVALUATION ON UNSEEN SPATIAL HOLDOUT POINTS (10-YEAR HORIZON)
================================================================================
```

| Metric | Coarse Baseline (No ML) | Downscaled XGBoost (Exp 14 Champion) | Honest Improvement (Δ) |
|:---|:---:|:---:|:---:|
| **Mean Absolute Error (MAE)** | 0.534°C | **0.319°C** | **+40.2% Error Drop** |
| **Root Mean Squared Error (RMSE)** | 0.746°C | **0.446°C** | **+40.2% Error Drop** |
| **Coefficient of Determination ($R^2$)** | 0.9908 | **0.9967** | **+0.0059 Variance Captured** |
| **Optimal Trees (Early Stopped)** | — | **5,555 / 6,000 trees** | Trained in **129.21s** (CUDA GPU) |

---

## 4. Top Physical & Agro-Meteorological Feature Importances

| Rank | Feature Name | Relative Importance | Physical / Agronomic Mechanism |
|:---:|:---|:---:|:---|
| **1** | `delta_elevation_m` | **15.34%** | Microclimate elevation relief relative to regional mean ($z_{\text{local}} - \bar{z}$) |
| **2** | `elevation_m` | **11.74%** | Absolute topographical altitude ASL from USGS 30m SRTM DEM |
| **3** | `theoretical_lapse_delta_c` | **7.02%** | Environmental lapse rate anomaly ($-6.5^\circ\text{C}/\text{km}$) |
| **4** | `soil_air_thermal_gradient` | **6.96%** | Ground sensible heat flux ($T_{\text{soil\_0-7cm}} - T_{\text{baseline}}$) |
| **5** | `precipitation_mm` | **5.25%** | Rain-induced evaporative cooling & boundary layer damping |
| **6** | `hour_cos` | **4.13%** | Diurnal solar heating cycle phase alignment |
| **7** | `surface_pressure_hpa` | **3.91%** | Barometric pressure & local boundary layer depth |
| **8** | `wind_speed_10m_kmh` | **3.71%** | Boundary layer mechanical mixing vs nocturnal stratification |
| **9** | `relative_humidity_2m_pct` | **3.68%** | Atmospheric moisture content & vapor pressure |
| **10** | `nocturnal_inversion_index` | **3.22%** | Valley cold-air drainage pooling under calm nocturnal wind |

---

## 5. Summary & SIH Defense Notes

The model converged cleanly at **5,555 trees** with early stopping triggering at step 5,755. This confirms that extending `n_estimators` unlocked an additional **+2.3%** true performance gain ($37.9\% \to 40.2\%$) over the 3,600-tree ceiling, while remaining 100% leak-free and grounded in authentic 30m DEM terrain physics.
