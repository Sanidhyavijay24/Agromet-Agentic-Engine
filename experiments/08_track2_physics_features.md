# Experiment 08: Track 2 Advanced Topographic & Boundary Layer Physics

## 1. Overview & Objective
- **Experiment ID:** `EXP-08` (Track 2)
- **Objective:** Expand the residual feature matrix from 21 to **28 physics-guided features** by injecting terrain morphology (TPI, slope magnitude, aspect), sloped solar irradiance coupling, ground-air sensible heat gradients, and evaporative latent cooling potential.
- **Validation Scheme:** Spatial Block Holdout Split (8 unseen test points, 70,272 hourly samples across all 2024).

---

## 2. Quantitative Results & Comparison

| Model Architecture & Feature Set | Test MAE (°C) | Test RMSE (°C) | Test $R^2$ | RMSE Drop (Δ%) | MAE Drop (Δ%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Coarse Baseline (Spatial Mean)** | 0.523°C | 0.726°C | 0.9923 | 0.0% | 0.0% |
| **Exp 02 Champion (21 Base Features)** | 0.311°C | 0.439°C | 0.9972 | +39.5% | +40.4% |
| **🏆 Track 2 XGBoost (28 Advanced Features)** | **0.287°C** | **0.400°C** | **0.9977** | **+44.9%** | **+45.1%** |
| **Track 2 LightGBM (28 Advanced Features)** | 0.307°C | 0.425°C | 0.9974 | +41.5% | — |

---

## 3. Top Feature Importances (28 Physics Features)
- **delta_elevation_m**: 14.26% 
- **elevation_m**: 9.53% 
- **theoretical_lapse_delta_c**: 8.66% 
- **soil_air_thermal_gradient**: 6.09% ⭐ (New Track 2)
- **precipitation_mm**: 5.83% 
- **hour_cos**: 3.69% 
- **slope_magnitude_deg**: 3.38% ⭐ (New Track 2)
- **doy_cos**: 3.06% 
- **thermal_inertia_lag_3h**: 2.97% 
- **relative_humidity_2m_pct**: 2.90% 

---

## 4. Key Takeaways
1. **Impact of 28 Physics Features:** Adding high-order terrain position (TPI), soil-canopy thermal gradient, and sloped solar insolation pushed RMSE reduction to **+44.9%**.
2. **Ground Thermal Coupling:** The soil-to-air thermal gradient directly captures boundary layer sensible heat flux, providing the model with physical drivers for midday heat spikes.
3. **Terrain Disambiguation:** TPI successfully separates ridge ventilation from basin cold-air pooling, improving nighttime frost and valley temperature predictions.
