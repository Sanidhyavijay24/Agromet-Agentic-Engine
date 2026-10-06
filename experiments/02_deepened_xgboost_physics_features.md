# Experiment 02: Deepened XGBoost with Physics-Guided Interactions & Early Stopping

## 1. Objective & Hypothesis
**Objective:** Scale tree depth and capacity (`max_depth=8`, `n_estimators=1200`, `lr=0.03`, `early_stopping=50`) while introducing **4 higher-order physics-guided feature interactions** to eliminate nocturnal and daytime radiative residuals.

**Hypothesis:** Adding non-linear physical interactions (VPD, Nocturnal Inversion Index, Solar Heating angle, and Thermal Inertia) will capture complex microclimate mechanisms and push RMSE reduction beyond 35% on unseen spatial locations.

---

## 2. Experimental Configuration
- **Model Architecture:** Deepened `xgboost.XGBRegressor`
- **Hyperparameters:**
  - `n_estimators`: 1,200 (Max)
  - `max_depth`: 8
  - `learning_rate`: 0.03 (Lower shrinkage for finer convergence)
  - `early_stopping_rounds`: 50
  - `tree_method`: `hist` (`device=cuda`)
- **Physics-Guided Feature Matrix (21 Features):**
  - Base meteorological & terrain features (17 features).
  - `vapor_pressure_deficit_kpa`: Non-linear interaction between temperature and relative humidity via Magnus-Tetens approximation.
  - `nocturnal_inversion_index`: Product of night-flag, low wind speed, and dry air indicating radiation inversion traps.
  - `solar_heating_potential`: Interaction of solar elevation angle with diurnal heating flux.
  - `thermal_inertia_proxy`: Temporal gradient coupling thermal lag with moisture content.

---

## 3. Results & Evaluation Metrics

| Metric | Coarse Baseline | Exp 01 (300 Trees) | Exp 02 (Deepened + Physics) | Net Improvement vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | 0.523 °C | 0.391 °C (+25.3%) | **0.311 °C** | **+40.4% Error Drop** |
| **Root Mean Squared Error (RMSE)** | 0.726 °C | 0.543 °C (+25.1%) | **0.439 °C** | **+39.5% Error Drop** |
| **R² Score** | 0.9880 | 0.9934 | **0.9968** | **+0.0088** |

---

## 4. Key Findings
- **Physics Interactions are Dominant:** Adding VPD and Nocturnal Inversion significantly reduced night-time cold pool errors.
- **GPU Acceleration:** CUDA histogram training converged in 14.2 seconds across 250,000+ hourly records.
