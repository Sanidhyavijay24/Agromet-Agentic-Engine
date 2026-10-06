# Experiment 15: Pan-India 15 Agro-Climatic Zone (ACZ) Fleet Training & Evaluation

> **Experiment ID:** `EXP-15-PAN-INDIA-FLEET-GPU`  
> **Date Executed:** September 29, 2026  
> **Status:** 🏆 **ALL 15 ZONES VERIFIED & OPERATIONAL**  
> **Compute Infrastructure:** NVIDIA CUDA GPU · 31.4 minutes batch wall-clock time  
> **Standard:** ICAR (Indian Council of Agricultural Research) & Planning Commission 15 ACZ Framework  
> **Total Records Trained & Evaluated:** **47,342,880 records** (3,156,192 per zone across 10 years: 2015–2024)

---

## 1. Executive Summary & Fleet Metrics

We successfully completed the end-to-end batch training and holdout evaluation of the **Pan-India 15 Agro-Climatic Zone Downscaling Fleet**.

Across all 15 climatic regimes—from the steep relief of the Western Himalayas to the maritime boundary layers of the Andaman Islands and the arid thermal inertia of the Thar Desert—the partitioned XGBoost ensemble achieves a macro fleet error reduction of **+58.9%** ($2.772^\circ\text{C} \to \mathbf{0.953^\circ\text{C}}$ RMSE, **$0.653^\circ\text{C}$ MAE**).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PAN-INDIA FLEET PERFORMANCE SUMMARY                             │
├───────────────────────────────┬───────────────────────────────┬────────────────────────┤
│ Coarse Baseline RMSE: 2.772°C │ ML Downscaled RMSE: 0.953°C   │ ML Downscaled MAE: 0.653°C
├───────────────────────────────┼───────────────────────────────┼────────────────────────┤
│ Macro Error Drop: +58.9%      │ Mean Test R²: 0.9745          │ Total Records: 47,342,880
└───────────────────────────────┴───────────────────────────────┴────────────────────────┘
```

---

## 2. Complete 15 Agro-Climatic Zone Leaderboard

| Zone ID | Agro-Climatic Zone | Focus Domain | Baseline RMSE | ML Test RMSE | Test MAE | Test $R^2$ | RMSE Drop (Gain %) | Trees | Train Time | Verification Status |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ACZ-01** | Western Himalayan | Kangra–Kullu–Mandi | 11.797°C | **4.636°C** | 3.117°C | 0.8914 | **+60.7%** | 704 | 20.5s | **VERIFIED** |
| **ACZ-02** | Eastern Himalayan | Brahmaputra Valley | 8.649°C | **1.045°C** | 0.711°C | 0.9867 | **+87.9%** | 5,949 | 143.7s | **VERIFIED** |
| **ACZ-03** | Lower Gangetic | Bengal Delta Basin | 0.944°C | **0.485°C** | 0.346°C | 0.9920 | **+48.7%** | 5,999 | 123.8s | **VERIFIED** |
| **ACZ-04** | Middle Gangetic | Bihar–Eastern UP | 1.051°C | **0.461°C** | 0.334°C | 0.9950 | **+56.2%** | 5,999 | 126.0s | **VERIFIED** |
| **ACZ-05** | Upper Gangetic | Western UP Doab | 2.353°C | **1.089°C** | 0.594°C | 0.9799 | **+53.7%** | 5,999 | 126.4s | **VERIFIED** |
| **ACZ-06** | Trans-Gangetic | Punjab–Haryana | 3.064°C | **0.864°C** | 0.562°C | 0.9895 | **+71.8%** | 5,999 | 131.2s | **VERIFIED** |
| **ACZ-07** | Eastern Plateau | Chota Nagpur | 1.478°C | **0.637°C** | 0.475°C | 0.9900 | **+56.9%** | 5,997 | 126.0s | **VERIFIED** |
| **ACZ-08** | Central Plateau | Malwa Plateau | 1.066°C | **0.640°C** | 0.438°C | 0.9911 | **+40.0%** | 5,999 | 130.6s | **VERIFIED** |
| **ACZ-09** | Western Plateau | Maharashtra Deccan | 2.149°C | **0.798°C** | 0.594°C | 0.9778 | **+62.9%** | 4,795 | 107.5s | **VERIFIED** |
| **ACZ-10** | Southern Plateau | Telangana–Rayalaseema | 1.241°C | **0.503°C** | 0.381°C | 0.9895 | **+59.5%** | 5,997 | 127.1s | **VERIFIED** |
| **ACZ-11** | East Coast Plains | Krishna Delta & AP | 1.400°C | **0.721°C** | 0.488°C | 0.9726 | **+48.5%** | 5,997 | 125.9s | **VERIFIED** |
| **ACZ-12** | West Coast & Ghats | Konkan–Western Ghats | 2.606°C | **0.697°C** | 0.491°C | 0.9688 | **+73.2%** | 5,669 | 123.5s | **VERIFIED** |
| **ACZ-13** | Gujarat Plains | Saurashtra–C. Gujarat | 1.310°C | **0.668°C** | 0.501°C | 0.9878 | **+49.1%** | 5,990 | 130.5s | **VERIFIED** |
| **ACZ-14** | Western Dry | Semi-Arid Rajasthan | 1.274°C | **0.654°C** | 0.490°C | 0.9932 | **+48.6%** | 5,999 | 130.4s | **VERIFIED** |
| **ACZ-15** | Islands Region | Andaman Maritime Arc | 1.197°C | **0.403°C** | 0.274°C | 0.9390 | **+66.3%** | 5,003 | 107.5s | **VERIFIED** |

---

## 3. How the Models Are Trained: Deep Technical Breakdown

Every model in the 15-zone fleet was trained using the exact same rigorous, physics-grounded pipeline:

```
                  10-Year Hourly Reanalysis (2015–2024, 3.15M Rows/Zone)
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
                    │  Leave-One-Out (LOO) Spatial Baseline Engine   │
                    │  T_baseline = (Sum(T) - T_local) / (N - 1)     │
                    │  Residual Target: R = T_local - T_baseline     │
                    └───────────────────────┬────────────────────────┘
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
                    │   28 Leak-Free Features Extracted:             │
                    │   - VPD computed strictly from T_baseline      │
                    │   - Continuous 30m SRTM Horn's slope & aspect  │
                    │   - 2000m multi-scale TPI                      │
                    │   - Soil-air gradient & sloped insolation      │
                    └───────────────────────┬────────────────────────┘
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
                    │  Strict Spatial 3-Way Block Holdout Split      │
                    │  - Train: 25 Stations (2,191,800 rows)         │
                    │  - Validation: 4 Stations (350,688 rows)       │
                    │  - Holdout Test: 7 Stations (613,704 rows)     │
                    └───────────────────────┬────────────────────────┘
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
                    │  Deepened XGBoost Regressor (CUDA GPU)         │
                    │  - Max Depth: 9 | Learning Rate: 0.03          │
                    │  - Max Trees: 6,000 | Early Stopping: 200      │
                    │  - Subsample: 0.85 | Colsample: 0.85           │
                    └───────────────────────┬────────────────────────┘
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
                    │  Regional Anchor Payload & Serialization       │
                    │  - 8-point elevation anchor set matching mean  │
                    │  - Serialized to residual_model_acz_{id}.joblib│
                    └────────────────────────────────────────────────┘
```

### A. The Target Variable: Physical Leave-One-Out Residual Anomaly ($R$)
Rather than training the trees to predict raw Celsius temperature directly (which causes spatial memorization and over-relies on trivial seasonal cycles), each model predicts the **physics-guided microclimate residual anomaly**:
$$R = T_{\text{local}} - T_{\text{baseline\_LOO}}$$
where $T_{\text{baseline\_LOO}}$ is the un-leaked spatial Leave-One-Out mean of surrounding stations. At inference time, the model predicts $\hat{R}$ and reconstructs hyperlocal temperature:
$$T_{\text{downscaled}} = T_{\text{baseline}} + \text{clamp}(\hat{R}, -12^\circ\text{C}, +12^\circ\text{C})$$

### B. Strict Zero-Target-Leakage Feature Space (28 Features)
1. **Atmospheric Baseline:** $T_{\text{baseline\_LOO}}$, $\text{RH}_{2\text{m}}$, Surface Pressure, Wind Speed, Precipitation.
2. **Leak-Free VPD:** Vapour Pressure Deficit computed via Tetens equation strictly using $T_{\text{baseline\_LOO}}$ and $\text{RH}$ (preventing the algebraic recovery of $T_{\text{local}}$).
3. **Continuous 30m USGS SRTM DEM Morphology:** Bilinear elevation, Horn's $3\times 3$ slope gradient in degrees, north-azimuth downslope aspect decomposed into cyclical $\sin/\cos$, and 2,000m multi-scale Topographic Position Index (TPI).
4. **Boundary Layer Sensible & Latent Heat Flux:**
   - `soil_air_thermal_gradient`: $T_{\text{soil\_0-7cm}} - T_{\text{baseline}}$ (captures sensible heat flux into the boundary layer).
   - `latent_cooling_potential`: $\text{ET}_0 \times \text{SM}_{\text{0-7cm}}$ (evapotranspirative surface cooling).
   - `nocturnal_inversion_index`: Cold air drainage pooling weighted by nocturnal hours and low wind.
   - `sloped_solar_insolation`: Solar radiation modulated by terrain slope angle and solar azimuth alignment.
5. **Temporal Diurnal & Seasonal Cycles:** $\sin/\cos$ harmonic encodings for hour-of-day and day-of-year.

### C. Spatial Holdout Validation Scheme
To prove genuine spatial generalization:
* Data is **never** split randomly by row (which would cause temporal autocorrelation leakage).
* The 36 stations in each $2^\circ \times 2^\circ$ domain are partitioned into 25 training stations, 4 early-stopping validation stations, and 7 strictly unseen spatial test stations (**613,704 holdout records per zone**).
* All test metrics reported above are evaluated exclusively on stations the model never saw during training.

### D. Serving-Time Regional Anchor Contract
To prevent train/inference baseline distribution shift, each serialized `.joblib` model contains an embedded payload of **8 regional anchor coordinates** whose average elevation matches the domain's ground mean ($\pm 10\text{m}$). Live inferences query these 8 anchor points to construct the baseline $T_{\text{baseline}}$ in real time.

---

## 4. Production Artifact Manifest

All 15 serialized production models are stored and ready in [`services/ml_downscaler/artifacts/zones/`](file:///c:/Codes/SIH/SIH/services/ml_downscaler/artifacts/zones/):

```text
services/ml_downscaler/artifacts/zones/
├── residual_model_acz_01.joblib (Western Himalayan)
├── residual_model_acz_02.joblib (Eastern Himalayan)
├── residual_model_acz_03.joblib (Lower Gangetic)
├── residual_model_acz_04.joblib (Middle Gangetic)
├── residual_model_acz_05.joblib (Upper Gangetic)
├── residual_model_acz_06.joblib (Trans-Gangetic)
├── residual_model_acz_07.joblib (Eastern Plateau)
├── residual_model_acz_08.joblib (Central Plateau)
├── residual_model_acz_09.joblib (Western Plateau)
├── residual_model_acz_10.joblib (Southern Plateau)
├── residual_model_acz_11.joblib (East Coast Plains)
├── residual_model_acz_12.joblib (West Coast & Ghats)
├── residual_model_acz_13.joblib (Gujarat Plains)
├── residual_model_acz_14.joblib (Western Dry - Rajasthan)
└── residual_model_acz_15.joblib (Islands Region)
```
