# Agromet Agentic Engine (A²E)

[![Kaggle Dataset & Models](https://img.shields.io/badge/Kaggle-Pan--India_15_ACZ_Models-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/sanidhyavijay24/pan-india-15-acz-1km-microclimate-dataset-and-models)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![Agromet Agentic Engine System Architecture](assets/a2e_system_banner.jpg)

A physics-guided microclimate downscaling framework and agromet advisory agent for Indian agriculture. It takes coarse weather forecasts (~25 km resolution from Open-Meteo / ERA5-Land) and downscales them to ~1 km resolution using 30m USGS SRTM terrain features and gradient boosting. A ReAct agent (Gemini Flash) then uses the downscaled weather alongside agronomic rules to generate farm-level advice for irrigation, frost risk, and pesticide spray windows.

---

## How It Works

1. **Target & Baseline:** The training target is Open-Meteo historical archive / ERA5-Land reanalysis grid data across 10 years (2015–2024). The baseline is a leave-one-out spatial neighbor average with standard environmental lapse rate adjustment (`-0.0065 °C/m`). The model learns spatial residuals (`R = T_local - T_baseline_LOO`).
2. **Physics Features (28 inputs):** Elevation differences, multi-scale Topographic Position Index (TPI), Horn's slope, aspect azimuth, solar elevation angle, vapor pressure deficit (VPD), and thermal inertia lag.
3. **Agromet Advisory Agent:** A multi-step ReAct agent with tools for crop thresholds (ICAR rules), psychrometric calculations (Delta-T, VPD, GDD), and bilingual synthesis (English/Hindi).

```
[ Coarse NWP Forecast ] + [ 30m USGS SRTM DEM ]
           │
           ▼
[ 28-Feature Physics Pipeline ] ──► [ XGBoost Residual Downscaler ] ──► 1 km Microclimate
                                                                              │
                                                                              ▼
[ ReAct Agent (Gemini Flash) ] ◄── [ Crop Agronomy + Delta-T / Soil Tools ] ──┘
           │
           ▼
[ Farm-Level Advisory: Spray Windows, Frost, Deficit Irrigation ]
```

---

## Benchmark Results

The flagship model covers **Zone XIV (Western Dry / Rajasthan)**, trained on 10 years of hourly data (3.15M rows) with 5,555 trees and evaluated on an 8-point spatial holdout:

| Model | Features | Baseline RMSE | ML Test RMSE | Test MAE | Test R² | Error Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Coarse Baseline (LOO Lapse)** | - | 0.746 °C | 0.746 °C | 0.538 °C | 0.9908 | 0.0% |
| **Linear Analytical Lapse (-6.5 °C/km)** | 1 | 0.746 °C | 0.648 °C | 0.457 °C | 0.9939 | +6.9% |
| **Random Forest (300 trees)** | 21 | 0.746 °C | 0.482 °C | 0.344 °C | 0.9966 | +30.7% |
| **LightGBM (1,200 trees)** | 21 | 0.746 °C | 0.444 °C | 0.315 °C | 0.9967 | +36.2% |
| **CatBoost (1,200 trees)** | 21 | 0.746 °C | 0.449 °C | 0.320 °C | 0.9967 | +35.5% |
| **Flagship XGBoost (5,555 trees)** | **28** | **0.746 °C** | **0.446 °C** | **0.319 °C** | **0.9967** | **+40.2%** |

Across all 15 Agro-Climatic Zones of India (47.3M rows), the simple unweighted mean baseline is 2.758 °C, downscaling to 0.912 °C ML RMSE (+60.6% mean error drop). High-relief mountainous zones (Eastern/Western Himalayas) show large gains (+86% to +88%), while flatter plains show ~40% to ~56%. See [experiments/15_pan_india_fleet_training_results.md](experiments/15_pan_india_fleet_training_results.md) for the full 15-zone breakdown.

---

## Setup & Running

### 1. Installation
```bash
git clone https://github.com/Sanidhyavijay24/Agromet-Agentic-Engine.git
cd Agromet-Agentic-Engine

# Install Python dependencies
pip install -r requirements.txt
cp .env.example .env
```

### 2. Download Model Artifacts
The Zone XIV model (`residual_model_acz_14.joblib`, 195 MB) and full fleet are hosted on Kaggle:
```bash
# Download only Zone XIV model needed for running (195 MB)
python scripts/fetch_artifacts.py

# Optional: Download all 15 zone models and full dataset (3.5 GB)
python scripts/fetch_artifacts.py --all

# Or single-file CLI download
kaggle datasets download -d sanidhyavijay24/pan-india-15-acz-1km-microclimate-dataset-and-models -f residual_model_acz_14.joblib -p services/ml_downscaler/artifacts/zones/
```

### 3. Run Tests
```bash
# Run pytest (69 passed, 3 skipped with model artifact; 57 passed, 15 skipped without)
python -m pytest tests/ -v

# Frontend tests (Bun runtime)
cd frontend && bun test && cd ..
```

### 4. Start Services
```bash
# Backend API (port 8000)
uvicorn services.api.main:app --reload

# Frontend Dashboard (port 3000)
cd frontend && bun run start
```

---

## API Endpoints

### 1. Downscaled Forecast: `POST /api/v1/forecast`
```json
{
  "latitude": 26.85,
  "longitude": 75.80,
  "days": 7
}
```
Returns 168 hours of 1 km downscaled temperature, relative humidity, wind speed, vapor pressure deficit, and delta-T.

### 2. Agromet Advisory: `POST /api/v1/advisory`
```json
{
  "latitude": 26.85,
  "longitude": 75.80,
  "crop": "Mustard",
  "growth_stage": "Flowering",
  "soil_type": "Sandy Loam",
  "language": "en"
}
```
Returns ReAct agent thought trace, psychrometric risk checks, and concrete spray / irrigation advice.

---

## Limitations

- **Reanalysis target, not station data:** Ground-truth targets are ERA5-Land reanalysis via Open-Meteo. Validation against physical weather station observations is not done yet.
- **Single active zone in repo:** Only Zone XIV (Rajasthan) is stored locally. Models for the other 14 zones are in the Kaggle dataset bundle.
- **Evaluation splits:** Results reflect single spatial group holdouts (8 of 36 points) and an unseen 2025 temporal holdout.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
