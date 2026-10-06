# Project Context: Agromet Agentic Engine (A²E)

## 1. Project Overview

- **Identity:** **Agromet Agentic Engine (A²E)**
- **Mission:** Production-grade 1 km² physics-guided microclimate downscaling framework coupled with an autonomous **Agentic AI Agrometeorological Advisory System** powered by **Google Gemini Flash**.
- **Kaggle Dataset Hub:** [Pan-India 15 ACZ 1km Microclimate Dataset & Models](https://www.kaggle.com/datasets/sanidhyavijay24/pan-india-15-acz-1km-microclimate-dataset-and-models)
- **Flagship Deployment Strategy:**
  - **Zone XIV (Western Dry / Rajasthan):** Flagship in-memory 1km downscaling engine (`residual_model_acz_14.joblib`, 195MB) active locally for sub-50ms live response times.
  - **Remaining 14 ICAR ACZ Datasets & Models:** Hosted on Kaggle Hub with spatial routing contracts.
  - **Clean Lean Footprint:** Offloaded ~3.5 GB of raw training Parquets to Kaggle, keeping only 30m USGS SRTM DEM rasters and flagship serving artifacts locally.

---

## 2. Tech Stack

- **ML & Physics Downscaler:** Python 3.11/3.14, FastAPI, Pydantic v2, XGBoost, Scikit-learn, Joblib, NumPy, Pandas.
- **Data & Rasters:** USGS 30m SRTM DEM (`.hgt`), Horn's slope, aspect azimuth, multi-scale Topographic Position Index (TPI), solar GHI, soil moisture telemetry.
- **Autonomous Agent:** ReAct engine powered by **Google Gemini Flash** (`gemini-flash-lite-latest` / `gemini-3.7-flash`) with structured Pydantic v2 schemas and deterministic fallback synthesis.
- **Frontend Stack:** **Pure Bun 1.3.12 + React 19 + TypeScript + Lucide** (sub-50ms bundle build time, Swiss Brutalist Coffee & Sand design tokens, zero Vite/npm dependencies).
- **Containerization:** Multi-stage Dockerfile (<350MB) + `docker-compose.yml` for 1-command deployment.

---

## 3. Directory Architecture

```
agromet-agentic-engine/
├── services/
│   ├── agent/                      # Autonomous ReAct advisory engine & specialized tools
│   │   ├── core.py                 # Multi-step ReAct agent orchestrator (Gemini Flash)
│   │   ├── schemas.py              # Strict Pydantic v2 schemas (AgrometActionPlan, AgentTrace)
│   │   └── tools/                  # Specialized tool suite
│   │       ├── downscaler_tool.py  # 1km microclimate temperature & RH downscaling
│   │       ├── indices_tool.py     # Physical indices (VPD, GDD, Spray Windows, Inversions)
│   │       ├── agronomy_tool.py    # ICAR physiological threshold knowledge base
│   │       └── soil_tool.py        # 30m DEM elevation + 0-7cm soil moisture context
│   ├── ml_downscaler/              # 28-feature physics engine & Zone XIV model
│   │   ├── artifacts/zones/        # residual_model_acz_14.joblib (195MB)
│   │   ├── inference.py            # Low-latency inference handler & in-memory cache
│   │   └── features.py             # 28-feature physics pipeline
│   ├── ingestion/                  # Open-Meteo, USGS SRTM DEM, and satellite telemetry clients
│   └── api/                        # FastAPI REST API endpoints
├── frontend/                       # Bun + React + TS dashboard
├── data/
│   └── raw/dem/                    # 30m USGS SRTM .hgt elevation rasters
├── experiments/                    # 16 complete empirical experiment reports (01 to 15)
└── tests/                          # Comprehensive pytest & bun test suites
```

---

## 4. Empirical Performance Benchmarks

- **Zone XIV (Western Dry / Rajasthan - Flagship):**
  - Raw Baseline RMSE: `0.746 °C`
  - ML Downscaled RMSE: `0.446 °C`
  - **Verified Error Drop:** **+40.2% RMSE Reduction** (MAE `0.319 °C`, $R^2 = 0.9967$)
  - Model: 5,555 trees on authentic 30m USGS SRTM DEM with zero target leakage.
- **Pan-India 15 ACZ Macro Average:**
  - Raw Baseline RMSE: `2.772 °C`
  - ML Downscaled RMSE: `0.953 °C`
  - **Macro Average Error Drop:** **+58.9% RMSE Reduction** (MAE `0.653 °C`, Mean $R^2 = 0.9745$)
  - Spans 47.3 million hourly rows across 10 years (2015–2024). High-relief mountain zones achieve >70–87% error drop, while plains and arid zones achieve ~40–56%.

---

## 5. Feature Status Checklist

- [x] **28-Feature Physics Pipeline:** Elevation lapse, TPI, Horn's slope, aspect azimuth, VPD, nocturnal inversion index, solar geometry, thermal inertia.
- [x] **Zone XIV Champion Model:** 195MB XGBoost model (5,555 trees) in `services/ml_downscaler/artifacts/zones/residual_model_acz_14.joblib`.
- [x] **Pan-India Kaggle Hub Publishing:** 47.3M 10-year dataset and 15 models published to Kaggle.
- [x] **ReAct Agromet Agent:** Multi-step autonomous tool-calling with bilingual English/Hindi structured advisories.
- [x] **FastAPI REST API:** Full endpoints (`/api/v1/forecast`, `/api/v1/advisory`, `/health`) with rate limiting and input validation.
- [x] **Bun Frontend Dashboard:** Swiss Brutalist interface with real-time psychrometric charts, farm coordinate selection, and agent trace rendering.
- [x] **Quality & Documentation Integrity:** All 16 experiment markdown files fully sanitized, free of legacy terms, and calibrated to empirical metrics.
- [x] **Automated Testing:** 69 backend tests passing (0 failures), 2 frontend bun tests passing.

---

## 6. Technical Debt & Maintenance

- None outstanding. All data leakage, synthetic coordinate approximations, and legacy references have been completely resolved.
