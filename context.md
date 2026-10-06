# Project Context: Agromet Agentic Engine (A²E)

## 1. Project Overview

- **Identity:** **Agromet Agentic Engine (A²E)**
- **Mission:** Production-grade 1 km² physics-guided microclimate downscaling framework coupled with an autonomous **Agentic AI Agrometeorological Advisory System** powered by **Google Gemini Flash (Free Tier)**.
- **Flagship Deployment Strategy:**
  - **Zone XIV (Western Dry / Rajasthan):** Flagship in-memory 1km downscaling engine (~165MB footprint) active locally for fast responses without runtime model downloading latency.
  - **Remaining 14 ICAR ACZ Models:** Hosted on the **Kaggle Model Hub** with full spatial routing contracts, ensuring lean server footprint and sub-second container startup.
  - **Clean Codebase:** Cleaned all legacy hackathon audit scripts, test dumps, and unused endpoints.

---

## 2. Tech Stack

- **ML & Physics Downscaler:** Python 3.11/3.14, FastAPI, Pydantic v2, XGBoost, Scikit-learn, Joblib, NumPy, Pandas.
- **Data & Rasters:** USGS 30m SRTM DEM (`.hgt` / GeoTIFF), Horn’s slope, aspect azimuth, multi-scale Topographic Position Index (TPI), solar GHI, soil moisture telemetry.
- **Autonomous Agent:** ReAct engine powered by **Google Gemini Flash** (`gemini-flash-lite-latest` / `gemini-3.7-flash`) with structured Pydantic v2 schemas and deterministic fallback synthesis.
- **Frontend Stack:** **Pure Bun 1.3.12 + React 19 + TypeScript + Lucide** (sub-50ms bundle build time, Swiss Brutalist Coffee & Sand design tokens, zero Vite/npm dependencies).
- **Containerization:** Multi-stage Dockerfile (<350MB) + `docker-compose.yml` for 1-command deployment to Render/HuggingFace Spaces/Railway/AWS.

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
│   ├── ingestion/                  # Open-Meteo, USGS SRTM DEM, and satellite telemetry clients
│   └── api/                        # FastAPI REST gateway
│       ├── main.py                 # Application root & CORS lifespan
│       ├── schemas.py              # Request/response validation contracts
│       └── routes/
│           ├── agent.py            # POST /advisory, POST /chat, GET /tools
│           ├── forecast.py         # GET /point, GET /zones/catalog, GET /zones/route
│           └── health.py           # GET /ping, GET /health
├── frontend/                       # Pure Bun + React 19 Frontend
│   ├── package.json                # Bun-only scripts (dev, build, test, start)
│   ├── tsconfig.json               # TypeScript strict configuration
│   ├── server.ts                   # Bun.serve production server with API reverse proxy
│   ├── public/                     # Static HTML template & built assets
│   │   ├── index.html
│   │   └── dist/                   # Bundled index.js (490KB) & index.css (7.9KB)
│   └── src/
│       ├── index.tsx               # React 19 client mount
│       ├── index.css               # Swiss brutalist design tokens & typography
│       ├── types/index.ts          # TypeScript interfaces matching backend Pydantic models
│       ├── services/api.ts         # Live FastAPI client + fallback generators
│       ├── components/
│       │   ├── Header.tsx          # Top navbar, tab switching, theme toggle
│       │   ├── MicroclimateComparator.tsx # 1km vs 25km comparator + 24h diurnal scrubber
│       │   ├── AgentTerminal.tsx   # Autonomous ReAct consultation & live tool traces
│       │   └── ZoneFleetHub.tsx    # 15 ACZ Pan-India Directory & Kaggle Hub
│       └── App.tsx                 # Root layout & view controller
├── data/
│   ├── processed/                  # Panchayat terrain metadata & DEM rasters
│   └── raw/                        # 15 ACZ 10-year training parquets & DEM rasters
├── scripts/                        # Ingestion, training, and benchmarking utilities
├── tests/                          # 72 passing backend tests (pytest) + 2 frontend tests (bun test)
├── Dockerfile                      # Production container (<350MB)
├── docker-compose.yml              # 1-command deployment
├── context.md                      # Single Source of Truth ledger
└── README.md
```

---

## 4. Test Verification Summary

- **Backend Test Suite:** `72 passed, 0 failed` (`python -m pytest tests/ -v`, execution time: ~29s).
- **Frontend Test Suite:** `2 passed, 0 failed` (`bun test`, execution time: ~33ms).
- **Frontend Bundle Performance:** Bundled in 52ms via native Bun bundler (`index.js`: 490KB minified, `index.css`: 7.9KB).

---

## 5. Deployment Guide

```bash
# 1. Local Development (Backend)
python -m uvicorn services.api.main:app --reload --port 8000

# 2. Local Development (Frontend)
cd frontend
bun install
bun run dev  # Starts Bun server on port 3000 with API proxy to 8000

# 3. Docker 1-Command Production Build
docker-compose up --build
```
