<div align="center">

# Agromet Agentic Engine (A²E)
### Physics-Guided 1 km² Microclimate Downscaling & Autonomous Agronomic Intelligence

![Agromet Agentic Engine System Architecture](assets/a2e_system_banner.jpg)

**Production-grade microclimate downscaling framework and autonomous ReAct agentic reasoning engine for precision agriculture across India.**

---

</div>

## Executive Summary

The **Agromet Agentic Engine (A²E)** bridges the spatial resolution gap between coarse global Numerical Weather Prediction (NWP) models (25 km – 50 km) and localized field-level microclimates (1 km²). Coarse NWP models systematically average out complex terrain relief, valley drainage cold pools, sand-dune thermal radiation, and stomatal boundary layer dynamics—leading to catastrophic misjudgments in frost mitigation, irrigation scheduling, and pesticide spray timing.

A²E resolves this challenge through a two-stage hybrid architecture:

1. **Physics-Guided 28-Feature Gradient Boosted Downscaler**: Ingests baseline forecast grids, 30m USGS SRTM digital elevation rasters, solar Global Horizontal Irradiance (GHI), and regolith thermal inertia lag to compute spatial residuals (`R = T_local - T_baseline_LOO`), achieving a **+58.9% macro RMSE error reduction** across India's 15 Agro-Climatic Zones (ACZs).
2. **Autonomous ReAct Agromet Agent**: Powered by **Google Gemini Flash** with specialized physical tools (`DownscalerTool`, `AgrometIndicesTool`, `AgronomyTool`, `SoilTerrainTool`), performing multi-step reasoning, psychrometric stress analysis (VPD, GDD, Delta-T), and synthesizing actionable, bilingual (English / Hindi) agronomic directives.

---

## System Architecture

```
[ NWP Baseline Forecast (25km) ] + [ USGS SRTM 30m DEM Rasters ] + [ Satellite Telemetry ]
                               |
                               v
   +-----------------------------------------------------------------------+
   |                     A²E Physics Feature Pipeline                      |
   |  • Horn's Slope & Aspect Azimuth       • Multi-Scale TPI (300m/1000m) |
   |  • Elevation Lapse Differential        • Thermal Regolith Memory      |
   |  • Psychrometric Vapor Pressure Deficit • Boundary Layer Flux          |
   +-----------------------------------------------------------------------+
                               |
                               v
   +-----------------------------------------------------------------------+
   |            28-Covariate Residual Downscaling Engine                   |
   |  • 15 ICAR Agro-Climatic Zone Partitioned Models                      |
   |  • Spatial Coordinate Router & Boundary Box Dispatch                  |
   |  • Zero Target Leakage Invariant: R = T_local - T_baseline_LOO        |
   +-----------------------------------------------------------------------+
                               |
                               v
   +-----------------------------------------------------------------------+
   |               Autonomous Agromet ReAct Agent (Gemini Flash)           |
   |  [DownscalerTool]   [AgrometIndicesTool]  [AgronomyTool] [SoilTool]   |
   |         |                    |                   |            |       |
   |  1km T / RH Grid        VPD / Delta-T      ICAR Crop Limits  DEM/Soil |
   +-----------------------------------------------------------------------+
                               |
                               v
   +-----------------------------------------------------------------------+
   |                  FastAPI Gateway & Pure Bun Web UI                    |
   |  • 1km vs 25km Diurnal Scrubber        • Live ReAct Tool Traces       |
   |  • Spray Window Delta-T Calculator     • Bilingual Action Plan (EN/HI)|
   +-----------------------------------------------------------------------+
```

---

## Key Capabilities

### 1. Physics-Guided 1 km² Downscaling
- **Spatial Block Partitioning**: Strict Leave-One-Panchayat-Out cross-validation ensuring spatial generalization without coordinate leakage.
- **Topographic Correction**: Real-time continuous sampling of USGS SRTM 30m rasters for elevation lapse rate (`Γ = 0.0065 °C/m`), topographic shelter indices, and aspect solar incidence angles.
- **Regolith Thermal Lag**: Incorporates 3-hour lag-convolved surface temperature memory to model diurnal arid hysteresis in desert and black-cotton vertisol regions.

### 2. Psychrometric Agromet Indices Engine
- **Vapor Pressure Deficit (VPD)**: Accurate computation using the Tetens formulation (`e_s = 0.61078 * exp((17.27 * T) / (T + 237.3))`) to detect stomatal closure risks (`> 2.5 kPa`).
- **Delta-T Spray Windows**: Computes wet-bulb depression to identify hazardous volatilization windows (`ΔT > 8°C`) and inversion drift risks (`ΔT < 2°C`).
- **Surface Inversion Trapping**: Detects nocturnal radiation inversions in low-lying valley dunes (`T_local - T_baseline < -0.4°C`) under calm wind conditions (`< 4 km/h`).

### 3. Autonomous ReAct Agent Loop
- **Multi-Tool Tool Calling**: Gemini Flash autonomously orchestrates queries across physical downscaling, agronomic registries, and soil telemetry before synthesizing verdicts.
- **ICAR Crop Knowledge Base**: Embedded physiological constraints for Bajra (Pearl Millet), Guar (Cluster Bean), Groundnut, Wheat, Mustard, and Cotton.
- **Dual Vernacular Synthesis**: Produces technical executive English summaries alongside natural, high-fidelity Hindi advisories for farmers.

---

## Pan-India 15 ACZ Benchmark Performance

Evaluated across **47,342,880 hourly observations** spanning 10 years (2015–2024) across all 15 ICAR Agro-Climatic Zones:

| Zone ID | Agro-Climatic Zone | Baseline RMSE | A²E Downscaled RMSE | A²E Test MAE | Test R² | Error Reduction | Deployment Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ACZ 01** | Western Himalayan Region | 11.797°C | **4.636°C** | 3.117°C | 0.8914 | **+60.7%** | Kaggle Hub Fleet |
| **ACZ 02** | Eastern Himalayan Region | 8.649°C | **1.045°C** | 0.711°C | 0.9867 | **+87.9%** | Kaggle Hub Fleet |
| **ACZ 03** | Lower Gangetic Plain | 0.944°C | **0.485°C** | 0.346°C | 0.9920 | **+48.7%** | Kaggle Hub Fleet |
| **ACZ 04** | Middle Gangetic Plain | 1.051°C | **0.461°C** | 0.334°C | 0.9950 | **+56.2%** | Kaggle Hub Fleet |
| **ACZ 05** | Upper Gangetic Plain | 2.353°C | **1.089°C** | 0.594°C | 0.9799 | **+53.7%** | Kaggle Hub Fleet |
| **ACZ 06** | Trans-Gangetic Plain | 3.064°C | **0.864°C** | 0.562°C | 0.9895 | **+71.8%** | Kaggle Hub Fleet |
| **ACZ 07** | Eastern Plateau & Hills | 1.478°C | **0.637°C** | 0.475°C | 0.9900 | **+56.9%** | Kaggle Hub Fleet |
| **ACZ 08** | Central Plateau & Hills | 1.066°C | **0.640°C** | 0.438°C | 0.9911 | **+40.0%** | Kaggle Hub Fleet |
| **ACZ 09** | Western Plateau & Hills | 2.149°C | **0.798°C** | 0.594°C | 0.9778 | **+62.9%** | Kaggle Hub Fleet |
| **ACZ 10** | Southern Plateau & Hills | 1.241°C | **0.503°C** | 0.381°C | 0.9895 | **+59.5%** | Kaggle Hub Fleet |
| **ACZ 11** | East Coast Plains & Hills | 1.400°C | **0.721°C** | 0.488°C | 0.9726 | **+48.5%** | Kaggle Hub Fleet |
| **ACZ 12** | West Coast Plains & Ghats | 2.606°C | **0.697°C** | 0.491°C | 0.9688 | **+73.2%** | Kaggle Hub Fleet |
| **ACZ 13** | Gujarat Plains & Hills | 1.310°C | **0.668°C** | 0.501°C | 0.9878 | **+49.1%** | Kaggle Hub Fleet |
| **ACZ 14** | **Western Dry (Rajasthan)** | 1.258°C | **0.473°C** | 0.352°C | 0.9921 | **+62.4%** | **Live In-Memory (165MB)** |
| **ACZ 15** | Islands (Andaman & Nicobar) | 2.215°C | **0.984°C** | 0.720°C | 0.9810 | **+55.6%** | Kaggle Hub Fleet |
| **ALL** | **Macro Pan-India Average** | **2.772°C** | **0.953°C** | **0.653°C** | **0.9745** | **+58.9%** | **Verified 10-Yr Benchmark** |

---

## Directory Layout

```
agromet-agentic-engine/
├── services/
│   ├── agent/                      # Autonomous ReAct agent & tool suite
│   │   ├── core.py                 # Multi-step ReAct agent orchestrator
│   │   ├── schemas.py              # Strict Pydantic v2 validation contracts
│   │   └── tools/                  # Downscaler, indices, agronomy, and soil tools
│   ├── ml_downscaler/              # 28-feature residual XGBoost downscaler
│   │   ├── inference.py            # Point & batch downscaling runtime
│   │   ├── dem_source.py           # USGS 30m SRTM raster extractor
│   │   └── zone_router.py          # Spatial bounding-box coordinate router
│   ├── ingestion/                  # Open-Meteo & telemetry ingestion clients
│   └── api/                        # FastAPI gateway
│       ├── main.py                 # Application root & CORS middleware
│       └── routes/                 # /agent, /forecast, and /health endpoints
├── frontend/                       # Pure Bun + React 19 Frontend
│   ├── src/
│   │   ├── index.tsx               # Client entrypoint
│   │   ├── index.css               # Swiss brutalist design tokens
│   │   ├── components/             # Comparator, Diurnal Scrubber, Agent Terminal
│   │   └── services/api.ts         # Live backend client & API proxy
│   ├── server.ts                   # Native Bun.serve HTTP server
│   └── package.json                # Bun build scripts
├── experiments/                    # 16 complete experimental reports & benchmarks
├── data/                           # Processed metadata & DEM rasters
├── tests/                          # 72 passing backend tests (pytest) + bun test
├── Dockerfile                      # Multi-stage container definition (<350MB)
├── docker-compose.yml              # 1-command deployment orchestration
└── context.md                      # Single Source of Truth ledger
```

---

## API Specification

### 1. Autonomous Agromet Consultation
`POST /api/v1/agent/advisory`

```json
{
  "latitude": 26.3571,
  "longitude": 73.0412,
  "crop": "Bajra",
  "crop_stage": "Grain Filling",
  "soil_type": "Sandy Loam",
  "user_query": "Forecast indicates 43.5°C peak ambient heat. Plants showing incipient afternoon flag leaf roll.",
  "forecast_hours": 48
}
```

**Response Contract (`AgentAdvisoryResponse`):**
```json
{
  "status": "success",
  "location": { "latitude": 26.3571, "longitude": 73.0412, "zone_id": 14 },
  "crop": "Bajra",
  "crop_stage": "Grain Filling",
  "indices": {
    "mean_vpd_kpa": 3.42,
    "max_vpd_kpa": 4.18,
    "thermal_inversion_risk": false,
    "frost_risk": false,
    "gdd_accumulated_c_days": 38.4,
    "heat_stress_hours": 6,
    "spray_windows": [
      {
        "start_time": "2026-10-07T06:00:00",
        "end_time": "2026-10-07T09:00:00",
        "duration_hours": 3,
        "suitability_score": 92.0,
        "recommended_action": "Execute scheduled pesticide application before thermal gust boundary"
      }
    ]
  },
  "action_plan": {
    "summary_headline": "Critical Thermal Stress Alert for Bajra (Grain Filling)",
    "verdict_category": "HEAT_STRESS_DEFENSE",
    "irrigation_advice": "Apply light evening sprinkler irrigation to alleviate canopy desiccation. Topsoil moisture index is critical.",
    "spray_recommendation": "Optimal spray window between 06:00 and 09:00 IST. Wind speed < 10 km/h and Delta-T within 4.5°C.",
    "crop_specific_protection": "Monitor flag leaf roll; ensure adequate moisture to avoid premature grain shriveling.",
    "vernacular_hindi": "बाजरा (दाना भराव अवस्था): आगामी 48 घंटों में तापमान 43°C से अधिक रहने की संभावना है...",
    "english_summary": "Heat stress mitigation protocol active for Zone XIV Pearl Millet."
  },
  "agent_trace": [
    {
      "tool_name": "run_1km_downscaler",
      "parameters": { "latitude": 26.3571, "longitude": 73.0412 },
      "result_summary": "Evaluated 48 hourly points, mean delta T: +1.42°C",
      "execution_time_ms": 42.1
    },
    {
      "tool_name": "calculate_microclimate_indices",
      "parameters": { "crop_base_temp_c": 10.0, "crop_max_temp_c": 38.0 },
      "result_summary": "Mean VPD: 3.42 kPa, Heat Stress: 6 hrs, Spray Windows: 1",
      "execution_time_ms": 3.8
    }
  ],
  "llm_model_used": "gemini-flash-lite-latest",
  "total_latency_ms": 842.0
}
```

---

## Getting Started

### Prerequisites
- **Python 3.11+**
- **Bun 1.3+** (`curl -fsSL https://bun.sh/install | bash`)
- **Google Gemini API Key** (Free tier supported)

### 1. Environment Configuration
Create a `.env` file in the project root:
```ini
ENVIRONMENT=development
BACKEND_URL=http://127.0.0.1:8000
PORT=3000
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL_NAME=gemini-flash-lite-latest
```

### 2. Launch Backend (FastAPI)
```powershell
python -m uvicorn services.api.main:app --reload --port 8000
```

### 3. Launch Frontend (Pure Bun)
```powershell
cd frontend
bun install
bun run dev
```
Navigate to `http://localhost:3000` to access the interface.

---

## Production Deployment

### 1-Command Docker Deployment
```powershell
docker-compose up --build
```
The multi-stage container builds in under 60 seconds with a memory footprint under **350 MB**, serving both the FastAPI gateway (Port 8000) and the Bun UI (Port 3000).

---

## Test Verification

Run the full regression test suite:
```powershell
# Run backend pytest suite (72 tests)
python -m pytest tests/ -v

# Run frontend bun test suite (2 tests)
cd frontend
bun test
```

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
