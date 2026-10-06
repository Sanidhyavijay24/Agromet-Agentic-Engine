# Agromet Agentic Engine (A²E)

> **Physics-Guided 1km Microclimate Downscaling & Autonomous Agronomic Intelligence**

Agromet Agentic Engine (A²E) is an open-source platform that bridges coarse numerical weather forecasts with farm-scale reality through **1 km² physics-guided residual ML downscaling** and an **autonomous Agentic AI advisory engine** covering all 15 ICAR Agro-Climatic Zones across India.

---

## 🏗️ Architecture Overview

- **1km Microclimate Downscaler:** Physics-constrained residual XGBoost models trained on a decade of hourly meteorological and 30m SRTM DEM terrain data (+58.9% macro error reduction).
- **Agentic AI Decision Engine:** Autonomous multi-step reasoning agent synthesizing microclimate anomalies, vapor pressure deficits, thermal inversion risks, and crop phenology into actionable farm advisories.
- **High-Performance FastAPI Gateway:** Sub-20ms batch inference across 168-hour forecast horizons.

---

## 🚀 Quick Start

### 1. Backend & ML Environment

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run test suite (17 ML tests)
python -m pytest tests/test_zone_router.py tests/test_downscaler.py tests/test_pan_india_serving.py -v

# 3. Start FastAPI Server
uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload
```
