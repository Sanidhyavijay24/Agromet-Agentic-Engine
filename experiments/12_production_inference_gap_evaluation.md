# Experiment 12: Live Production Inference Gap Evaluation

## 1. Executive Summary
- **Date Executed:** September 2026
- **Objective:** Quantify the accuracy gap when the ML residual model runs in live production mode using real-time NWP forecasts (which lack measured ground-truth covariates like soil temperature) compared to offline research benchmarks.
- **Core Question:** When live Open-Meteo 7-day NWP forecasts are ingested without measured soil sensors, does the model maintain significant accuracy improvement over raw NWP data?

---

## 2. Experimental Setup
- **Offline Mode:** Full 28-feature matrix using true historical meteorological and soil parameters.
- **Production Mode (Forecast Pipeline):** 28-feature matrix where unavailable soil covariates are dynamically estimated via atmospheric energy balance proxies (`delta_elevation_m`, `radiation_flux`, `lagged_air_temp`).
- **Evaluation Set:** 70,272 test observations.

---

## 3. Comparative Results

| Operational Mode | Test RMSE | Test MAE | Test R² | Error Drop vs Raw NWP |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Coarse NWP (No Downscaling)** | 0.726 °C | 0.523 °C | 0.9880 | 0.0% |
| **Production Mode (Live Forecast Proxies)** | **0.462 °C** | **0.334 °C** | **0.9962** | **+36.4%** |
| **Offline Research Mode (Full Ground Sensors)** | **0.428 °C** | **0.302 °C** | **0.9972** | **+40.6%** |

---

## 4. Architectural Takeaway
- **Production Gap:** Only a minor 0.034 °C RMSE penalty occurs when using live atmospheric proxy estimations in production.
- **Operational Viability:** Live production downscaling retains a strong **+36.4% error reduction** over raw NWP data, ensuring high-quality real-time advisory generation even without dense ground sensor arrays.
