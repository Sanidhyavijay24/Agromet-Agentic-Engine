# Machine Learning Downscaler Experiments Leaderboard & Comprehensive Analysis

> **Experiment Tracking System for Agromet Agentic Engine (A²E)**  
> Benchmarks performance, tracks metrics, and establishes production downscaling champions across all 15 Agro-Climatic Zones of India.

---

## 1. Chronological Experiment Evolution (Zone XIV - Rajasthan Benchmark)

| Exp ID | Experiment Title | Model Architecture | Features | Test RMSE | Test MAE | Test R² | Error Drop (%) | Key Innovation / Outcome |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **EXP-01** | Initial XGBoost Baseline | XGBoost (300 trees, d=6) | 8 | 0.543 °C | 0.391 °C | 0.9934 | +25.1% | Proved residual learning formulation (`R = T_local - T_base`). |
| **EXP-02** | Deepened XGBoost + Physics | XGBoost (1200 trees, d=8) | 21 | 0.439 °C | 0.311 °C | 0.9968 | +39.5% | Introduced VPD, Nocturnal Inversion, and Solar Heating interactions. |
| **EXP-03** | 5-Fold Spatial Cross-Validation | XGBoost (1000 trees, d=8) | 21 | 0.448 °C | 0.319 °C | 0.9971 | +38.0% | Proved spatial stability across 5 distinct geographic partitions. |
| **EXP-04** | Analytical Lapse Rate Baseline | Linear Lapse (-6.5 °C/km) | 1 | 0.648 °C | 0.457 °C | 0.9939 | +6.9% | Quantified limitation of pure 1D atmospheric physics. |
| **EXP-05** | Random Forest Regressor | Random Forest (300 trees) | 21 | 0.482 °C | 0.344 °C | 0.9966 | +30.7% | Benchmarked bagging vs boosting under spatial holdout. |
| **EXP-06** | LightGBM Regressor | LightGBM (1200 trees) | 21 | 0.444 °C | 0.315 °C | 0.9967 | +36.2% | Leaf-wise GBDT benchmark. |
| **EXP-07** | CatBoost Regressor | CatBoost (1200 trees) | 21 | 0.449 °C | 0.320 °C | 0.9967 | +35.5% | Oblivious symmetric tree benchmark. |
| **EXP-08** | Advanced Topographic Physics | XGBoost (1200 trees, d=8) | 28 | 0.428 °C | 0.302 °C | 0.9972 | +40.6% | Injected TPI, slope magnitude, aspect, and soil thermal flux. |
| **EXP-09** | 10-Year Horizon Scaling | XGBoost (1199 trees, d=8) | 28 | 0.425 °C | 0.298 °C | 0.9978 | +41.0% | Scaled training to 10 full years (2015-2024, 3.15M rows). |
| **EXP-10** | Leak-Free Pipeline Audit | XGBoost (1200 trees, d=8) | 28 | 0.430 °C | 0.305 °C | 0.9975 | +40.3% | Eliminated temporal lookahead bias and train-serve lag. |
| **EXP-11** | Unseen 2025 Holdout Validation | Champion Model (Exp 09) | 28 | 0.421 °C | 0.299 °C | 0.9972 | +39.6% | Verified zero temporal degradation on unseen 2025 data. |
| **EXP-12** | Production Inference Gap Audit | Champion Model (Exp 09) | 28 | 0.462 °C | 0.334 °C | 0.9962 | +36.4% | Quantified live NWP forecast proxy behavior vs offline truth. |
| **EXP-13** | Quality & Metrics Audit | Methodological Review | - | - | - | - | - | Established strict baseline definitions and zero-leakage standards. |
| **EXP-14** | Converged 30m SRTM Retraining | XGBoost (5555 trees, d=8) | 28 | **0.446 °C** | **0.319 °C** | **0.9967** | **+40.2%** | **Flagship Zone XIV Champion:** 30m SRTM DEM, 5,555 trees. |
| **EXP-15** | Pan-India 15 ACZ Fleet | 15x XGBoost Models | 28 | **0.953 °C** | **0.653 °C** | **0.9745** | **+58.9%** | **Pan-India Fleet:** 47.3M rows, 15 agro-climatic zones. |

---

## 2. Key Takeaways & Architecture Principles
1. **Zone XIV (Rajasthan) Precision:** Verified **+40.2% RMSE reduction** ($0.746^\circ	ext{C} 	o 0.446^\circ	ext{C}$, MAE $0.319^\circ	ext{C}$, $R^2 = 0.9967$) on authentic 30m USGS SRTM DEM elevation rasters.
2. **Pan-India Macro Performance:** Fleet macro average of **+58.9% RMSE reduction** ($2.772^\circ	ext{C} 	o 0.953^\circ	ext{C}$), with high-relief mountain zones achieving up to +87.9% error drop.
3. **Sub-Millisecond Serving:** Pre-warmed in-memory model caches deliver 1km² microclimate predictions in `<1ms`, powering the ReAct autonomous agromet advisory engine.
