# Experiment 09: 10-Year Multi-Decadal Horizon Scaling Benchmark

## 1. Hypothesis & Context
- **Objective:** Evaluate how the 28-feature physics-guided GBDT champion scales when expanding training data from 1 single year (2024, 316,224 rows) to **10 full calendar years (2015-2024, 3,156,192 rows)** across the Jaipur Rural / Chaksu domain.
- **Hypothesis:** Multi-decadal training exposes the model to historical climate extremes (heatwaves, Western Disturbances, monsoonal variations), deepening tree capacity and improving generalization on unseen spatial points.

---

## 2. Dataset Horizon & Resource Allocation
- **Temporal Horizon:** January 1, 2015 00:00 UTC to December 31, 2024 23:00 UTC (10 complete years).
- **Total Records:** 3,156,192 hourly rows across 36 spatial grid points.
- **Training Configuration:**
  - 28 Spatial Training Points (2,454,816 rows).
  - 8 Spatial Validation Points (701,376 rows).
  - CUDA GPU Histogram tree method (`device=cuda`).
  - Trees: 1,199 converged estimators (`learning_rate=0.03`, `max_depth=8`).

---

## 3. Experimental Results

| Metric | 1-Year Baseline Model (2024) | 10-Year Scaled Champion (2015-2024) |
| :--- | :---: | :---: |
| **Training Records** | 245,952 | **2,454,816** |
| **Validation Records** | 70,272 | **701,376** |
| **Validation R²** | 0.9968 | **0.9978** |
| **Training Convergence Time** | 14.2s | **124.6s (CUDA GPU)** |
| **Peak VRAM Usage** | 0.6 GB | **1.8 GB** |

---

## 4. Summary & Verification
- Scaling from 1 year to 10 years increased sample density by 10x while maintaining rapid GPU convergence (< 2.5 minutes).
- The 10-year model successfully captured historical extreme weather transitions, forming the backbone for production deployment.
