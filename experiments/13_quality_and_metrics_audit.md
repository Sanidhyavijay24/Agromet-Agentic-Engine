# Experiment 13: Quality and Metrics Audit of the Downscaler

> **Audit date:** 2026-09-27  **Auditor:** Teammate 3
> **Scope:** training data, feature pipeline, train/serve parity, evaluation method, and the
> reported metrics of the production champion (`residual_model.joblib`, Exp 10).
> **Data available:** the 1-year parquet (2024). The 10-year parquet was not in this checkout;
> results that depend on it are marked **needs 10-yr**.
> **Detailed working log:** `docs/audit_log.md`. **Handover for Lead ML:** `docs/preflight_checklist_16_zone_fleet.md`.

---

## 1. Summary

**The headline +60.6% is inflated by target leakage and cannot be quoted as it stands.**
The 10-year ingest computed one input feature (VPD) from the very temperature the model is
asked to predict, so the model could read the answer off its inputs. The leak is fixed in the
ingestion code and a patcher repairs the existing parquet, but **the shipped model must be
retrained** before any headline number is reported.

**The underlying method is sound, and the honest skill is substantial.** A model retrained on
leak-free features scores **+41.2% ± 3.0%** RMSE gain across five random spatial splits, rising
to **+47.7%** once it is trained to convergence. It beats every no-ML baseline decisively. Those
are 1-year figures; the 10-year retrain should be measured, not assumed.

**Two more problems decide what reaches farmers:** production feeds the model a different
baseline than it was trained on (F5, fix agreed), and its terrain inputs are fabricated by a
formula — every production slope lies outside anything seen in training (F17). The ablation
shows the terrain features can simply be dropped at no cost until real DEM terrain exists.

### What to say about accuracy today

| Claim | Status | Say instead |
|---|---|---|
| "+60.6% RMSE reduction" (Exp 10) | **Inflated by leakage (F22); not reproducible from this checkout** | Withdraw until retrained |
| "R² = 0.9986" | Temperature R², dominated by the diurnal and seasonal cycle the baseline already carries | Report **residual R²** — the share of the local deviation explained |
| "+8.9% on the live pipeline" (Exp 12) | Scored against a lapse-rate formula, not data; its conclusions contradict its own table | Withdraw |
| Honest offline skill, leak-free features | **+41.2% ± 3.0%** (5 splits, 1-year), **+47.7%** converged | Quote as a range, with the split variance |

---

## 2. Findings, most severe first

Status: **FIXED** in code · **ACTION** needed by Lead ML · **OPEN** needs data or a decision.

### Critical

| # | Finding | Evidence | Status |
|---|---|---|---|
| **F22** | **Target leakage through VPD.** `ingest_multi_year_dataset.py:162` computed `vapor_pressure_deficit_kpa` from `temperature_2m_c` — the point's own temperature. With humidity and the baseline also given, the model could invert the VPD formula to recover `T_local − baseline`. `latent_cooling_potential` inherited it. Every other pipeline, including production, computes VPD from the baseline | Shuffling VPD raises test RMSE by **8.1 °C** (≈ the annual temperature spread). On leak-free features the champion scores **+31.7% on its own training points** and +21.0% on test, while a fresh model on identical inputs scores +41.7% | **FIXED** in ingest; **ACTION** patch parquet + retrain |
| **F5** | **Production baseline ≠ training baseline.** Training: regional leave-one-out mean. Production: the panchayat's own point forecast. The residual lands on the wrong reference | Live: correct baseline +26–31%; production mode displaces forecasts by 0.34 °C | **ACTION** anchor-mean baseline (agreed) |
| **F17** | **Every production slope is outside the training range.** Training saw 0.026–0.418°; production feeds 0.5–19.6° from a formula, so every prediction takes the "steepest slope seen" branch. Aspect is a constant in production (F1) | Season-matched parity: out-of-range **100%**; `aspect_cos` 1 distinct value vs 36 | **OPEN** DEM files; **ACTION** see recommendation 3 |

### High

| # | Finding | Evidence | Status |
|---|---|---|---|
| **F21** | **Training data spliced from two models.** Archive default is ERA5-Land for 2015–2016 and ECMWF IFS from 2017; production is IFS. In the same hours they differ by 1.16 °C (air), 1.43 °C (soil), 3.8 points (RH) | Year-by-year identity against named models (differences exactly 0.000) | **ACTION** pin `models=ecmwf_ifs`, start 2017; **needs 10-yr** for the `source_era` experiment |
| **F8** | **The model was under-trained.** It stopped at the 1200-tree cap | Learning curve: converges at **3614** trees; test gain **+41.8% → +47.7%** (+5.9 points) | **ACTION** raise the cap (confirm on 10-yr) |
| **A1** | **The test set contains no valley points.** The 7 test points span 360–475 m; every valley point (down to 278 m) is in train or validation. Valleys are where nocturnal inversions and frost are strongest | Elevation-band breakdown has no valley band | **ACTION** stratify the split by elevation |
| **A2** | **Spatial autocorrelation lifts the score by ~10 points.** Gain falls from +43.9% to +35.4% (0.15° buffer) and +33.9% (0.21°) when a point's grid neighbours are excluded from training | Buffered leave-one-point-out, 8 points | Informs claims — see §4 |
| **F6/F14** | **Elevation source conflict.** Registry (Bhoonidhi) vs training (Open-Meteo): 10 of 71 sites differ by > 50 m (Bhankri −118 m, Chaksu +107 m) | Terrain comparison script | **OPEN** elevation decision |
| **F15** | Formula slopes saturate at 19.6° for any site far from 380 m | Across 71 sites: mean 13.3°, max 19.6° | **OPEN** DEM files |
| **F3** | Training aspect measured counter-clockwise from east, fed as azimuth from north | Code review | **OPEN** with terrain fix |

### Medium

| # | Finding | Evidence | Status |
|---|---|---|---|
| **F9** | Headline R² is the wrong metric | Champion on 1-yr: temperature R² 0.9955 vs residual R² 0.37. A regression test shows a model that predicts **no local deviation at all** still scores temperature R² > 0.98 | **ACTION** report residual R² |
| **A3** | **A single split is ±3 points of noise.** Five spatial seeds: +37.0% to +46.0% | Split-variance experiment | Report ranges; the Exp-log distinctions such as 60.5 vs 60.6 are not meaningful |
| **A4** | **Gain-based importance misleads.** `elevation_m` has 17.7% gain importance but ~0 permutation importance; removing the whole elevation group costs only 0.0016 °C | Permutation + ablation | Stop citing gain importance |
| **A5** | **The model is weakest where advisories matter most** — at night (+17% vs +26% day), in pre-monsoon (+13%) and winter (+17%), versus the SW monsoon (+36%) | Breakdown of the champion on leak-free features | **needs 10-yr retrain** to confirm the pattern survives |
| **A6** | The 1-year file's baseline is not leave-one-out (max 0.27 °C off) — it appears to predate the LOO fix, so its target differs from the 10-year set's | B4, with automatic diagnosis now built in | Re-run B4 to confirm |
| **F4** | DNI/GHI swap between train and serve | Parity PSI 0.043 after fix | **FIXED** |
| **F7** | `terrain.py` hardcodes 380 m; artifact says 373.86 m | Code review | **OPEN** |

### Resolved

| # | Finding | Resolution |
|---|---|---|
| **F20** | Soil moisture and soil–air gradient shift between training and live | Both sides are ECMWF IFS (2024 file vs 2026 live), so it is weather or an IFS version change, **not** a data-source mismatch |

---

## 3. What holds up

- **The ML adds real value beyond simple physics.** Scored on the same test points (champion,
  leak-free features, 1-year):

  | Method | RMSE gain |
  |---|---|
  | Standard lapse rate | **−1.8%** (worse than doing nothing) |
  | Elevation-linear fit | +1.9% |
  | Hour × month × elevation lookup (no ML) | +2.7% |
  | Champion | **+21.0%** |
  | Retrained on honest features | **+41.7%** |

  A no-ML lookup achieves only **13%** of the champion's gain, so the model is learning more than
  "high ground is cooler".
- **The training data is clean** on the 1-year file: no nulls, no duplicate hours, complete
  hourly coverage, every variable within physical limits, constant attributes per point, and a
  correctly computed residual target.
- **Production uses the same model family as 80% of training** (ECMWF IFS), so once F21 is
  handled the provider itself is not a source of skew.
- **Soil features carry genuine signal:** removing them costs +0.050 °C RMSE — the largest
  ablation effect.

---

## 4. How far the numbers generalise

| Setting | Relevant score | Why |
|---|---|---|
| Panchayats **inside** the 26.5–27.2 °N × 75.5–76.2 °E box | unbuffered (**~+44%** on 1-yr) | Every in-box site is within ~0.1° of a training grid point, like the unbuffered test |
| **New zones** or sites outside the box | buffered (**~+34%** on 1-yr) | No nearby training point; spatial autocorrelation no longer helps |
| Registry sites in other states (Kangra, Leh, Nashik…) | **not evaluated** | Outside the training domain; elevation up to 3500 m vs 278–599 m in training |

---

## 5. Recommendations for Lead ML, in order

1. **Patch the parquet and retrain** (F22): `python scripts/audit/fix_vpd_leak_in_parquet.py --input <10-yr parquet>`.
   Report the retrained model's residual R² and gain as a range over several seeds.
2. **Fix the production baseline** (F5): the anchor-mean baseline with per-zone anchors stored
   in each artifact.
3. **Drop the terrain features until real DEM terrain exists.** The ablation shows removing them
   *improves* test RMSE slightly (0.406 → 0.394 °C) and it eliminates F1, F3, F15 and F17 in one
   step. Re-add them only when training and inference compute them from the same DEM.
4. **Pin `models=ecmwf_ifs` and start at 2017** in all ingestion (F21).
5. **Train to convergence:** cap ≥ 4000 trees, patience 200 (F8).
6. **Stratify the spatial split by elevation** so valleys appear in the test set (A1).
7. **Report honestly:** residual R², split variance, and the in-box vs new-zone distinction.
8. **Withdraw or correct** the Exp 10 headline and the Exp 12 write-up.

Steps 1, 3, 4 and 5 all change the feature pipeline or the data, so they must land **before the
16-zone fleet run** — otherwise every one of them costs 16 retrains.

---

## 6. Still open — needs the 10-year parquet

| Item | Script |
|---|---|
| Reproduce Exp 10 exactly (with the leak), then measure the drop after the VPD fix | `b1_reproduce_metrics.py` |
| Does dropping ERA5-Land years help on IFS-era data? | `b2_training_experiments.py --only source_era` |
| Weather vs source for every shifted feature | `p0_parity_check.py` (interannual test) |
| Confirm learning curve, ablation and seasonal weakness at 10-year scale | `b2_training_experiments.py`, `b2_eval_breakdown.py` |

---

## 7. Method notes and corrections made during the audit

Recorded so the conclusions above can be trusted for the right reasons.

- **First parity run was confounded by season.** A full training year was compared with a
  48-hour window, producing 7 false MAJOR shifts. Fixed by season-matching (±14 days); raw PSI is
  still printed so the confound stays visible.
- **Rainfall PSI returned inf** on identical climates (90% zeros collapse the bins). Replaced by a
  dry-share plus wet-distribution comparison with noise-aware thresholds: false-alarm rate 1.5%,
  detection 100% in 200 synthetic trials.
- **A same-dates source test was abandoned.** Pairing the forecast API's `past_days` with the
  archive returned 100% identical values even 50 days back — Open-Meteo serves the same stored
  data from both — so it measured nothing. Replaced by direct model identification.
- **The learning-curve gain was under-estimated beforehand.** The expected gain from more trees was
  stated as 1–3%; the measured gain is 5.9 points.
- **An early elevation figure was wrong.** Tunga was reported as 390 m (Δ +52 m) from an unverified
  coordinate; the registry coordinate gives 354 m (Δ +88 m).
- **Audit-script bugs caught by synthetic testing before any real result was reported:**
  permutation importance corrupted the target when shuffling `baseline_temp_c`; buffered CV
  starved the model of training points; per-point residual R² went strongly negative on points the
  model improved. All fixed and pinned by regression tests (`tests/test_audit_helpers.py`).
- **Leakage test validated against the bug.** `tests/test_no_target_leakage.py` flags exactly
  `vapor_pressure_deficit_kpa` and `latent_cooling_potential` on the original ingest code, and
  nothing on the fixed code.
