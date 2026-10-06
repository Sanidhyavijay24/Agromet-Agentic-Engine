"""
@file b4_data_quality.py
@description B4: data-quality audit of the training parquet. No model, no retraining.
@module scripts/audit

CHECKS
------
Structure    nulls, duplicate (point, hour) rows, hourly gaps per point, equal coverage
Physics      every weather variable inside a physically possible range
Consistency  each point has one latitude / longitude / elevation
Target       baseline_temp_c really is the leave-one-out mean of the other points, and
             residual_anomaly_c really is temperature minus that baseline -- if either is
             wrong, every metric the project reports is computed against the wrong truth
Features     constant or near-constant columns (a feature with one value teaches nothing)
Source       share of rows from ERA5-Land (pre-2017) vs ECMWF IFS (2017+), since production is IFS
Terrain      whether the DEM needed for real terrain features is present

RUN
---
    python scripts/audit/b4_data_quality.py
    python scripts/audit/b4_data_quality.py --input data/raw/jaipur_10yr_training_data.parquet
Writes data/processed/audit/b4_data_quality.json. Exit code 1 if any check FAILS.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import pandas as pd

from _common import IFS_ERA_START_YEAR, load_dataset, write_json

# Physically possible ranges for the Jaipur-region training data (hourly values).
RANGES: dict[str, tuple[float, float]] = {
    "temperature_2m_c": (-10.0, 52.0),
    "relative_humidity_2m_pct": (0.0, 100.0),
    "surface_pressure_hpa": (850.0, 1050.0),
    "wind_speed_10m_kmh": (0.0, 150.0),
    "precipitation_mm": (0.0, 200.0),
    "solar_radiation_w_m2": (0.0, 1200.0),
    "shortwave_radiation_w_m2": (0.0, 1400.0),
    "soil_temperature_0_to_7cm_c": (-5.0, 75.0),
    "soil_moisture_0_to_7cm_m3m3": (0.0, 0.7),
    "et0_evapotranspiration_mm": (0.0, 2.5),
    "elevation_m": (0.0, 2000.0),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    args = ap.parse_args()

    print("=" * 96)
    print("B4: TRAINING DATA QUALITY")
    print("=" * 96)
    df, source_file = load_dataset(args.input)
    checks: list[dict[str, Any]] = []

    def record(name: str, status: str, detail: str, **extra: Any) -> None:
        checks.append({"check": name, "status": status, "detail": detail, **extra})
        print(f"  [{status:<4}] {name:<38} {detail}")

    print()
    # ── structure ────────────────────────────────────────────────────────
    nulls = df.isna().sum()
    bad = nulls[nulls > 0]
    record("nulls", "PASS" if bad.empty else "FAIL",
           "none" if bad.empty else f"{int(bad.sum()):,} nulls in {len(bad)} column(s): {dict(bad.head(6))}")

    dup = int(df.duplicated(subset=["point_id", "timestamp"]).sum())
    record("duplicate point-hours", "PASS" if dup == 0 else "FAIL", f"{dup:,} duplicate (point, timestamp) rows")

    ts = pd.to_datetime(df["timestamp"], utc=True)
    span_hours = int((ts.max() - ts.min()).total_seconds() // 3600) + 1
    per_point = df.groupby("point_id").size()
    gaps = {pid: span_hours - int(n) for pid, n in per_point.items() if n != span_hours}
    record("hourly coverage", "PASS" if not gaps else "WARN",
           f"{ts.min():%Y-%m-%d} .. {ts.max():%Y-%m-%d}, {span_hours:,} h expected per point; "
           + ("all complete" if not gaps else f"{len(gaps)} point(s) short, worst missing {max(gaps.values()):,} h"))

    # ── physics ──────────────────────────────────────────────────────────
    out_of_range = {}
    for col, (lo, hi) in RANGES.items():
        if col in df.columns:
            n = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n:
                out_of_range[col] = {"rows": n, "min": float(df[col].min()), "max": float(df[col].max()),
                                     "allowed": [lo, hi]}
    record("physical ranges", "PASS" if not out_of_range else "FAIL",
           "all variables within physical limits" if not out_of_range
           else f"{len(out_of_range)} variable(s) out of range: {list(out_of_range)}",
           out_of_range=out_of_range)

    # ── per-point consistency ────────────────────────────────────────────
    inconsistent = [c for c in ("latitude", "longitude", "elevation_m")
                    if c in df.columns and (df.groupby("point_id")[c].nunique() > 1).any()]
    record("static attributes per point", "PASS" if not inconsistent else "FAIL",
           "lat / lon / elevation constant per point" if not inconsistent
           else f"varies within a point: {inconsistent}")

    # ── target construction ──────────────────────────────────────────────
    grp = df.groupby("timestamp")["temperature_2m_c"]
    total, count = grp.transform("sum"), grp.transform("count")
    loo = np.where(count > 1, (total - df["temperature_2m_c"]) / (count - 1), df["temperature_2m_c"])
    loo_err = float(np.max(np.abs(loo - df["baseline_temp_c"])))
    diagnosis = ""
    if loo_err >= 1e-3 and "residual_anomaly_c" in df.columns:
        # Most likely cause: the file predates the leave-one-out fix and its baseline is the
        # plain regional mean INCLUDING the point itself. Then, exactly,
        #   stored - LOO = (T - full_mean) / (N - 1) = residual / (N - 1).
        full_mean_signature = (df["baseline_temp_c"] - loo) - df["residual_anomaly_c"] / (count - 1)
        if float(np.max(np.abs(full_mean_signature))) < 1e-6:
            diagnosis = ("; diagnosis: the baseline is the FULL regional mean including the point "
                         "itself -- this file predates the leave-one-out fix, so its target differs "
                         "from the 10-year training set's")
        else:
            diagnosis = "; not explained by a self-inclusive mean -- inspect how this file was built"
    record("baseline is the leave-one-out mean", "PASS" if loo_err < 1e-3 else "FAIL",
           f"max |recomputed - stored| = {loo_err:.2e} C{diagnosis}", max_abs_error=loo_err)

    if "residual_anomaly_c" in df.columns:
        res_err = float(np.max(np.abs(df["temperature_2m_c"] - df["baseline_temp_c"] - df["residual_anomaly_c"])))
        record("residual = temperature - baseline", "PASS" if res_err < 1e-3 else "FAIL",
               f"max error {res_err:.2e} C; target mean {df['residual_anomaly_c'].mean():+.4f} C, "
               f"sd {df['residual_anomaly_c'].std():.3f} C", max_abs_error=res_err)

    # ── degenerate features ──────────────────────────────────────────────
    from services.ml_downscaler.train import FEATURES_LIST
    degenerate = {}
    for f in FEATURES_LIST:
        if f in df.columns:
            per_pt = df.groupby("point_id")[f].first() if f in (
                "topographic_position_index", "slope_magnitude_deg", "aspect_sin", "aspect_cos") else None
            distinct = int(df[f].round(6).nunique())
            if distinct <= 2 or float(df[f].std()) < 1e-9:
                degenerate[f] = distinct
            if per_pt is not None and per_pt.round(6).nunique() <= 2:
                degenerate[f"{f} (per point)"] = int(per_pt.round(6).nunique())
    record("no constant features", "PASS" if not degenerate else "WARN",
           "every feature varies" if not degenerate else f"near-constant: {degenerate}")

    slope = df.groupby("point_id")["slope_magnitude_deg"].first()
    record("terrain slope range", "INFO",
           f"training slopes {slope.min():.3f}-{slope.max():.3f} deg across {slope.size} points "
           f"(grid-neighbour method); inference feeds 0.5-19.6 deg from a formula (F17)")

    # ── source era ───────────────────────────────────────────────────────
    years = ts.dt.year
    era5 = int((years < IFS_ERA_START_YEAR).sum())
    share = era5 / len(df)
    record("source era", "WARN" if era5 else "PASS",
           f"{share:.0%} of rows ERA5-Land (before {IFS_ERA_START_YEAR}), {1 - share:.0%} ECMWF IFS; "
           f"production is ECMWF IFS" + (" -- mixed-source training set (F21)" if era5 else ""),
           era5_land_rows=era5, ifs_rows=int(len(df) - era5))

    # ── DEM availability ─────────────────────────────────────────────────
    try:
        from services.ml_downscaler.dem_source import dem_availability
        dem = dem_availability()
        record("DEM tiles for real terrain", "PASS" if dem["available"] else "WARN",
               f"{dem['tile_count']} tile(s) in data/raw/dem" if dem["available"]
               else "none: terrain features fall back to the analytical formula (F1)")
    except Exception as exc:  # pragma: no cover
        record("DEM tiles for real terrain", "WARN", f"could not check: {exc}")

    failed = [c for c in checks if c["status"] == "FAIL"]
    warned = [c for c in checks if c["status"] == "WARN"]
    print()
    print(f"  {len(checks)} checks: {len(failed)} FAIL, {len(warned)} WARN, "
          f"{sum(c['status'] == 'PASS' for c in checks)} PASS")
    path = write_json("b4_data_quality.json", {"source_file": source_file, "rows": int(len(df)),
                                               "points": int(df["point_id"].nunique()), "checks": checks})
    print(f"  wrote {path}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
