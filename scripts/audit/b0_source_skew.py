"""
@file b0_source_skew.py
@description Identify which weather model produced the TRAINING data (year by year) and which
             produces the PRODUCTION forecast, and measure how far apart the sources are.
@module scripts/audit

WHY
---
Every feature the downscaler sees comes from Open-Meteo. The training data was fetched from
the archive API with no `models=` parameter, and production calls the forecast API the same
way, so each side got whatever model Open-Meteo served by default. Whether those are the same
model decides whether "train/serve skew" is a data-provider problem at all.

METHOD
------
Open-Meteo lets you name a model explicitly. If the default series is identical (mean
absolute difference 0.000) to a named model's series, that named model IS the default.
Checked per year for the archive, and for the next days of the forecast.

WHAT DOES NOT WORK, AND WHY IT IS NOT USED
------------------------------------------
Pairing the forecast API's `past_days` with the archive for the same hours looks like a clean
source comparison, but for at least the last ~60 days Open-Meteo serves IDENTICAL values from
both (tested at 2, 8, 15, 30 and 50 days back: 100% identical on all 12 features). Recent
archive data is the same IFS data the forecast API holds, so that pairing measures nothing.

FINDING (2026-09-27, Kadera)
----------------------------
  archive default 2015-2016 == ERA5-Land      (IFS archive has no data before 2017)
  archive default 2017-2025 == ECMWF IFS HRES
  forecast default         == ECMWF IFS HRES
So ~80% of a 2015-2024 training set shares production's source; the 2015-2016 fifth does
not, and differs from IFS by roughly 0.6-1.6 C in the same hours.

RUN
---
    python scripts/audit/b0_source_skew.py
Network only. Writes data/processed/audit/b0_source_identity.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT_DIR / "data" / "processed" / "audit"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"

VARS = ["temperature_2m", "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm", "relative_humidity_2m"]
ARCHIVE_CANDIDATES = ["ecmwf_ifs", "era5_land", "era5", "era5_seamless", "ecmwf_ifs_analysis_long_window"]
FORECAST_CANDIDATES = ["ecmwf_ifs", "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "jma_seamless",
                       "meteofrance_seamless", "ukmo_seamless", "cma_grapes_global"]


def series(url: str, lat: float, lon: float, **params: Any) -> dict[str, np.ndarray] | None:
    p = {"latitude": lat, "longitude": lon, "hourly": ",".join(VARS), "timezone": "UTC", **params}
    r = requests.get(url, params=p, timeout=60)
    if not r.ok:
        return None
    h = r.json().get("hourly", {})
    return {v: np.array([np.nan if x is None else x for x in h.get(v, [])], float) for v in VARS}


def mad(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    m = np.isfinite(a[:n]) & np.isfinite(b[:n])
    return float(np.mean(np.abs(a[:n][m] - b[:n][m]))) if m.any() else None


def identify(default: dict[str, np.ndarray] | None, named: dict[str, dict[str, np.ndarray] | None]) -> tuple[str, dict]:
    """The named model whose temperature series is identical to the default, if any."""
    diffs = {}
    for model, s in named.items():
        d = mad(default["temperature_2m"] if default else None, s["temperature_2m"] if s else None)
        diffs[model] = d
    exact = [m for m, d in diffs.items() if d is not None and d < 1e-6]
    return (exact[0] if exact else "unidentified"), diffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=26.9124)
    ap.add_argument("--lon", type=float, default=75.7873)
    ap.add_argument("--years", type=int, nargs="*", default=list(range(2015, 2026)))
    args = ap.parse_args()

    print("=" * 96)
    print(f"B0: WHICH MODEL PRODUCED THE DATA?   point {args.lat}, {args.lon}")
    print("=" * 96)
    result: dict[str, Any] = {"point": [args.lat, args.lon], "archive_by_year": {}, "forecast": {}}

    print(f"\n  TRAINING SOURCE (archive default), first days of June each year")
    print(f"  {'year':<8}{'identified as':<20}  mean |T diff| vs each candidate")
    for y in args.years:
        s, e = f"{y}-06-01", f"{y}-06-03"
        default = series(ARCHIVE, args.lat, args.lon, start_date=s, end_date=e)
        named = {m: series(ARCHIVE, args.lat, args.lon, start_date=s, end_date=e, models=m)
                 for m in ARCHIVE_CANDIDATES}
        who, diffs = identify(default, named)
        result["archive_by_year"][y] = {"model": who, "diffs": diffs}
        shown = "  ".join(f"{m}={'-' if d is None else f'{d:.2f}'}" for m, d in diffs.items())
        print(f"  {y:<8}{who:<20}  {shown}")

    print(f"\n  PRODUCTION SOURCE (forecast default), next 3 days")
    default = series(FORECAST, args.lat, args.lon, forecast_days=3)
    named = {m: series(FORECAST, args.lat, args.lon, forecast_days=3, models=m) for m in FORECAST_CANDIDATES}
    who, diffs = identify(default, named)
    result["forecast"] = {"model": who, "diffs": diffs}
    print(f"  identified as: {who}")
    for m, d in diffs.items():
        print(f"    {m:<24} mean |T diff| {'-' if d is None else f'{d:.3f}'}")

    # How far apart are the two training-era sources, in the same hours?
    print(f"\n  SAME HOURS, TWO SOURCES: ERA5-Land vs ECMWF IFS (2024-06-01..07)")
    land = series(ARCHIVE, args.lat, args.lon, start_date="2024-06-01", end_date="2024-06-07", models="era5_land")
    ifs = series(ARCHIVE, args.lat, args.lon, start_date="2024-06-01", end_date="2024-06-07", models="ecmwf_ifs")
    gap = {}
    for v in VARS:
        d = mad(land[v] if land else None, ifs[v] if ifs else None)
        gap[v] = d
        print(f"    {v:<30} mean |diff| {'-' if d is None else f'{d:.3f}'}")
    result["era5_land_vs_ifs_same_hours"] = gap

    by_model: dict[str, list[int]] = {}
    for y, r in result["archive_by_year"].items():
        by_model.setdefault(r["model"], []).append(int(y))
    print("\n  " + "-" * 92)
    for m, ys in by_model.items():
        print(f"  training years served by {m}: {min(ys)}-{max(ys)} ({len(ys)} of {len(args.years)})")
    prod = result["forecast"]["model"]
    same = [y for y, r in result["archive_by_year"].items() if r["model"] == prod]
    print(f"  production is served by {prod}; training years from the same source: "
          f"{len(same)} of {len(args.years)}")
    if len(same) < len(args.years):
        print("  RECOMMENDATION: pin models=ecmwf_ifs in ingestion and train on 2017+ only, so every")
        print("  training row comes from the model production is served by.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "b0_source_identity.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {out.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
