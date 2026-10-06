"""
@file p0_patch_parquet_terrain.py
@description PHASE 0: recompute the 4 terrain columns in a training parquet from a real DEM,
             without re-ingesting any weather data.
@module scripts/audit

WHY
---
The terrain columns in the training parquet were computed from gradients between grid
points about 30 km apart (scripts/ingest_multi_year_dataset.py:202-232), not from a DEM.
At inference a different method is used again. Both are wrong, and they are wrong in
different directions, which is the train/serve skew.

The weather columns are fine. Only these 4 columns change:
    topographic_position_index, slope_magnitude_deg, aspect_sin, aspect_cos
plus sloped_solar_insolation, which is derived from slope and aspect and so must be
recomputed to stay consistent.

This writes a NEW file and never overwrites the input. After running it, the model has
to be retrained on the patched file for the change to take effect.

RUN
---
    # inspect what would change, write nothing
    python scripts/audit/p0_patch_parquet_terrain.py --dry-run

    # write the patched copy
    python scripts/audit/p0_patch_parquet_terrain.py \
        --input data/raw/jaipur_10yr_training_data.parquet \
        --output data/raw/jaipur_10yr_training_data_cartodem.parquet

Requires: pyarrow (or fastparquet), and rasterio for GeoTIFF DEM tiles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

TERRAIN_COLUMNS = [
    "topographic_position_index",
    "slope_magnitude_deg",
    "aspect_sin",
    "aspect_cos",
]

# Which TPI scale to write into topographic_position_index.
# 2000 m matches the "position within the wider valley system" idea that the original
# 0.28 deg neighbour method was reaching for, at a physically defined scale.
TPI_COLUMN_SOURCE = "tpi_2000m"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/jaipur_10yr_training_data.parquet")
    ap.add_argument("--output", default=None,
                    help="Defaults to <input stem>_cartodem.parquet")
    ap.add_argument("--dem-dir", default=None)
    ap.add_argument("--tpi-scale", default=TPI_COLUMN_SOURCE, choices=["tpi_300m", "tpi_2000m"])
    ap.add_argument("--dry-run", action="store_true", help="Report changes, write nothing")
    args = ap.parse_args()

    from services.ml_downscaler.dem_source import dem_availability, extract_terrain

    in_path = ROOT_DIR / args.input if not Path(args.input).is_absolute() else Path(args.input)
    if not in_path.exists():
        # fall back to the 1-year file so the script is still useful on a partial checkout
        alt = ROOT_DIR / "data" / "raw" / "jaipur_1yr_training_data.parquet"
        if alt.exists():
            print(f"! {in_path.name} not found; using {alt.name} instead")
            print("  (the production champion was trained on the 10-year file --")
            print("   get it from Lead ML before treating these numbers as final)")
            in_path = alt
        else:
            raise SystemExit(f"No training parquet found at {in_path}")

    out_path = (
        Path(args.output) if args.output
        else in_path.with_name(in_path.stem + "_cartodem" + in_path.suffix)
    )
    if not out_path.is_absolute():
        out_path = ROOT_DIR / out_path
    if out_path.resolve() == in_path.resolve():
        raise SystemExit(f"--output must differ from --input ({in_path}); this script never overwrites its input.")

    print("=" * 78)
    print("PHASE 0: PATCH TERRAIN COLUMNS FROM DEM")
    print("=" * 78)

    avail = dem_availability(args.dem_dir)
    if not avail["available"]:
        raise SystemExit(
            f"No DEM tiles in {avail['directory']}.\n"
            "Put the Bhoonidhi CartoDEM GeoTIFFs there (or pass --dem-dir) and rerun.\n"
            "Without a DEM this script would just write another set of invented numbers."
        )
    print(f"  DEM      : {avail['tile_count']} tile(s), {avail['tiles'][0]['kind']}")
    cov = avail["coverage"]
    print(f"  coverage : {cov['south']:.3f}-{cov['north']:.3f} N, {cov['west']:.3f}-{cov['east']:.3f} E")

    print(f"  reading  : {in_path.relative_to(ROOT_DIR)}")
    df = pd.read_parquet(in_path)
    print(f"  rows     : {len(df):,}   columns: {len(df.columns)}")

    required = ["point_id", "latitude", "longitude", "elevation_m"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise SystemExit(f"Parquet is missing required columns: {missing_required}")

    # The terrain columns may legitimately be absent: the 1-year parquet does not store
    # them, and train.py's ensure_interaction_features() derives them at fit time using the
    # grid-neighbour method. Writing real DEM values into the parquet is what stops that
    # derivation from happening, because ensure_interaction_features only fills columns
    # that are missing. So create them here when absent rather than refusing to run.
    absent_terrain = [c for c in TERRAIN_COLUMNS if c not in df.columns]
    if absent_terrain:
        print(f"  note: parquet does not store {len(absent_terrain)} terrain column(s):")
        print(f"        {', '.join(absent_terrain)}")
        print("        They are normally derived at train time by the grid-neighbour method.")
        print("        Writing DEM values into the parquet overrides that derivation, because")
        print("        ensure_interaction_features() only computes columns that are missing.")
        for col in absent_terrain:
            df[col] = np.nan

    points = (
        df[["point_id", "latitude", "longitude", "elevation_m"]]
        .drop_duplicates(subset="point_id")
        .sort_values("point_id")
        .reset_index(drop=True)
    )
    print(f"  points   : {len(points)}")
    print()

    # ── recompute terrain per point ──────────────────────────────────────
    new_terrain: dict[str, dict[str, float]] = {}
    uncovered: list[str] = []
    elev_deltas: list[float] = []

    for _, row in points.iterrows():
        pid = row["point_id"]
        t = extract_terrain(float(row["latitude"]), float(row["longitude"]), args.dem_dir)
        if t["elevation_m"] is None:
            uncovered.append(pid)
            continue
        new_terrain[pid] = {
            "topographic_position_index": float(t[args.tpi_scale]),
            "slope_magnitude_deg": float(t["slope_magnitude_deg"]),
            "aspect_sin": float(t["aspect_sin"]),
            "aspect_cos": float(t["aspect_cos"]),
            "dem_elevation_m": float(t["elevation_m"]),
        }
        elev_deltas.append(float(t["elevation_m"]) - float(row["elevation_m"]))

    if uncovered:
        print(f"! {len(uncovered)} of {len(points)} points are NOT covered by the DEM:")
        print(f"    {', '.join(uncovered[:10])}{' ...' if len(uncovered) > 10 else ''}")
        print("  Patching only some points would mix two terrain methods inside one")
        print("  training set, which is worse than either method alone. Aborting.")
        raise SystemExit(1)

    print("  all points covered by DEM")
    print()

    # ── report what changes ──────────────────────────────────────────────
    print("-" * 78)
    print("CHANGE SUMMARY (per spatial point, not per row)")
    print("-" * 78)
    old_by_point = (
        df.groupby("point_id")[TERRAIN_COLUMNS].first().reindex(list(new_terrain.keys()))
    )
    stats: dict[str, Any] = {}
    for col in TERRAIN_COLUMNS:
        old = old_by_point[col].to_numpy(dtype=float)
        new = np.array([new_terrain[p][col] for p in old_by_point.index], dtype=float)
        if np.all(np.isnan(old)):
            # column did not exist in the parquet; nothing to diff against
            stats[col] = {
                "old_mean": None, "new_mean": float(np.mean(new)),
                "old_distinct": 0, "new_distinct": int(len(set(np.round(new, 6)))),
                "mean_abs_change": None, "max_abs_change": None,
            }
            print(f"  {col}")
            print(f"      old: (not stored in parquet)")
            print(f"      new: mean {np.mean(new):+9.3f}  "
                  f"{stats[col]['new_distinct']:3d} distinct values")
            continue
        d = new - old
        stats[col] = {
            "old_mean": float(np.mean(old)), "new_mean": float(np.mean(new)),
            "old_distinct": int(len(set(np.round(old, 6)))),
            "new_distinct": int(len(set(np.round(new, 6)))),
            "mean_abs_change": float(np.mean(np.abs(d))),
            "max_abs_change": float(np.max(np.abs(d))),
        }
        print(f"  {col}")
        print(f"      old: mean {np.mean(old):+9.3f}  {stats[col]['old_distinct']:3d} distinct values")
        print(f"      new: mean {np.mean(new):+9.3f}  {stats[col]['new_distinct']:3d} distinct values")
        print(f"      change: mean|d| {np.mean(np.abs(d)):.3f}   max|d| {np.max(np.abs(d)):.3f}")

    ed = np.array(elev_deltas)
    print()
    print(f"  elevation_m (left UNCHANGED in the parquet, shown for reference):")
    print(f"      DEM minus parquet: mean {ed.mean():+.2f} m, mean|d| {np.abs(ed).mean():.2f} m, "
          f"max|d| {np.abs(ed).max():.2f} m")
    dem_mean_elev = float(np.mean([v["dem_elevation_m"] for v in new_terrain.values()]))
    parquet_mean_elev = float(points["elevation_m"].mean())
    print(f"      domain mean elevation: parquet {parquet_mean_elev:.2f} m -> DEM {dem_mean_elev:.2f} m")
    print()
    print("  DECISION NEEDED: elevation_m feeds delta_elevation_m and")
    print("  theoretical_lapse_delta_c, the model's #1 and #2 features. Switching it to")
    print("  the DEM means domain_mean_elevation_m changes too, and the live anchor set")
    print("  has to be rechecked against the new mean. This script deliberately does not")
    print("  touch elevation_m -- decide that with Lead ML, then use --patch-elevation.")

    if args.dry_run:
        print()
        print("  --dry-run: nothing written")
        report = ROOT_DIR / "data" / "processed" / "audit" / "p0_parquet_terrain_change.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        with open(report, "w", encoding="utf-8") as f:
            json.dump({
                "input": str(in_path), "rows": int(len(df)), "points": int(len(points)),
                "tpi_scale": args.tpi_scale, "columns": stats,
                "domain_mean_elevation_parquet": parquet_mean_elev,
                "domain_mean_elevation_dem": dem_mean_elev,
            }, f, indent=2)
        print(f"  wrote {report.relative_to(ROOT_DIR)}")
        return

    # ── apply ────────────────────────────────────────────────────────────
    for col in TERRAIN_COLUMNS:
        df[col] = df["point_id"].map({p: v[col] for p, v in new_terrain.items()}).astype(float)

    # sloped_solar_insolation is a function of slope and aspect, so it must follow
    if "sloped_solar_insolation" in df.columns and {"hour_sin", "hour_cos", "solar_radiation_w_m2"} <= set(df.columns):
        alignment = df["aspect_sin"] * df["hour_sin"] + df["aspect_cos"] * df["hour_cos"]
        slope_factor = np.sin(np.radians(df["slope_magnitude_deg"]))
        df["sloped_solar_insolation"] = (df["solar_radiation_w_m2"] / 1000.0) * (
            1.0 + slope_factor * alignment
        )
        print()
        print("  recomputed sloped_solar_insolation from the new slope and aspect")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print()
    print("=" * 78)
    print(f"  wrote {out_path.relative_to(ROOT_DIR)}")
    print(f"  original left untouched: {in_path.relative_to(ROOT_DIR)}")
    print()
    print("  NEXT: Lead ML retrains on the patched file, then inference must read terrain")
    print("  from the same DEM via dem_source.extract_terrain -- not from terrain.py's")
    print("  analytical fallback, or the skew comes straight back.")
    print("=" * 78)


if __name__ == "__main__":
    main()
