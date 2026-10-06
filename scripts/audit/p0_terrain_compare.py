"""
@file p0_terrain_compare.py
@description PHASE 0 AUDIT: three-way comparison of terrain and elevation sources.
@module scripts/audit

WHAT THIS ANSWERS
-----------------
1. Does the Bhoonidhi CartoDEM extract actually cover the 36-point training box
   (26.5-27.2 N, 75.5-76.2 E)? If not, the training side cannot be fixed.
2. For all 71 registry panchayats: how far apart are the registry elevation,
   the Open-Meteo elevation the model trained against, and the CartoDEM elevation?
   (The Bhankri question: 398 m vs 516 m.)
3. For the 4 terrain features actually fed to the model, how far apart are
   (a) the analytical formula currently used at inference,
   (b) the grid-neighbour method used during training, and
   (c) real CartoDEM values?

RUN
---
    python scripts/audit/p0_terrain_compare.py
    python scripts/audit/p0_terrain_compare.py --skip-open-meteo   # offline
    python scripts/audit/p0_terrain_compare.py --dem-dir D:/bhoonidhi/cartodem

Writes: data/processed/audit/p0_terrain_comparison.csv
        data/processed/audit/p0_terrain_summary.json
Nothing is modified in the model, the parquet, or the terrain cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUT_DIR = ROOT_DIR / "data" / "processed" / "audit"

# The training lattice, reproduced exactly as services/ml_downscaler/dataset.py builds it
TRAIN_BOX = {"north": 27.2, "south": 26.5, "west": 75.5, "east": 76.2, "step": 0.14}


def build_training_grid() -> list[dict[str, Any]]:
    lats = np.arange(TRAIN_BOX["south"], TRAIN_BOX["north"] + TRAIN_BOX["step"] / 2, TRAIN_BOX["step"])
    lons = np.arange(TRAIN_BOX["west"], TRAIN_BOX["east"] + TRAIN_BOX["step"] / 2, TRAIN_BOX["step"])
    out, idx = [], 1
    for la in lats:
        for lo in lons:
            out.append({"point_id": f"GRID_{idx:03d}", "latitude": round(float(la), 4),
                        "longitude": round(float(lo), 4)})
            idx += 1
    return out


def fetch_open_meteo_elevations(coords: list[tuple[float, float]]) -> list[float | None]:
    """Batch elevation lookup. This is the source the model's elevation feature was trained on."""
    import requests

    out: list[float | None] = []
    CHUNK = 100
    for i in range(0, len(coords), CHUNK):
        chunk = coords[i : i + CHUNK]
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/elevation",
                params={
                    "latitude": ",".join(str(a) for a, _ in chunk),
                    "longitude": ",".join(str(b) for _, b in chunk),
                },
                timeout=60,
            )
            r.raise_for_status()
            vals = r.json().get("elevation", [])
            out.extend([float(v) if v is not None else None for v in vals])
            if len(vals) != len(chunk):
                out.extend([None] * (len(chunk) - len(vals)))
        except Exception as exc:
            print(f"  ! Open-Meteo elevation request failed for chunk {i}: {exc}")
            out.extend([None] * len(chunk))
    return out


def formula_terrain(elevation_m: float, domain_mean: float = 380.0) -> dict[str, float]:
    """
    Reproduces services/ml_downscaler/terrain.py analytical fallback: the values currently
    cached in panchayat_terrain_metadata.json and fed to the model at inference time.
    """
    delta = elevation_m - domain_mean
    tpi = delta * 0.82
    rel = abs(delta)
    if rel < 10.0:
        slope = 0.5
    elif rel < 30.0:
        slope = 1.2 + (rel - 10.0) * 0.05
    elif rel < 60.0:
        slope = 2.2 + (rel - 30.0) * 0.08
    else:
        slope = 4.6 + min(15.0, (rel - 60.0) * 0.10)
    aspect_deg = 135.0 if delta >= 0 else 225.0
    return {
        "topographic_position_index": round(tpi, 3),
        "slope_magnitude_deg": round(slope, 3),
        "aspect_deg": aspect_deg,
        "aspect_sin": round(math.sin(math.radians(aspect_deg)), 4),
        "aspect_cos": round(math.cos(math.radians(aspect_deg)), 4),
    }


def grid_neighbour_terrain(points: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """
    Reproduces the TRAINING-side terrain method from scripts/ingest_multi_year_dataset.py
    (lines ~202-232): TPI and gradients computed from other grid points within 0.28 deg
    (about 30 km), not from a DEM.

    NOTE ON A BUG THIS EXPOSES: the original computes
        aspect_rad = arctan2(grad_ns, grad_ew)
    which is an angle measured counter-clockwise from EAST, then feeds its sin/cos to the
    model as if they were azimuth-from-north components. It is reproduced faithfully here
    so the comparison shows what the model was actually trained on.
    """
    have_elev = [p for p in points if p.get("elevation_m") is not None]
    if not have_elev:
        return {}
    lat = np.array([p["latitude"] for p in have_elev])
    lon = np.array([p["longitude"] for p in have_elev])
    elev = np.array([float(p["elevation_m"]) for p in have_elev])
    ids = [p["point_id"] for p in have_elev]

    out: dict[str, dict[str, float]] = {}
    for i, pid in enumerate(ids):
        dists = np.sqrt((lat - lat[i]) ** 2 + (lon - lon[i]) ** 2)
        mask = (dists > 0) & (dists <= 0.28)
        if not mask.any():
            out[pid] = {"topographic_position_index": 0.0, "slope_magnitude_deg": 0.5,
                        "aspect_sin": 0.0, "aspect_cos": 1.0}
            continue
        tpi = float(elev[i] - elev[mask].mean())
        d_lat = lat[mask] - lat[i]
        d_lon = lon[mask] - lon[i]
        d_elev = elev[mask] - elev[i]
        grad_ns = float(np.mean(d_elev / np.maximum(1e-5, np.abs(d_lat * 111000.0)) * np.sign(d_lat)))
        grad_ew = float(np.mean(d_elev / np.maximum(1e-5, np.abs(d_lon * 100000.0)) * np.sign(d_lon)))
        slope = float(np.degrees(np.arctan(math.hypot(grad_ns, grad_ew))))
        aspect_rad = float(np.arctan2(grad_ns, grad_ew))
        out[pid] = {
            "topographic_position_index": round(tpi, 3),
            "slope_magnitude_deg": round(slope, 4),
            "aspect_sin": round(math.sin(aspect_rad), 4),
            "aspect_cos": round(math.cos(aspect_rad), 4),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem-dir", default=None, help="Directory holding CartoDEM GeoTIFF or .hgt tiles")
    ap.add_argument("--skip-open-meteo", action="store_true", help="Skip network elevation lookups")
    args = ap.parse_args()

    from services.api.panchayat_registry import PANCHAYAT_REGISTRY
    from services.ml_downscaler.dem_source import dem_availability, extract_terrain

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    # ── 1. DEM availability and coverage of the training box ─────────────
    print("=" * 78)
    print("1. DEM AVAILABILITY")
    print("=" * 78)
    avail = dem_availability(args.dem_dir)
    summary["dem"] = avail
    if not avail["available"]:
        print(f"  NO DEM TILES FOUND in {avail['directory']}")
        print("  Put the Bhoonidhi CartoDEM GeoTIFFs there, or pass --dem-dir.")
        print("  Every terrain value below will come from the formula, which is the bug.")
    else:
        print(f"  directory : {avail['directory']}")
        print(f"  tiles     : {avail['tile_count']}")
        for t in avail["tiles"][:12]:
            px = t["approx_pixel_m"]
            print(f"    - {t['name']:<34} {t['kind']:<18} {t['shape']} "
                  f"~{px[0] if px else '?'}m x {px[1] if px else '?'}m")
        if avail["tile_count"] > 12:
            print(f"    ... and {avail['tile_count'] - 12} more")
        cov = avail["coverage"]
        print(f"  coverage  : {cov['south']:.3f}-{cov['north']:.3f} N, "
              f"{cov['west']:.3f}-{cov['east']:.3f} E")
        covers_box = (
            cov["south"] <= TRAIN_BOX["south"] and cov["north"] >= TRAIN_BOX["north"]
            and cov["west"] <= TRAIN_BOX["west"] and cov["east"] >= TRAIN_BOX["east"]
        )
        summary["covers_training_box"] = covers_box
        print(f"  training box (26.5-27.2 N, 75.5-76.2 E) covered: "
              f"{'YES - training side can be fixed' if covers_box else 'NO - inference only'}")

    # ── 2. Training grid: three terrain methods compared ─────────────────
    print()
    print("=" * 78)
    print("2. TRAINING GRID (36 points)")
    print("=" * 78)
    grid = build_training_grid()
    if not args.skip_open_meteo:
        print(f"  fetching Open-Meteo elevation for {len(grid)} grid points...")
        oms = fetch_open_meteo_elevations([(p["latitude"], p["longitude"]) for p in grid])
        for p, e in zip(grid, oms):
            p["elevation_m"] = e
    else:
        for p in grid:
            p["elevation_m"] = None

    for p in grid:
        p["dem"] = extract_terrain(p["latitude"], p["longitude"], args.dem_dir)

    neighbour = grid_neighbour_terrain(grid)

    dem_elevs = [p["dem"]["elevation_m"] for p in grid if p["dem"]["elevation_m"] is not None]
    om_elevs = [p["elevation_m"] for p in grid if p["elevation_m"] is not None]
    if om_elevs:
        print(f"  Open-Meteo elevation : mean {np.mean(om_elevs):7.2f} m  "
              f"range {min(om_elevs):.0f}-{max(om_elevs):.0f}")
    if dem_elevs:
        print(f"  CartoDEM elevation   : mean {np.mean(dem_elevs):7.2f} m  "
              f"range {min(dem_elevs):.0f}-{max(dem_elevs):.0f}  ({len(dem_elevs)}/36 points)")
    if om_elevs and dem_elevs and len(dem_elevs) == len(om_elevs) == len(grid):
        d = np.array(dem_elevs) - np.array(om_elevs)
        print(f"  difference           : mean {d.mean():+.2f} m  "
              f"mean|d| {np.abs(d).mean():.2f} m  max|d| {np.abs(d).max():.2f} m")
        summary["grid_elevation_delta"] = {
            "mean": float(d.mean()), "mean_abs": float(np.abs(d).mean()),
            "max_abs": float(np.abs(d).max()),
        }
        print()
        print("  IMPORTANT: the model's domain_mean_elevation_m is baked into the artifact.")
        print(f"    Open-Meteo grid mean : {np.mean(om_elevs):.2f} m  (what the model expects)")
        print(f"    CartoDEM grid mean   : {np.mean(dem_elevs):.2f} m  (after a terrain fix)")
        print("    If the training data switches to CartoDEM, domain_mean_elevation_m must be")
        print("    recomputed from it and the anchor-point set re-checked against the new mean.")

    # terrain feature comparison on the grid
    rows_grid = []
    for p in grid:
        nb = neighbour.get(p["point_id"], {})
        dem = p["dem"]
        rows_grid.append({
            "kind": "grid",
            "id": p["point_id"],
            "latitude": p["latitude"],
            "longitude": p["longitude"],
            "elev_registry": "",
            "elev_open_meteo": p["elevation_m"] if p["elevation_m"] is not None else "",
            "elev_cartodem": dem["elevation_m"] if dem["elevation_m"] is not None else "",
            "tpi_training_neighbour": nb.get("topographic_position_index", ""),
            "tpi_formula": "",
            "tpi_cartodem_300m": dem["tpi_300m"] if dem["tpi_300m"] is not None else "",
            "tpi_cartodem_2000m": dem["tpi_2000m"] if dem["tpi_2000m"] is not None else "",
            "slope_training_neighbour": nb.get("slope_magnitude_deg", ""),
            "slope_formula": "",
            "slope_cartodem": dem["slope_magnitude_deg"] if dem["slope_magnitude_deg"] is not None else "",
            "aspect_sin_training_neighbour": nb.get("aspect_sin", ""),
            "aspect_sin_formula": "",
            "aspect_sin_cartodem": dem["aspect_sin"] if dem["aspect_sin"] is not None else "",
            "aspect_deg_cartodem": dem["aspect_deg"] if dem["aspect_deg"] is not None else "",
            "dem_source": dem["source"],
            "note": dem["note"],
        })

    if neighbour and dem_elevs:
        nb_slope = [v["slope_magnitude_deg"] for v in neighbour.values()]
        dem_slope = [p["dem"]["slope_magnitude_deg"] for p in grid
                     if p["dem"]["slope_magnitude_deg"] is not None]
        nb_tpi = [v["topographic_position_index"] for v in neighbour.values()]
        dem_tpi = [p["dem"]["tpi_2000m"] for p in grid if p["dem"]["tpi_2000m"] is not None]
        print()
        print("  terrain features, training method vs CartoDEM:")
        print(f"    slope  training(neighbour) mean {np.mean(nb_slope):6.3f} deg   "
              f"CartoDEM mean {np.mean(dem_slope):6.3f} deg")
        print(f"    TPI    training(neighbour) mean {np.mean(nb_tpi):+7.2f} m     "
              f"CartoDEM(2km) mean {np.mean(dem_tpi):+7.2f} m")
        summary["grid_terrain"] = {
            "slope_training_mean": float(np.mean(nb_slope)),
            "slope_cartodem_mean": float(np.mean(dem_slope)),
            "tpi_training_mean": float(np.mean(nb_tpi)),
            "tpi_cartodem_2km_mean": float(np.mean(dem_tpi)),
        }

    # ── 3. Registry panchayats: elevation source conflict ────────────────
    print()
    print("=" * 78)
    print("3. REGISTRY PANCHAYATS (71 sites)")
    print("=" * 78)
    pans = list(PANCHAYAT_REGISTRY.values())
    if not args.skip_open_meteo:
        print(f"  fetching Open-Meteo elevation for {len(pans)} panchayats...")
        pan_om = fetch_open_meteo_elevations([(p.latitude, p.longitude) for p in pans])
    else:
        pan_om = [None] * len(pans)

    rows_pan = []
    deltas_om, deltas_dem = [], []
    for meta, om in zip(pans, pan_om):
        dem = extract_terrain(meta.latitude, meta.longitude, args.dem_dir)
        fml = formula_terrain(meta.elevation_m)
        d_om = (meta.elevation_m - om) if om is not None else None
        d_dem = (meta.elevation_m - dem["elevation_m"]) if dem["elevation_m"] is not None else None
        if d_om is not None:
            deltas_om.append(abs(d_om))
        if d_dem is not None:
            deltas_dem.append(abs(d_dem))
        rows_pan.append({
            "kind": "panchayat",
            "id": meta.panchayat_id,
            "latitude": meta.latitude,
            "longitude": meta.longitude,
            "elev_registry": meta.elevation_m,
            "elev_open_meteo": om if om is not None else "",
            "elev_cartodem": dem["elevation_m"] if dem["elevation_m"] is not None else "",
            "tpi_training_neighbour": "",
            "tpi_formula": fml["topographic_position_index"],
            "tpi_cartodem_300m": dem["tpi_300m"] if dem["tpi_300m"] is not None else "",
            "tpi_cartodem_2000m": dem["tpi_2000m"] if dem["tpi_2000m"] is not None else "",
            "slope_training_neighbour": "",
            "slope_formula": fml["slope_magnitude_deg"],
            "slope_cartodem": dem["slope_magnitude_deg"] if dem["slope_magnitude_deg"] is not None else "",
            "aspect_sin_training_neighbour": "",
            "aspect_sin_formula": fml["aspect_sin"],
            "aspect_sin_cartodem": dem["aspect_sin"] if dem["aspect_sin"] is not None else "",
            "aspect_deg_cartodem": dem["aspect_deg"] if dem["aspect_deg"] is not None else "",
            "dem_source": dem["source"],
            "note": dem["note"],
        })

    # the 4 pilots, shown explicitly
    def fmt(value: Any, width: int, places: int = 1) -> str:
        """Right-aligned number, or a dash when the value is missing."""
        if value == "" or value is None:
            return "-".rjust(width)
        return f"{float(value):.{places}f}".rjust(width)

    print()
    header = (f"  {'site':<14}{'registry':>10}{'open-meteo':>12}{'cartodem':>10}"
              f"{'slope_fml':>11}{'slope_dem':>11}{'asp_fml':>9}{'asp_dem':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    pilots = ("KADERA_001", "BHANKRI_002", "TUNGA_003", "DADHIKAR_004")
    for r in rows_pan:
        if r["id"] not in pilots:
            continue
        print(f"  {r['id']:<14}"
              f"{fmt(r['elev_registry'], 10)}"
              f"{fmt(r['elev_open_meteo'], 12)}"
              f"{fmt(r['elev_cartodem'], 10)}"
              f"{fmt(r['slope_formula'], 11, 2)}"
              f"{fmt(r['slope_cartodem'], 11, 2)}"
              f"{fmt(r['aspect_sin_formula'], 9, 3)}"
              f"{fmt(r['aspect_sin_cartodem'], 9, 3)}")

    if deltas_om:
        big = [r for r in rows_pan if r["elev_open_meteo"] != ""
               and abs(r["elev_registry"] - r["elev_open_meteo"]) > 50]
        print()
        print(f"  registry vs Open-Meteo : mean|d| {np.mean(deltas_om):.1f} m  "
              f"max|d| {max(deltas_om):.1f} m  |  {len(big)}/{len(pans)} sites differ by >50 m")
        summary["registry_vs_open_meteo"] = {
            "mean_abs": float(np.mean(deltas_om)), "max_abs": float(max(deltas_om)),
            "sites_over_50m": len(big),
        }
        for r in sorted(big, key=lambda x: -abs(x["elev_registry"] - x["elev_open_meteo"]))[:10]:
            print(f"    {r['id']:<16} registry {r['elev_registry']:7.1f}  "
                  f"open-meteo {r['elev_open_meteo']:7.1f}  "
                  f"delta {r['elev_registry'] - r['elev_open_meteo']:+7.1f} m")
    if deltas_dem:
        print()
        print(f"  registry vs CartoDEM   : mean|d| {np.mean(deltas_dem):.1f} m  "
              f"max|d| {max(deltas_dem):.1f} m  ({len(deltas_dem)}/{len(pans)} sites have DEM cover)")
        summary["registry_vs_cartodem"] = {
            "mean_abs": float(np.mean(deltas_dem)), "max_abs": float(max(deltas_dem)),
            "sites_covered": len(deltas_dem),
        }

    # how much variety does each method actually produce?
    print()
    print("  distinct values produced across all 71 panchayats:")
    for label, key in [("formula", "aspect_sin_formula"), ("CartoDEM", "aspect_sin_cartodem")]:
        vals = {r[key] for r in rows_pan if r[key] != ""}
        print(f"    aspect_sin, {label:<9}: {len(vals):3d} distinct value(s)"
              + ("   <-- this is the bug: aspect is a 2-state flag" if len(vals) <= 2 else ""))
    for label, key in [("formula", "slope_formula"), ("CartoDEM", "slope_cartodem")]:
        vals = [r[key] for r in rows_pan if r[key] != ""]
        if vals:
            print(f"    slope_deg,  {label:<9}: {len(set(vals)):3d} distinct, "
                  f"mean {np.mean(vals):.2f}, range {min(vals):.2f}-{max(vals):.2f}")

    # ── write outputs ────────────────────────────────────────────────────
    csv_path = OUT_DIR / "p0_terrain_comparison.csv"
    all_rows = rows_grid + rows_pan
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = OUT_DIR / "p0_terrain_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=" * 78)
    print(f"  wrote {csv_path.relative_to(ROOT_DIR)}")
    print(f"  wrote {json_path.relative_to(ROOT_DIR)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
