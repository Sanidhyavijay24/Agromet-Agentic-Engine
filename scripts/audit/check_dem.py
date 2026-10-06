"""
@file check_dem.py
@description DEM doctor: tells you whether your Bhoonidhi CartoDEM files are in the right
             place, in a readable format, and cover the area the project needs.
@module scripts/audit

Runs offline. Reads nothing but the DEM files themselves.

RUN
---
    python scripts/audit/check_dem.py
    python scripts/audit/check_dem.py --dem-dir "D:/downloads/bhoonidhi"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

TRAIN_BOX = {"north": 27.2, "south": 26.5, "west": 75.5, "east": 76.2}
PILOTS = {
    "Kadera": (26.9124, 75.7873),
    "Bhankri": (26.8761, 75.8512),
    "Tunga": (26.9891, 75.9834),
    "Dadhikar": (27.6180, 76.5620),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem-dir", default=None)
    args = ap.parse_args()

    from services.ml_downscaler.dem_source import DEM_DATA_DIR, dem_availability, extract_terrain

    directory = Path(args.dem_dir) if args.dem_dir else DEM_DATA_DIR

    print("=" * 76)
    print("DEM DOCTOR")
    print("=" * 76)
    print(f"  looking in: {directory}")
    print()

    # ── 1. does the directory exist and what is in it ────────────────────
    if not directory.exists():
        print("  [X] That directory does not exist.")
        print()
        print("  Create it and put your CartoDEM GeoTIFFs inside:")
        print(f"      mkdir -p \"{directory}\"")
        print()
        print("  Or point the project at wherever your files already are:")
        print("      set DEM_DATA_DIR=D:\\path\\to\\your\\dem          (cmd)")
        print("      $env:DEM_DATA_DIR='D:\\path\\to\\your\\dem'       (PowerShell)")
        print("      python scripts/audit/check_dem.py --dem-dir D:/path/to/your/dem")
        raise SystemExit(1)

    all_files = [p for p in directory.rglob("*") if p.is_file()]
    tif = [p for p in all_files if p.suffix.lower() in (".tif", ".tiff")]
    hgt = [p for p in all_files if p.suffix.lower() == ".hgt"]
    zips = [p for p in all_files if p.suffix.lower() == ".zip"]
    other = [p for p in all_files if p not in tif + hgt + zips]

    print(f"  files found: {len(all_files)}")
    print(f"    GeoTIFF (.tif/.tiff) : {len(tif)}")
    print(f"    SRTM    (.hgt)       : {len(hgt)}")
    print(f"    archives (.zip)      : {len(zips)}")
    if other:
        print(f"    other                : {len(other)}  ({', '.join(sorted({p.suffix or 'no ext' for p in other}))})")

    if zips and not tif and not hgt:
        print()
        print("  [!] You have zip archives but no extracted rasters.")
        print("      Bhoonidhi delivers CartoDEM as zips. Extract them first:")
        for z in zips[:5]:
            try:
                with zipfile.ZipFile(z) as zf:
                    inner = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
                print(f"        {z.name}  ->  {len(inner)} raster(s) inside"
                      + (f": {inner[0]}" if inner else " (no .tif found!)"))
            except Exception as exc:
                print(f"        {z.name}  ->  could not read: {exc}")
        print()
        print("      PowerShell:  Expand-Archive -Path *.zip -DestinationPath .")
        raise SystemExit(1)

    if not tif and not hgt:
        print()
        print("  [X] No readable DEM rasters (.tif/.tiff/.hgt) here.")
        raise SystemExit(1)

    # ── 2. can rasterio read them ────────────────────────────────────────
    if tif:
        try:
            import rasterio  # noqa: F401
            print()
            print("  [OK] rasterio is installed")
        except ImportError:
            print()
            print("  [X] rasterio is NOT installed, so GeoTIFF files cannot be read.")
            print("      pip install rasterio")
            raise SystemExit(1)

    # ── 3. probe each tile ───────────────────────────────────────────────
    print()
    avail = dem_availability(directory)
    if not avail["available"]:
        print("  [X] Files are present but none could be indexed.")
        print("      Most common cause: the raster is in a PROJECTED CRS (UTM) rather than")
        print("      geographic lat/lon. Reproject to EPSG:4326:")
        print("          rio warp input.tif output_4326.tif --dst-crs EPSG:4326")
        print("      (or gdalwarp -t_srs EPSG:4326 input.tif output_4326.tif)")
        raise SystemExit(1)

    print(f"  [OK] {avail['tile_count']} usable tile(s):")
    for t in avail["tiles"][:15]:
        w, s, e, n = t["bounds"]
        px = t["approx_pixel_m"]
        res = f"~{px[0]:.0f}m x {px[1]:.0f}m" if px else "unknown res"
        print(f"      {t['name']:<38} {s:7.3f}-{n:7.3f}N  {w:7.3f}-{e:7.3f}E  {res}")
    if avail["tile_count"] > 15:
        print(f"      ... and {avail['tile_count'] - 15} more")

    cov = avail["coverage"]
    print()
    print(f"  total coverage: {cov['south']:.3f}-{cov['north']:.3f} N, "
          f"{cov['west']:.3f}-{cov['east']:.3f} E")

    # resolution sanity
    for t in avail["tiles"]:
        px = t["approx_pixel_m"]
        if px and max(px) > 100:
            print(f"  [!] {t['name']} has ~{max(px):.0f} m pixels. CartoDEM should be ~30 m.")
            print("      A coarse raster will not resolve field-scale slope or aspect.")
            break

    # ── 4. coverage of what the project needs ────────────────────────────
    print()
    print("-" * 76)
    print("COVERAGE CHECK")
    print("-" * 76)

    covers_box = (
        cov["south"] <= TRAIN_BOX["south"] and cov["north"] >= TRAIN_BOX["north"]
        and cov["west"] <= TRAIN_BOX["west"] and cov["east"] >= TRAIN_BOX["east"]
    )
    print(f"  training box (26.5-27.2 N, 75.5-76.2 E): {'COVERED' if covers_box else 'NOT COVERED'}")
    if covers_box:
        print("      -> the TRAINING side can be fixed (patch the parquet, then retrain)")
    else:
        print("      -> only inference can be fixed; training keeps the old terrain method.")
        print("         You need CartoDEM tiles spanning 26N-28N, 75E-77E for the full fix.")

    print()
    print("  pilot panchayats:")
    ok = 0
    for name, (lat, lon) in PILOTS.items():
        t = extract_terrain(lat, lon, directory)
        if t["elevation_m"] is None:
            print(f"      {name:<10} NOT COVERED   ({t['note']})")
        else:
            ok += 1
            asp = f"{t['aspect_deg']:.0f}deg" if t["aspect_deg"] is not None else "flat"
            print(f"      {name:<10} elev {t['elevation_m']:7.1f} m   "
                  f"slope {t['slope_magnitude_deg']:5.2f}deg   aspect {asp:>7}   "
                  f"TPI(300m) {t['tpi_300m']:+7.2f}")

    print()
    if ok == len(PILOTS) and covers_box:
        print("  [OK] Everything the Phase 0 scripts need is in place. Next:")
        print(f"      python scripts/audit/p0_terrain_compare.py --dem-dir \"{directory}\"")
        print(f"      python scripts/audit/p0_patch_parquet_terrain.py --dem-dir \"{directory}\" --dry-run")
    elif ok:
        print(f"  [!] {ok}/{len(PILOTS)} pilots covered. Partial coverage is enough to inspect,")
        print("      but p0_patch_parquet_terrain.py will refuse to run rather than mix two")
        print("      terrain methods inside one training set.")
    print("=" * 76)


if __name__ == "__main__":
    main()
