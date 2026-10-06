"""
@file build_imd_normals.py
@description Build 1991-2020 RAINFALL normals from IMD's 0.25-degree gridded daily rainfall
             for every registry panchayat, replacing the ERA5 rainfall normals.
@module scripts/climatology

STATUS: NOT EXECUTED DURING DEVELOPMENT. It needs the `imdlib` package and a download of
~30 yearly IMD files (roughly 1 GB) from IMD Pune. Run it, and paste the output back if
anything fails -- the imdlib calls are written defensively but have not been exercised.

WHY
---
The default normals come from ERA5 reanalysis, which is a model. IMD's gridded rainfall is
built from rain-gauge observations (Pai et al. 2014, "Development of a new high spatial
resolution (0.25 x 0.25 degree) long period (1901-2010) daily gridded rainfall data set over
India", Mausam 65(1): 1-18) and is the reference IMD itself uses for departures. For Jaipur,
ERA5 gives an annual normal of ~526 mm, noticeably drier than IMD's figure, so switching
changes the departures farmers see.

Once written, the files are picked up automatically by
services.advisory_engine.climatology.get_normals() for RAINFALL; temperature normals stay
on ERA5.

RUN
---
    pip install imdlib
    python scripts/climatology/build_imd_normals.py --download          # first time
    python scripts/climatology/build_imd_normals.py                     # files already present
    python scripts/climatology/build_imd_normals.py --only KADERA_001 BHANKRI_002
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

IMD_SOURCE_TEMPLATE = (
    "IMD 0.25-degree gridded daily rainfall (Pai et al. 2014), grid cell {lat:.2f}N {lon:.2f}E"
)
MAX_CELL_DISTANCE_DEG = 0.2  # a registry point must lie within one grid cell of a valid cell


def _open_dataset(start: int, end: int, data_dir: Path, download: bool):
    try:
        import imdlib as imd
    except ImportError:
        raise SystemExit("imdlib is not installed. Run: pip install imdlib")

    data_dir.mkdir(parents=True, exist_ok=True)
    if download:
        print(f"downloading IMD gridded rainfall {start}-{end} into {data_dir} (large; be patient)")
        data = imd.get_data("rain", start, end, fn_format="yearwise", file_dir=str(data_dir))
    else:
        data = imd.open_data("rain", start, end, fn_format="yearwise", file_dir=str(data_dir))
    ds = data.get_xarray()

    # Tolerate naming differences between imdlib versions.
    var = "rain" if "rain" in ds.data_vars else list(ds.data_vars)[0]
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    return ds, var, lat_name, lon_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT_DIR / "data" / "raw" / "imd"))
    ap.add_argument("--start", type=int, default=1991)
    ap.add_argument("--end", type=int, default=2020)
    ap.add_argument("--download", action="store_true", help="Download the yearly IMD files first")
    ap.add_argument("--only", nargs="*", help="Panchayat ids to build (default: all in the registry)")
    args = ap.parse_args()

    from services.advisory_engine.climatology import CLIMATOLOGY_DIR, _cache_path, compute_normals
    from services.api.panchayat_registry import PANCHAYAT_REGISTRY
    from services.api.schemas import DailyTemperatureRecord

    ds, var, lat_name, lon_name = _open_dataset(args.start, args.end, Path(args.data_dir), args.download)
    print(f"opened IMD dataset: variable '{var}', {ds.sizes}")

    sites = PANCHAYAT_REGISTRY.values()
    if args.only:
        wanted = set(args.only)
        sites = [s for s in sites if s.panchayat_id in wanted]

    CLIMATOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    built, skipped = 0, []
    for site in sites:
        cell = ds[var].sel({lat_name: site.latitude, lon_name: site.longitude}, method="nearest")
        clat, clon = float(cell[lat_name]), float(cell[lon_name])
        if abs(clat - site.latitude) > MAX_CELL_DISTANCE_DEG or abs(clon - site.longitude) > MAX_CELL_DISTANCE_DEG:
            skipped.append(f"{site.panchayat_id} (nearest cell {clat:.2f},{clon:.2f} too far)")
            continue

        series = cell.to_series()
        records = []
        for ts, value in series.items():
            if value is None or value != value or value < 0:  # NaN or IMD's -999 missing flag
                continue
            d = ts.date() if hasattr(ts, "date") else date.fromisoformat(str(ts)[:10])
            records.append(DailyTemperatureRecord(date=d, precipitation_mm=float(value), source="archive"))
        if len(records) < 0.9 * 365 * (args.end - args.start + 1):
            skipped.append(f"{site.panchayat_id} (only {len(records)} valid days; ocean or masked cell?)")
            continue

        normals = compute_normals(records, site.latitude, site.longitude,
                                  source=IMD_SOURCE_TEMPLATE.format(lat=clat, lon=clon),
                                  period=(args.start, args.end))
        path = _cache_path("imd", site.latitude, site.longitude, CLIMATOLOGY_DIR)
        path.write_text(normals.to_json(), encoding="utf-8")
        annual = sum(v for v in normals.monthly_rain if v is not None)
        jjas = sum(v for v in normals.monthly_rain[5:9] if v is not None)
        print(f"  {site.panchayat_id:<16} cell {clat:.2f},{clon:.2f}  annual {annual:6.0f} mm  JJAS {jjas:6.0f} mm")
        built += 1

    print(f"\nbuilt {built} IMD rainfall normal file(s) in {CLIMATOLOGY_DIR}")
    for s in skipped:
        print(f"  skipped {s}")


if __name__ == "__main__":
    main()
