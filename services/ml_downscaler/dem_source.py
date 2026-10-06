"""
@file dem_source.py
@description Source-agnostic Digital Elevation Model reader for terrain morphology extraction.
             Reads ISRO Bhoonidhi CartoDEM GeoTIFF tiles (primary) and SRTM .hgt tiles
             (fallback), and computes bilinear elevation, Horn's 3x3 slope, downslope aspect
             azimuth, and true multi-scale Topographic Position Index.
@module services/ml_downscaler

WHY THIS MODULE EXISTS
----------------------
`dem_raster.py` silently returns fabricated values when no DEM tile is found:
elevation collapses to the domain mean, slope to 0.5 deg, and aspect to a hardcoded
135 deg. Those fabricated values were then cached to
`data/processed/panchayat_terrain_metadata.json` and consumed as if they were measured.
This module never fabricates. When no tile covers a coordinate it reports
`source="none"` with None values, so the caller must decide what to do.

CONVENTIONS
-----------
Slope:  degrees from horizontal, Horn (1981) 3x3 kernel.
Aspect: downslope azimuth in degrees clockwise from north (0=N, 90=E, 180=S, 270=W),
        decomposed to sin/cos for cyclical model input.
TPI:    elevation minus the mean elevation of a square neighbourhood of the given
        radius, in metres. Positive = local high (ridge), negative = local low (valley).
        No domain-mean term is mixed in; this is a purely local measure.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np

try:
    from loguru import logger
except ImportError:  # pragma: no cover - logging is not essential to correctness
    import logging

    logger = logging.getLogger("dem_source")  # type: ignore[assignment]

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEM_DATA_DIR = Path(os.getenv("DEM_DATA_DIR", ROOT_DIR / "data" / "raw" / "dem"))

# Radii (metres) for multi-scale TPI. 300 m captures micro-relief (field to hillslope),
# 2000 m captures position within the wider valley/ridge system.
TPI_RADII_M: tuple[int, ...] = (300, 2000)

GEOTIFF_SUFFIXES = (".tif", ".tiff", ".TIF", ".TIFF")
HGT_SUFFIXES = (".hgt", ".HGT")

DemSourceKind = Literal["cartodem_geotiff", "srtm_hgt", "none"]


class TerrainSample(TypedDict):
    """Terrain morphology at a single coordinate. Values are None when no DEM covers it."""

    latitude: float
    longitude: float
    elevation_m: float | None
    slope_magnitude_deg: float | None
    aspect_deg: float | None
    aspect_sin: float | None
    aspect_cos: float | None
    tpi_300m: float | None
    tpi_2000m: float | None
    source: DemSourceKind
    tile: str | None
    note: str


# ─────────────────────────────────────────────────────────────────────────────
#  Tile discovery and access
# ─────────────────────────────────────────────────────────────────────────────


class _Tile:
    """A single DEM tile with its geographic bounds and a lazily opened data handle."""

    def __init__(self, path: Path, kind: DemSourceKind) -> None:
        self.path = path
        self.kind = kind
        self._data: np.ndarray | None = None
        # Geographic bounds, degrees: (west, south, east, north)
        self.bounds: tuple[float, float, float, float] | None = None
        # Pixel size in degrees (lon, lat); lat is positive magnitude
        self.px: tuple[float, float] | None = None
        self.nodata: float | None = None
        self.shape: tuple[int, int] | None = None
        self._probe()

    # -- metadata ---------------------------------------------------------
    def _probe(self) -> None:
        if self.kind == "srtm_hgt":
            self._probe_hgt()
        else:
            self._probe_geotiff()

    def _probe_hgt(self) -> None:
        """SRTM .hgt: name encodes the SW corner, e.g. N26E075. 1x1 degree, 3601 or 1201 square."""
        stem = self.path.stem.upper()
        try:
            lat_sign = 1 if stem[0] == "N" else -1
            lat0 = lat_sign * int(stem[1:3])
            lon_sign = 1 if stem[3] == "E" else -1
            lon0 = lon_sign * int(stem[4:7])
        except (ValueError, IndexError):
            logger.warning("Cannot parse SRTM tile name: {}", self.path.name)
            return

        size_bytes = self.path.stat().st_size
        dim = 3601 if size_bytes >= 3601 * 3601 * 2 else 1201
        self.shape = (dim, dim)
        self.bounds = (float(lon0), float(lat0), float(lon0 + 1), float(lat0 + 1))
        step = 1.0 / (dim - 1)
        self.px = (step, step)
        self.nodata = -32768.0

    def _probe_geotiff(self) -> None:
        """CartoDEM GeoTIFF: read bounds and pixel size from the georeferencing header."""
        try:
            import rasterio
        except ImportError:
            logger.error(
                "rasterio is required to read GeoTIFF DEM tiles. Install it with: pip install rasterio"
            )
            return

        try:
            with rasterio.open(self.path) as ds:
                if ds.crs is not None and not ds.crs.is_geographic:
                    logger.warning(
                        "DEM tile {} is projected ({}), not lat/lon. Reproject it to EPSG:4326 first.",
                        self.path.name,
                        ds.crs,
                    )
                    return
                b = ds.bounds
                self.bounds = (float(b.left), float(b.bottom), float(b.right), float(b.top))
                self.px = (abs(float(ds.transform.a)), abs(float(ds.transform.e)))
                self.nodata = float(ds.nodata) if ds.nodata is not None else None
                self.shape = (int(ds.height), int(ds.width))
        except Exception as exc:
            logger.warning("Failed to probe GeoTIFF {}: {}", self.path.name, exc)

    # -- data -------------------------------------------------------------
    def load(self) -> np.ndarray | None:
        """Load the tile's elevation grid into memory (memory-mapped for .hgt)."""
        if self._data is not None:
            return self._data
        if self.shape is None:
            return None
        try:
            if self.kind == "srtm_hgt":
                self._data = np.memmap(self.path, dtype=">i2", mode="r", shape=self.shape)
            else:
                import rasterio

                with rasterio.open(self.path) as ds:
                    self._data = ds.read(1)
            return self._data
        except Exception as exc:
            logger.warning("Failed to load DEM tile {}: {}", self.path.name, exc)
            return None

    def covers(self, lat: float, lon: float) -> bool:
        if self.bounds is None:
            return False
        w, s, e, n = self.bounds
        return (w <= lon <= e) and (s <= lat <= n)

    def rowcol(self, lat: float, lon: float) -> tuple[float, float]:
        """Fractional (row, col) for a coordinate. Row 0 is the northern edge."""
        assert self.bounds is not None and self.px is not None
        w, s, e, n = self.bounds
        px_lon, px_lat = self.px
        col = (lon - w) / px_lon - 0.5
        row = (n - lat) / px_lat - 0.5
        return row, col


_TILE_INDEX: list[_Tile] | None = None


def discover_tiles(dem_dir: Path | str | None = None, refresh: bool = False) -> list[_Tile]:
    """Index every DEM tile in the data directory. GeoTIFF is preferred over .hgt."""
    global _TILE_INDEX
    if _TILE_INDEX is not None and not refresh:
        return _TILE_INDEX

    directory = Path(dem_dir) if dem_dir else DEM_DATA_DIR
    tiles: list[_Tile] = []
    if not directory.exists():
        logger.warning("DEM directory does not exist: {}", directory)
        _TILE_INDEX = tiles
        return tiles

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in GEOTIFF_SUFFIXES:
            tiles.append(_Tile(path, "cartodem_geotiff"))
        elif path.suffix in HGT_SUFFIXES:
            tiles.append(_Tile(path, "srtm_hgt"))

    usable = [t for t in tiles if t.bounds is not None]
    logger.info(
        "Indexed {} DEM tile(s) in {} ({} GeoTIFF, {} .hgt)",
        len(usable),
        directory,
        sum(1 for t in usable if t.kind == "cartodem_geotiff"),
        sum(1 for t in usable if t.kind == "srtm_hgt"),
    )
    # GeoTIFF first so CartoDEM wins when a coordinate is covered by both
    usable.sort(key=lambda t: 0 if t.kind == "cartodem_geotiff" else 1)
    _TILE_INDEX = usable
    return usable


def _find_tile(lat: float, lon: float, dem_dir: Path | str | None = None) -> _Tile | None:
    for tile in discover_tiles(dem_dir):
        if tile.covers(lat, lon):
            return tile
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Terrain computation
# ─────────────────────────────────────────────────────────────────────────────


def _clean_window(window: np.ndarray, nodata: float | None) -> np.ndarray:
    """Replace nodata and physically impossible values with the window's valid mean."""
    w = np.asarray(window, dtype=np.float64)
    invalid = ~np.isfinite(w) | (w < -500.0) | (w > 9000.0)
    if nodata is not None:
        invalid |= np.isclose(w, nodata)
    if invalid.all():
        return w
    if invalid.any():
        w = np.where(invalid, float(np.nanmean(w[~invalid])), w)
    return w


def _metres_per_degree(lat: float) -> tuple[float, float]:
    """Metres per degree of longitude and latitude at a given latitude (WGS-84 approximation)."""
    lat_rad = math.radians(lat)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
    return abs(m_per_deg_lon), abs(m_per_deg_lat)


def _bilinear(grid: np.ndarray, row: float, col: float) -> float:
    """Bilinear interpolation at fractional (row, col) with edge clamping."""
    h, w = grid.shape
    r0 = int(math.floor(row))
    c0 = int(math.floor(col))
    fr = row - r0
    fc = col - c0
    r0 = max(0, min(h - 2, r0))
    c0 = max(0, min(w - 2, c0))
    z00 = float(grid[r0, c0])
    z01 = float(grid[r0, c0 + 1])
    z10 = float(grid[r0 + 1, c0])
    z11 = float(grid[r0 + 1, c0 + 1])
    top = z00 * (1 - fc) + z01 * fc
    bot = z10 * (1 - fc) + z11 * fc
    return top * (1 - fr) + bot * fr


def _horn_slope_aspect(
    grid: np.ndarray, r: int, c: int, dx_m: float, dy_m: float
) -> tuple[float, float]:
    """
    Horn's 3x3 slope and downslope aspect.

    Returns:
        (slope_degrees, aspect_degrees_clockwise_from_north)

    The 3x3 window is indexed with row increasing southward:
        [ (r-1,c-1) (r-1,c) (r-1,c+1) ]    <- north
        [ (r  ,c-1) (r  ,c) (r  ,c+1) ]
        [ (r+1,c-1) (r+1,c) (r+1,c+1) ]    <- south
    """
    z = grid[r - 1 : r + 2, c - 1 : c + 2]
    # East-positive rate of change
    dz_dx = (
        (z[0, 2] + 2.0 * z[1, 2] + z[2, 2]) - (z[0, 0] + 2.0 * z[1, 0] + z[2, 0])
    ) / (8.0 * dx_m)
    # Row-increasing (southward) rate of change, then flip to north-positive
    dz_drow = (
        (z[2, 0] + 2.0 * z[2, 1] + z[2, 2]) - (z[0, 0] + 2.0 * z[0, 1] + z[0, 2])
    ) / (8.0 * dy_m)
    dz_dnorth = -dz_drow

    slope_deg = math.degrees(math.atan(math.hypot(dz_dx, dz_dnorth)))

    # The gradient points uphill; the downslope direction is its negative.
    if abs(dz_dx) < 1e-12 and abs(dz_dnorth) < 1e-12:
        aspect_deg = float("nan")  # flat: aspect is undefined, not 135 degrees
    else:
        aspect_deg = math.degrees(math.atan2(-dz_dx, -dz_dnorth)) % 360.0
    return slope_deg, aspect_deg


def _tpi(grid: np.ndarray, row: float, col: float, elev: float,
         radius_px_r: int, radius_px_c: int) -> float:
    """Elevation minus the mean of a square neighbourhood, in metres."""
    h, w = grid.shape
    r = int(round(row))
    c = int(round(col))
    r_lo, r_hi = max(0, r - radius_px_r), min(h, r + radius_px_r + 1)
    c_lo, c_hi = max(0, c - radius_px_c), min(w, c + radius_px_c + 1)
    neighbourhood = grid[r_lo:r_hi, c_lo:c_hi]
    if neighbourhood.size == 0:
        return 0.0
    return float(elev - float(np.mean(neighbourhood)))


def extract_terrain(
    lat: float,
    lon: float,
    dem_dir: Path | str | None = None,
) -> TerrainSample:
    """
    Extract terrain morphology at a coordinate from the best available DEM tile.

    Returns a TerrainSample. When no tile covers the coordinate, every measured field
    is None and `source` is "none". This function never substitutes invented values.
    """
    empty: TerrainSample = {
        "latitude": lat,
        "longitude": lon,
        "elevation_m": None,
        "slope_magnitude_deg": None,
        "aspect_deg": None,
        "aspect_sin": None,
        "aspect_cos": None,
        "tpi_300m": None,
        "tpi_2000m": None,
        "source": "none",
        "tile": None,
        "note": "",
    }

    tile = _find_tile(lat, lon, dem_dir)
    if tile is None:
        empty["note"] = "no DEM tile covers this coordinate"
        return empty

    grid_full = tile.load()
    if grid_full is None or tile.px is None:
        empty["note"] = f"tile {tile.path.name} could not be read"
        return empty

    row_f, col_f = tile.rowcol(lat, lon)
    h, w = grid_full.shape

    # Largest neighbourhood we need, in pixels, so we can read one window for everything
    m_per_deg_lon, m_per_deg_lat = _metres_per_degree(lat)
    px_lon_m = tile.px[0] * m_per_deg_lon
    px_lat_m = tile.px[1] * m_per_deg_lat
    max_radius_m = max(TPI_RADII_M)
    pad_c = int(math.ceil(max_radius_m / max(px_lon_m, 1e-6))) + 2
    pad_r = int(math.ceil(max_radius_m / max(px_lat_m, 1e-6))) + 2

    r_center = int(round(row_f))
    c_center = int(round(col_f))
    r_lo, r_hi = max(0, r_center - pad_r), min(h, r_center + pad_r + 1)
    c_lo, c_hi = max(0, c_center - pad_c), min(w, c_center + pad_c + 1)

    window = _clean_window(np.array(grid_full[r_lo:r_hi, c_lo:c_hi]), tile.nodata)
    if window.size == 0 or window.shape[0] < 3 or window.shape[1] < 3:
        empty["source"] = tile.kind
        empty["tile"] = tile.path.name
        empty["note"] = "coordinate too close to tile edge for a 3x3 kernel"
        return empty

    # Coordinates within the extracted window
    row_w = row_f - r_lo
    col_w = col_f - c_lo

    elev = _bilinear(window, row_w, col_w)

    r_i = max(1, min(window.shape[0] - 2, int(round(row_w))))
    c_i = max(1, min(window.shape[1] - 2, int(round(col_w))))
    slope_deg, aspect_deg = _horn_slope_aspect(window, r_i, c_i, px_lon_m, px_lat_m)

    tpi_values: dict[int, float] = {}
    for radius_m in TPI_RADII_M:
        rr = max(1, int(round(radius_m / max(px_lat_m, 1e-6))))
        rc = max(1, int(round(radius_m / max(px_lon_m, 1e-6))))
        tpi_values[radius_m] = _tpi(window, row_w, col_w, elev, rr, rc)

    flat = math.isnan(aspect_deg)
    return {
        "latitude": lat,
        "longitude": lon,
        "elevation_m": round(elev, 2),
        "slope_magnitude_deg": round(slope_deg, 4),
        "aspect_deg": None if flat else round(aspect_deg, 2),
        "aspect_sin": 0.0 if flat else round(math.sin(math.radians(aspect_deg)), 6),
        "aspect_cos": 0.0 if flat else round(math.cos(math.radians(aspect_deg)), 6),
        "tpi_300m": round(tpi_values[300], 3),
        "tpi_2000m": round(tpi_values[2000], 3),
        "source": tile.kind,
        "tile": tile.path.name,
        "note": "flat cell, aspect undefined (sin=cos=0)" if flat else "",
    }


def dem_availability(dem_dir: Path | str | None = None) -> dict[str, Any]:
    """Summarise which DEM tiles are present and what area they cover."""
    tiles = discover_tiles(dem_dir, refresh=True)
    directory = Path(dem_dir) if dem_dir else DEM_DATA_DIR
    if not tiles:
        return {
            "available": False,
            "directory": str(directory),
            "tile_count": 0,
            "tiles": [],
            "coverage": None,
        }
    wests = [t.bounds[0] for t in tiles if t.bounds]
    souths = [t.bounds[1] for t in tiles if t.bounds]
    easts = [t.bounds[2] for t in tiles if t.bounds]
    norths = [t.bounds[3] for t in tiles if t.bounds]
    return {
        "available": True,
        "directory": str(directory),
        "tile_count": len(tiles),
        "tiles": [
            {
                "name": t.path.name,
                "kind": t.kind,
                "bounds": t.bounds,
                "shape": t.shape,
                "pixel_deg": t.px,
                "approx_pixel_m": (
                    round(t.px[0] * _metres_per_degree((t.bounds[1] + t.bounds[3]) / 2)[0], 2),
                    round(t.px[1] * _metres_per_degree((t.bounds[1] + t.bounds[3]) / 2)[1], 2),
                )
                if t.px and t.bounds
                else None,
            }
            for t in tiles
        ],
        "coverage": {
            "west": min(wests),
            "south": min(souths),
            "east": max(easts),
            "north": max(norths),
        },
    }
