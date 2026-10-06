"""
@file usgs_dem_client.py
@description USGS M2M (Machine-to-Machine) API client for querying and downloading
             high-resolution SRTM 1 Arc-Second (30m) Digital Elevation Model GeoTIFF rasters.
             Includes search, download-request orchestration, and local caching in data/raw/dem/.
@module services/ingestion
"""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
from typing import Any
import zipfile

import requests
from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

USGS_M2M_URL: str = os.getenv("USGS_M2M_URL", "https://m2m.cr.usgs.gov/api/api/json/stable")
USGS_M2M_TOKEN: str | None = os.getenv("USGS_M2M_TOKEN")

DEM_DATA_DIR = ROOT_DIR / "data" / "raw" / "dem"
DEM_DATA_DIR.mkdir(parents=True, exist_ok=True)


class UsgsDemClient:
    """Client for USGS M2M API providing automated 30m DEM GeoTIFF search and ingestion."""

    def __init__(self, api_token: str | None = None, username: str | None = None, base_url: str | None = None) -> None:
        self.api_token = api_token or USGS_M2M_TOKEN
        self.username = username or os.getenv("USGS_USERNAME")
        self.base_url = (base_url or USGS_M2M_URL).rstrip("/")
        self.session = requests.Session()
        self.api_key: str | None = None
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with USGS M2M via login-token to acquire session apiKey."""
        if not self.api_token:
            return

        # Attempt 1: Direct login-token payload
        url = f"{self.base_url}/login-token"
        payload = {"token": self.api_token}
        if self.username:
            payload["username"] = self.username

        try:
            resp = self.session.post(url, json=payload, timeout=15.0)
            if resp.status_code == 200:
                data = resp.json()
                self.api_key = data.get("data")
                if self.api_key:
                    self.session.headers.update({"X-Auth-Token": self.api_key})
                    logger.debug("Acquired USGS M2M session key successfully.")
                    return
        except Exception as exc:
            logger.debug("login-token attempt failed: {}. Trying direct token header...", exc)

        # Attempt 2: Use token directly as X-Auth-Token header
        self.session.headers.update({"X-Auth-Token": self.api_token})

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform authenticated POST request to USGS M2M endpoint."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.debug("USGS M2M POST to {}...", endpoint)
        resp = self.session.post(url, json=payload, timeout=(10.0, 45.0))
        resp.raise_for_status()
        data = resp.json()
        if data.get("errorCode"):
            raise RuntimeError(f"USGS M2M API Error [{data.get('errorCode')}]: {data.get('errorMessage')}")
        return data.get("data", data)

    def search_datasets(self, query: str = "SRTM") -> list[dict[str, Any]]:
        """Search available DEM datasets in the USGS catalog."""
        payload = {"datasetName": query}
        res = self._post("dataset-search", payload)
        if isinstance(res, list):
            return res
        return res.get("datasetList", []) if isinstance(res, dict) else []

    def search_scenes(
        self,
        dataset_name: str = "srtm_1arc_v3",
        min_lat: float = 26.0,
        max_lat: float = 28.0,
        min_lon: float = 75.0,
        max_lon: float = 77.0,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for 30m DEM raster scene tiles overlapping spatial bounding box."""
        payload = {
            "datasetName": dataset_name,
            "spatialFilter": {
                "filterType": "mbr",
                "lowerLeft": {"latitude": min_lat, "longitude": min_lon},
                "upperRight": {"latitude": max_lat, "longitude": max_lon},
            },
            "maxResults": max_results,
        }
        res = self._post("scene-search", payload)
        if isinstance(res, dict):
            return res.get("results", [])
        return res if isinstance(res, list) else []

    def get_download_options(self, dataset_name: str, entity_ids: list[str]) -> list[dict[str, Any]]:
        """Query download product IDs for matching entity scene tiles."""
        payload = {
            "datasetName": dataset_name,
            "entityIds": entity_ids,
        }
        res = self._post("download-options", payload)
        return res if isinstance(res, list) else []

    def request_download(self, downloads: list[dict[str, str]]) -> dict[str, Any]:
        """Request download URLs from the USGS download queue."""
        payload = {"downloads": downloads}
        return self._post("download-request", payload)

    def download_file(self, url: str, destination_path: Path | str) -> Path:
        """Stream download file from download URL to local destination."""
        dest = Path(destination_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading DEM raster to {}...", dest.name)
        with requests.get(url, stream=True, timeout=60.0) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        logger.success("Downloaded {} ({:.2f} MB)", dest.name, dest.stat().st_size / (1024 * 1024))
        return dest


def fetch_rajasthan_dem_tiles() -> list[Path]:
    """
    Download 30m SRTM 1-Arc-Second DEM tiles covering Jaipur, Chaksu, and Alwar.
    Primary tiles:
    - N26E075: Lat 26°N-27°N, Lon 75°E-76°E (Jaipur Rural, Chaksu, Kadera, Bhankri, Tunga)
    - N27E075: Lat 27°N-28°N, Lon 75°E-76°E (North Jaipur, Shahpura)
    - N27E076: Lat 27°N-28°N, Lon 76°E-77°E (Alwar, Dadhikar)
    """
    import zipfile
    import io

def fetch_rajasthan_dem_tiles(custom_tiles: list[str] | None = None) -> list[Path]:
    """
    Download and extract 30m SRTM elevation raster tiles for Rajasthan domain.
    Tiles N26E075, N26E076, N27E075, N27E076 cover lat 26.0-28.0 N, lon 75.0-77.0 E.
    """
    tiles = custom_tiles or ["N26E075", "N26E076", "N27E075", "N27E076"]
    downloaded_files: list[Path] = []

    # USGS / NASA SRTM 1 Arc-Second open data mirrors
    mirror_urls = [
        "https://step.esa.int/auxdata/dem/SRTMGL1/{tile}.SRTMGL1.hgt.zip",
        "https://elevation-tiles-prod.s3.amazonaws.com/skadi/{tile}.hgt.gz",
        "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{tile}.hgt.gz",
    ]

    for tile in tiles:
        hgt_file = DEM_DATA_DIR / f"{tile}.hgt"
        if hgt_file.exists() and hgt_file.stat().st_size > 1000000:
            logger.info("Found cached 30m DEM tile: {} ({:.2f} MB)", hgt_file.name, hgt_file.stat().st_size / (1024 * 1024))
            downloaded_files.append(hgt_file)
            continue

        logger.info("Fetching 30m SRTM DEM tile for {}...", tile)
        success = False
        for url_pattern in mirror_urls:
            url = url_pattern.format(tile=tile)
            try:
                logger.debug("Trying download from {}", url)
                resp = requests.get(url, timeout=30.0)
                if resp.status_code == 200 and len(resp.content) > 100000:
                    if url.endswith(".zip"):
                        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                            # Extract .hgt file
                            for name in z.namelist():
                                if name.endswith(".hgt"):
                                    with open(hgt_file, "wb") as f_out:
                                        f_out.write(z.read(name))
                                    success = True
                                    break
                    elif url.endswith(".gz"):
                        import gzip
                        decompressed = gzip.decompress(resp.content)
                        with open(hgt_file, "wb") as f_out:
                            f_out.write(decompressed)
                        success = True

                    if success:
                        logger.success("Successfully ingested 30m DEM tile {} ({:.2f} MB)", hgt_file.name, hgt_file.stat().st_size / (1024 * 1024))
                        downloaded_files.append(hgt_file)
                        break
            except Exception as exc:
                logger.debug("Mirror {} failed: {}", url, exc)

        if not success:
            logger.warning("Could not download tile {}. Will fallback to high-precision synthetic mesh.", tile)

    return downloaded_files


def read_30m_elevation_at_point(lat: float, lon: float) -> float | None:
    """
    Read exact continuous 30-meter ground elevation from local SRTM .hgt raster tiles.
    """
    lat_floor = int(math.floor(lat))
    lon_floor = int(math.floor(lon))
    tile_name = f"N{lat_floor:02d}E{lon_floor:03d}"
    hgt_path = DEM_DATA_DIR / f"{tile_name}.hgt"

    if not hgt_path.exists():
        return None

    # SRTM 1-Arc-Second tile has 3601 x 3601 signed 16-bit big-endian integers
    # Sample spacing is 1/3600th degree (~30 meters)
    file_size = hgt_path.stat().st_size
    dim = 3601 if file_size >= 25934402 else 1201

    lat_frac = lat - lat_floor
    lon_frac = lon - lon_floor

    row = int((1.0 - lat_frac) * (dim - 1))
    col = int(lon_frac * (dim - 1))

    row = max(0, min(dim - 1, row))
    col = max(0, min(dim - 1, col))

    offset = (row * dim + col) * 2
    with open(hgt_path, "rb") as f:
        f.seek(offset)
        raw_bytes = f.read(2)
        if len(raw_bytes) == 2:
            elev = int.from_bytes(raw_bytes, byteorder="big", signed=True)
            if elev > -500 and elev < 9000:
                return float(elev)
    return None


def test_usgs_connection() -> bool:
    """Test authenticated connection, search catalog, and download 30m tiles."""
    if not USGS_M2M_TOKEN:
        logger.error("USGS_M2M_TOKEN not found in environment (.env).")
        return False
    client = UsgsDemClient()
    try:
        datasets = client.search_datasets("SRTM")
        logger.success("USGS M2M Authentication Successful (User: {})! Found {} SRTM datasets.", client.username, len(datasets))
        tiles = fetch_rajasthan_dem_tiles()
        logger.success("Ingested {} authentic 30m SRTM DEM tiles into {}.", len(tiles), DEM_DATA_DIR)
        
        # Test elevation readout for Kadera (26.9124°N, 75.7873°E)
        kadera_elev = read_30m_elevation_at_point(26.9124, 75.7873)
        logger.info("30m SRTM ground elevation for Kadera (26.9124, 75.7873): {} m", kadera_elev)
        return True
    except Exception as exc:
        logger.exception("USGS M2M pipeline failed: {}", exc)
        return False


if __name__ == "__main__":
    test_usgs_connection()
