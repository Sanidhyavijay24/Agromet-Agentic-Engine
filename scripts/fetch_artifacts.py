"""
@file fetch_artifacts.py
@description Helper script to download Zone XIV model or full fleet from Kaggle,
             and fetch the 4 required 30m SRTM DEM tiles for Zone XIV.
@module scripts
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DEST = ROOT_DIR / "services" / "ml_downscaler" / "artifacts" / "zones"
DEM_DEST = ROOT_DIR / "data" / "raw" / "dem"
KAGGLE_DATASET = "sanidhyavijay24/pan-india-15-acz-1km-microclimate-dataset-and-models"
ZONE_14_FILENAME = "residual_model_acz_14.joblib"

# The 4 USGS SRTM 30m tiles required for Zone XIV (Jaipur / Chaksu domain)
ZONE_14_DEM_TILES = ["N26E075", "N26E076", "N27E075", "N27E076"]
ESA_SRTM_MIRROR = "https://step.esa.int/auxdata/dem/SRTMGL1"


def download_single_file_via_kagglehub(filename: str) -> Path | None:
    """Download a single specific file using kagglehub."""
    try:
        import kagglehub
        print(f"Downloading {filename} via kagglehub...")
        path = kagglehub.dataset_download(KAGGLE_DATASET, path=filename)
        return Path(path)
    except Exception as e:
        print(f"kagglehub single-file download unavailable: {e}")
        return None


def download_single_file_via_cli(filename: str, target_dir: Path) -> bool:
    """Download a single file using the kaggle CLI."""
    try:
        print(f"Executing: kaggle datasets download -d {KAGGLE_DATASET} -f {filename} -p {target_dir}")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-f", filename, "-p", str(target_dir)],
            check=True
        )
        zip_path = target_dir / (filename + ".zip")
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            zip_path.unlink()
        return True
    except Exception as e:
        print(f"Kaggle CLI single-file download failed: {e}")
        return False


def download_all_via_kagglehub() -> Path | None:
    """Download entire dataset bundle via kagglehub."""
    try:
        import kagglehub
        print(f"Downloading full dataset bundle {KAGGLE_DATASET} via kagglehub...")
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        return Path(path)
    except Exception as e:
        print(f"kagglehub full download unavailable: {e}")
        return None


def download_all_via_cli(target_dir: Path) -> bool:
    """Download entire dataset bundle via kaggle CLI."""
    try:
        print(f"Executing: kaggle datasets download -d {KAGGLE_DATASET} -p {target_dir}")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(target_dir)],
            check=True
        )
        zip_name = KAGGLE_DATASET.split("/")[-1] + ".zip"
        zip_path = target_dir / zip_name
        if zip_path.exists():
            print(f"Extracting {zip_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            zip_path.unlink()
        return True
    except Exception as e:
        print(f"Kaggle CLI full download failed: {e}")
        return False


def place_artifacts(src_dir: Path, target_model: str | None = None) -> None:
    """Place downloaded files into destination directories."""
    MODEL_DEST.mkdir(parents=True, exist_ok=True)
    DEM_DEST.mkdir(parents=True, exist_ok=True)

    placed_models = 0
    dem_count = 0

    if src_dir.is_file():
        if src_dir.name.endswith(".joblib"):
            dest = MODEL_DEST / src_dir.name
            shutil.copy2(src_dir, dest)
            print(f"Placed model artifact: {dest.relative_to(ROOT_DIR)}")
            placed_models += 1
        return

    for root, _, files in os.walk(src_dir):
        for f in files:
            p = Path(root) / f
            if target_model and f == target_model:
                dest = MODEL_DEST / f
                shutil.copy2(p, dest)
                print(f"Placed flagship model artifact: {dest.relative_to(ROOT_DIR)}")
                placed_models += 1
            elif not target_model and f.startswith("residual_model_acz_") and f.endswith(".joblib"):
                dest = MODEL_DEST / f
                shutil.copy2(p, dest)
                placed_models += 1
            elif f.endswith(".hgt"):
                dest = DEM_DEST / f
                shutil.copy2(p, dest)
                dem_count += 1

    if placed_models > 0:
        print(f"Placed {placed_models} model artifact(s) into {MODEL_DEST.relative_to(ROOT_DIR)}")
    if dem_count > 0:
        print(f"Placed {dem_count} DEM tile(s) into {DEM_DEST.relative_to(ROOT_DIR)}")


def ensure_zone14_dem_tiles() -> None:
    """Download the 4 required 30m SRTM DEM tiles for Zone XIV if not present."""
    DEM_DEST.mkdir(parents=True, exist_ok=True)
    missing_tiles = [tile for tile in ZONE_14_DEM_TILES if not (DEM_DEST / f"{tile}.hgt").exists()]

    if not missing_tiles:
        print(f"All 4 Zone XIV DEM tiles present in {DEM_DEST.relative_to(ROOT_DIR)}: {[f'{t}.hgt' for t in ZONE_14_DEM_TILES]}")
        return

    print(f"Downloading {len(missing_tiles)} missing 30m DEM tile(s) for Zone XIV ({missing_tiles})...")
    for tile in missing_tiles:
        url = f"{ESA_SRTM_MIRROR}/{tile}.SRTMGL1.hgt.zip"
        dest_hgt = DEM_DEST / f"{tile}.hgt"
        try:
            print(f"  Fetching {tile}.SRTMGL1.hgt.zip from public SRTM mirror...")
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    for name in z.namelist():
                        if name.endswith(".hgt"):
                            with open(dest_hgt, "wb") as f:
                                f.write(z.read(name))
                            print(f"  Saved {tile}.hgt ({os.path.getsize(dest_hgt) / (1024*1024):.1f} MB)")
            else:
                print(f"  Failed to fetch {tile} (HTTP {resp.status_code})")
        except Exception as e:
            print(f"  Error fetching {tile}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch model artifacts from Kaggle and DEM tiles.")
    parser.add_argument("--all", action="store_true", help="Download all 15 zone models and full dataset (3.5 GB).")
    parser.add_argument("--source-dir", type=str, help="Path to already downloaded Kaggle files.")
    args = parser.parse_args()

    if args.source_dir:
        src = Path(args.source_dir)
        if not src.exists():
            print(f"Source path does not exist: {src}")
            sys.exit(1)
        place_artifacts(src, None if args.all else ZONE_14_FILENAME)
        ensure_zone14_dem_tiles()
        return

    tmp_dir = ROOT_DIR / "data" / "raw" / "kaggle_temp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.all:
            print("Downloading full 15-zone fleet and multi-year dataset...")
            p = download_all_via_kagglehub()
            if p and p.exists():
                place_artifacts(p, target_model=None)
            elif download_all_via_cli(tmp_dir):
                place_artifacts(tmp_dir, target_model=None)
            else:
                print("Could not complete automatic full download.")
        else:
            print("Downloading only Zone XIV flagship model (residual_model_acz_14.joblib)...")
            p = download_single_file_via_kagglehub(ZONE_14_FILENAME)
            if p and p.exists():
                place_artifacts(p, target_model=ZONE_14_FILENAME)
            elif download_single_file_via_cli(ZONE_14_FILENAME, tmp_dir):
                place_artifacts(tmp_dir, target_model=ZONE_14_FILENAME)
            else:
                print(f"Could not automatically download {ZONE_14_FILENAME}.")
                print("Download manually with:")
                print(f"  kaggle datasets download -d {KAGGLE_DATASET} -f {ZONE_14_FILENAME} -p services/ml_downscaler/artifacts/zones/")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    ensure_zone14_dem_tiles()


if __name__ == "__main__":
    main()
