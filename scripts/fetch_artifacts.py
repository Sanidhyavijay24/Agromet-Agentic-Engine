"""
@file fetch_artifacts.py
@description Helper script to download Zone XIV model or full fleet from Kaggle.
@module scripts
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DEST = ROOT_DIR / "services" / "ml_downscaler" / "artifacts" / "zones"
DEM_DEST = ROOT_DIR / "data" / "raw" / "dem"
KAGGLE_DATASET = "sanidhyavijay24/pan-india-15-acz-1km-microclimate-dataset-and-models"
ZONE_14_FILENAME = "residual_model_acz_14.joblib"


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


def check_dem_status() -> None:
    """Check if local DEM tiles are present."""
    dem_tiles = list(DEM_DEST.glob("*.hgt")) if DEM_DEST.exists() else []
    if dem_tiles:
        print(f"DEM tiles present in data/raw/dem/: {[t.name for t in dem_tiles]}")
    else:
        print("Note: No .hgt DEM tiles found in data/raw/dem/.")
        print("The model falls back to domain-level elevation averages or Open-Meteo elevation API.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch model artifacts from Kaggle.")
    parser.add_argument("--all", action="store_true", help="Download all 15 zone models and full dataset (3.5 GB).")
    parser.add_argument("--source-dir", type=str, help="Path to already downloaded Kaggle files.")
    args = parser.parse_args()

    if args.source_dir:
        src = Path(args.source_dir)
        if not src.exists():
            print(f"Source path does not exist: {src}")
            sys.exit(1)
        place_artifacts(src, None if args.all else ZONE_14_FILENAME)
        check_dem_status()
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

    check_dem_status()


if __name__ == "__main__":
    main()
