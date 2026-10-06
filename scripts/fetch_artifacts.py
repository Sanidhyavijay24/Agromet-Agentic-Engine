"""
@file fetch_artifacts.py
@description Helper script to download Zone XIV flagship model and DEM tiles from Kaggle.
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


def download_via_kagglehub(target_dir: Path) -> Path | None:
    """Download dataset via kagglehub Python library."""
    try:
        import kagglehub
        print(f"Downloading {KAGGLE_DATASET} via kagglehub...")
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        return Path(path)
    except Exception as e:
        print(f"kagglehub download unavailable: {e}")
        return None


def download_via_cli(target_dir: Path) -> bool:
    """Download dataset via kaggle CLI."""
    try:
        print(f"Executing: kaggle datasets download -d {KAGGLE_DATASET} -p {target_dir}")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(target_dir)],
            check=True
        )
        # Unzip if downloaded as zip
        zip_path = target_dir / (KAGGLE_DATASET.split("/")[-1] + ".zip")
        if zip_path.exists():
            print(f"Unzipping {zip_path.name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            zip_path.unlink()
        return True
    except Exception as e:
        print(f"Kaggle CLI download failed: {e}")
        return False


def place_artifacts(src_dir: Path) -> None:
    """Move relevant model and DEM files to repo directories."""
    MODEL_DEST.mkdir(parents=True, exist_ok=True)
    DEM_DEST.mkdir(parents=True, exist_ok=True)

    placed_model = False
    dem_count = 0

    for root, _, files in os.walk(src_dir):
        for f in files:
            p = Path(root) / f
            if f.startswith("residual_model_acz_") and f.endswith(".joblib"):
                dest = MODEL_DEST / f
                shutil.copy2(p, dest)
                print(f"Placed model artifact: {dest.relative_to(ROOT_DIR)}")
                placed_model = True
            elif f.endswith(".hgt"):
                dest = DEM_DEST / f
                shutil.copy2(p, dest)
                dem_count += 1

    if dem_count > 0:
        print(f"Placed {dem_count} DEM tile(s) into {DEM_DEST.relative_to(ROOT_DIR)}")

    if not placed_model:
        print(f"Warning: residual_model_acz_14.joblib was not found in {src_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch model artifacts and DEM tiles from Kaggle.")
    parser.add_argument("--source-dir", type=str, help="Path to already downloaded Kaggle files.")
    args = parser.parse_args()

    if args.source_dir:
        src = Path(args.source_dir)
        if not src.exists():
            print(f"Source directory does not exist: {src}")
            sys.exit(1)
        place_artifacts(src)
        return

    tmp_dir = ROOT_DIR / "data" / "raw" / "kaggle_temp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = download_via_kagglehub(tmp_dir)
    if downloaded_path and downloaded_path.exists():
        place_artifacts(downloaded_path)
        return

    if download_via_cli(tmp_dir):
        place_artifacts(tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    print("\nCould not automatically download dataset.")
    print("Please download manually:")
    print(f"  kaggle datasets download -d {KAGGLE_DATASET}")
    print("Or visit:")
    print(f"  https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
    print("Then run: python scripts/fetch_artifacts.py --source-dir <unzipped_folder>")


if __name__ == "__main__":
    main()
