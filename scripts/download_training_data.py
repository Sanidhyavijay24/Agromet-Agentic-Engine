"""
@file download_training_data.py
@description Extracts full 1-year (2024) historical reanalysis dataset across Jaipur/Chaksu domain for ML downscaler training.
@module scripts
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from loguru import logger
except ImportError:
    import logging

    class _LoguruCompatLogger:
        def __init__(self, name: str) -> None:
            self._logger = logging.getLogger(name)
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
                self._logger.addHandler(handler)
                self._logger.setLevel(logging.INFO)

        def _format_msg(self, msg: str, *args: Any) -> str:
            if args:
                try:
                    return msg.format(*args)
                except Exception:
                    return f"{msg} {args}"
            return msg

        def info(self, msg: str, *args: Any) -> None:
            self._logger.info(self._format_msg(msg, *args))

        def warning(self, msg: str, *args: Any) -> None:
            self._logger.warning(self._format_msg(msg, *args))

        def error(self, msg: str, *args: Any) -> None:
            self._logger.error(self._format_msg(msg, *args))

        def success(self, msg: str, *args: Any) -> None:
            self._logger.info(self._format_msg(msg, *args))

    logger = _LoguruCompatLogger("download_training_data")  # type: ignore[assignment]

import pandas as pd

def _simple_table(rows: list[list[Any]], headers: list[str]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
    
    header_str = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_strs = [" | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row)) for row in rows]
    return f"{header_str}\n{sep_str}\n" + "\n".join(row_strs)

try:
    from tabulate import tabulate
except ImportError:
    tabulate = lambda rows, headers=(), tablefmt="": _simple_table(rows, headers)  # type: ignore[assignment]

from services.ml_downscaler.dataset import (
    DEFAULT_BOUNDING_BOX,
    extract_downscaling_dataset,
    generate_grid_coordinates,
)


def main() -> None:
    """
    Download the complete 1-year historical dataset across the Jaipur/Chaksu domain.
    """
    logger.info("=================================================================")
    logger.info("     SIH 26074: FULL 1-YEAR (2024) DATASET EXTRACTION PIPELINE    ")
    logger.info("=================================================================")

    output_dir = ROOT_DIR / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "jaipur_1yr_training_data.parquet"
    csv_path = output_dir / "jaipur_1yr_training_data.csv"

    start_date = "2024-01-01"
    end_date = "2024-12-31"

    # 1. Generate grid coordinates for Jaipur Rural / Chaksu bounding box
    coords = generate_grid_coordinates(
        north=DEFAULT_BOUNDING_BOX["north"],
        south=DEFAULT_BOUNDING_BOX["south"],
        west=DEFAULT_BOUNDING_BOX["west"],
        east=DEFAULT_BOUNDING_BOX["east"],
        step_deg=0.14,
    )
    logger.info(
        "Initiating extraction for {} spatial grid points ({} to {})...",
        len(coords),
        start_date,
        end_date,
    )

    t0 = time.perf_counter()
    df = extract_downscaling_dataset(
        coords=coords,
        start_date=start_date,
        end_date=end_date,
    )
    elapsed = time.perf_counter() - t0

    # 2. Save extracted dataset
    logger.info("Saving dataset to Parquet and CSV in {}...", output_dir)
    try:
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        parquet_saved = True
    except Exception as exc:
        logger.warning("Could not save to Parquet ({}). Saving CSV as primary format.", exc)
        parquet_saved = False

    df.to_csv(csv_path, index=False)
    logger.success("Saved CSV dataset to {}", csv_path)
    if parquet_saved:
        logger.success("Saved Parquet dataset to {}", parquet_path)

    # 3. Validation & Profiling Metrics
    total_rows = len(df)
    null_counts = int(df.isnull().sum().sum())
    csv_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    parquet_size_mb = os.path.getsize(parquet_path) / (1024 * 1024) if parquet_saved else 0.0

    summary_table = [
        ["Temporal Coverage", f"{start_date} to {end_date} (366 Days, Leap Year)"],
        ["Spatial Points", f"{len(coords)} grid points across Jaipur/Chaksu"],
        ["Total Hourly Rows", f"{total_rows:,}"],
        ["Feature Count", len(df.columns)],
        ["Null Value Count", null_counts],
        ["Extraction Time", f"{elapsed:.2f} seconds"],
        ["Throughput", f"{total_rows / max(elapsed, 0.001):.1f} rows/second"],
        ["Temperature Range (°C)", f"{df['temperature_2m_c'].min():.1f}°C to {df['temperature_2m_c'].max():.1f}°C"],
        ["Precipitation Total (mm)", f"{df['precipitation_mm'].sum():.1f} mm"],
        ["Max Residual |R| (°C)", f"{df['residual_anomaly_c'].abs().max():.2f}°C"],
        ["CSV File Size", f"{csv_size_mb:.2f} MB"],
        ["Parquet File Size", f"{parquet_size_mb:.2f} MB" if parquet_saved else "N/A"],
    ]

    print("\n" + "=" * 65)
    print("        1-YEAR DATASET EXTRACTION SUMMARY REPORT")
    print("=" * 65)
    print(tabulate(summary_table, headers=["Metric", "Result"], tablefmt="fancy_grid"))
    print("=" * 65 + "\n")

    assert null_counts == 0, f"Found {null_counts} null values in dataset!"
    assert total_rows > 300_000, f"Expected >300k rows, got {total_rows}"
    logger.success("1-Year dataset extraction verified and complete!")


if __name__ == "__main__":
    main()
