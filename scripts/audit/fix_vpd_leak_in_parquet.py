"""
@file fix_vpd_leak_in_parquet.py
@description Remove the VPD target leak (audit finding F22) from an existing training parquet,
             without re-ingesting any weather data. Writes a new file; never overwrites.
@module scripts/audit

THE LEAK
--------
scripts/ingest_multi_year_dataset.py computed the model FEATURE vapor_pressure_deficit_kpa from
temperature_2m_c -- the point's own temperature, which is the quantity being predicted (the
target is temperature minus the regional baseline). With humidity and the baseline also given
as features, the model could recover the answer by inverting the VPD formula. The ingest script
is now fixed; this script repairs parquet files built before the fix.

WHAT IT DOES
------------
1. Detects which definition the file uses, by recomputing VPD both ways and comparing:
     from temperature_2m_c  -> LEAKED
     from baseline_temp_c   -> already correct (the script then writes nothing)
2. If leaked: recomputes vapor_pressure_deficit_kpa from the baseline, and the columns built
   from it (latent_cooling_potential, and evaporative_demand if present). The old local-
   temperature VPD is kept as vpd_local_kpa, which the VPD downscaling target legitimately uses.

The model must be RETRAINED on the output for the fix to take effect.

RUN
---
    python scripts/audit/fix_vpd_leak_in_parquet.py --input data/raw/jaipur_10yr_training_data.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def tetens_vpd(temp_c: pd.Series, rh_pct: pd.Series) -> np.ndarray:
    rh = rh_pct.clip(lower=1.0, upper=100.0)
    es = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    return np.maximum(0.0, es - es * (rh / 100.0)).to_numpy()


def detect(df: pd.DataFrame) -> str:
    """'leaked', 'correct' or 'unknown', by which temperature reproduces the stored VPD."""
    if "vapor_pressure_deficit_kpa" not in df.columns:
        return "absent"
    stored = df["vapor_pressure_deficit_kpa"].to_numpy()
    from_local = np.max(np.abs(stored - tetens_vpd(df["temperature_2m_c"], df["relative_humidity_2m_pct"])))
    from_base = np.max(np.abs(stored - tetens_vpd(df["baseline_temp_c"], df["relative_humidity_2m_pct"])))
    if from_local < 1e-6:
        return "leaked"
    if from_base < 1e-6:
        return "correct"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None, help="Defaults to <input stem>_vpdfix.parquet")
    args = ap.parse_args()

    src = Path(args.input) if Path(args.input).is_absolute() else ROOT_DIR / args.input
    out = Path(args.output) if args.output else src.with_name(src.stem + "_vpdfix" + src.suffix)
    if not out.is_absolute():
        out = ROOT_DIR / out
    if out.resolve() == src.resolve():
        sys.exit(f"  --output must differ from --input ({src}); this script never overwrites its input.")
    df = pd.read_parquet(src)
    state = detect(df)
    print(f"  {src.name}: {len(df):,} rows; VPD definition: {state.upper()}")

    if state == "absent":
        print("  no vapor_pressure_deficit_kpa column: it is computed at train time by")
        print("  ensure_interaction_features(), which already uses the baseline. Nothing to fix.")
        return
    if state == "correct":
        print("  VPD is already computed from the baseline. Nothing to fix.")
        return
    if state == "unknown":
        sys.exit("  VPD matches neither definition; inspect how this file was built before patching.")

    df["vpd_local_kpa"] = df["vapor_pressure_deficit_kpa"]
    df["vapor_pressure_deficit_kpa"] = tetens_vpd(df["baseline_temp_c"], df["relative_humidity_2m_pct"])
    fixed = ["vapor_pressure_deficit_kpa"]
    if "latent_cooling_potential" in df.columns:
        df["latent_cooling_potential"] = df["et0_evapotranspiration_mm"] * df["vapor_pressure_deficit_kpa"]
        fixed.append("latent_cooling_potential")
    if "evaporative_demand" in df.columns:
        df["evaporative_demand"] = (df["et0_evapotranspiration_mm"] * df["vapor_pressure_deficit_kpa"]
                                    * (df["solar_radiation_w_m2"] / 1000.0))
        fixed.append("evaporative_demand")

    assert detect(df) == "correct"
    shift = float(np.mean(np.abs(df["vapor_pressure_deficit_kpa"] - df["vpd_local_kpa"])))
    df.to_parquet(out, index=False)
    print(f"  recomputed from the baseline: {', '.join(fixed)} (mean VPD change {shift:.3f} kPa)")
    print(f"  kept the local-temperature VPD as vpd_local_kpa (for the VPD target only)")
    print(f"  wrote {out}")
    print("  NEXT: retrain on this file. The champion artifact still carries the leak until then.")


if __name__ == "__main__":
    main()
