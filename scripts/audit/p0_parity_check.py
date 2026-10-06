"""
@file p0_parity_check.py
@description PHASE 0: compare the distribution of every one of the 28 model features
             between the TRAINING data and what the LIVE inference path actually builds.
@module scripts/audit

WHY
---
The model's offline score and its live score differ (+60.6% claimed vs about +26%
measured). A feature that means one thing in training and another at serving time
explains that kind of gap, and it is invisible in any accuracy metric.

SEASONAL CONFOUND -- READ THIS BEFORE TRUSTING ANY OUTPUT
--------------------------------------------------------
The training set spans whole years. A live forecast spans a day or three. Comparing the
two directly makes every seasonal variable look broken: an annual mean temperature of
24.9 C against a late-September mean of 27.4 C is summer versus the yearly average, not
train/serve skew. The first version of this script did exactly that and reported 7 MAJOR
shifts, nearly all of them artifacts.

So the comparison is restricted to training rows from the SAME TIME OF YEAR as the live
window (+/- --doy-window days, wrapping at year end). Both the raw and the seasonally
matched PSI are printed, and only the matched one carries a verdict.

Calendar features (doy_sin/cos, hour_sin/cos) are reported separately and never given a
verdict: a 48-hour window holds 2 day-of-year values against 366 in training, so any
divergence measure on them is meaningless by construction, not evidence of a bug.

ZERO-INFLATED VARIABLES (rainfall) are compared on the share of dry hours plus the
distribution of wet hours, because PSI on raw values collapses when most values are 0.

thermal_inertia_lag_3h is computed along each live series exactly as the forecast route
does it (compute_thermal_inertia_lag), then passed in as a covariate.

WEATHER OR SOURCE SKEW: for every shifted feature, the live mean is compared with the range
of per-year seasonal means in training. Inside that range = ordinary weather variation;
outside every year = the two sources measure the variable differently. Needs 3+ years.

RUN
---
    python scripts/audit/p0_parity_check.py --anchors
    python scripts/audit/p0_parity_check.py --hours 72 --doy-window 14
    python scripts/audit/p0_parity_check.py --input data/raw/jaipur_10yr_training_data.parquet

Writes: data/processed/audit/p0_parity_report.json
Read-only: touches no model, no parquet, no cache.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUT_DIR = ROOT_DIR / "data" / "processed" / "audit"

TRAIN_BOX = {"north": 27.2, "south": 26.5, "west": 75.5, "east": 76.2, "step": 0.14}

LIVE_HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,"
    "direct_normal_irradiance,shortwave_radiation_instant,soil_temperature_0_to_7cm,"
    "soil_moisture_0_to_7cm,et0_fao_evapotranspiration"
)

# Features that encode position in the calendar or the day. A short live window cannot
# cover their training range, so a divergence score on them says nothing about skew.
CALENDAR_FEATURES = {"doy_sin", "doy_cos", "hour_sin", "hour_cos"}


def build_grid() -> list[tuple[float, float]]:
    lats = np.arange(TRAIN_BOX["south"], TRAIN_BOX["north"] + TRAIN_BOX["step"] / 2, TRAIN_BOX["step"])
    lons = np.arange(TRAIN_BOX["west"], TRAIN_BOX["east"] + TRAIN_BOX["step"] / 2, TRAIN_BOX["step"])
    return [(round(float(a), 4), round(float(o), 4)) for a in lats for o in lons]


def psi(train: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index. Rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 major.
    Computed on quantile bins of the training distribution.
    """
    train = train[np.isfinite(train)]
    live = live[np.isfinite(live)]
    if train.size < 10 or live.size < 10:
        return float("nan")
    edges = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0 if np.allclose(np.mean(train), np.mean(live)) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    t_frac = np.histogram(train, bins=edges)[0] / train.size
    l_frac = np.histogram(live, bins=edges)[0] / live.size
    eps = 1e-6
    return float(np.sum((np.clip(l_frac, eps, None) - np.clip(t_frac, eps, None))
                        * np.log(np.clip(l_frac, eps, None) / np.clip(t_frac, eps, None))))


# A feature is zero-inflated when at least this share of training values is exactly 0.
ZERO_INFLATED_FRACTION = 0.5


def zero_inflated_compare(train: np.ndarray, live: np.ndarray) -> dict[str, Any]:
    """
    Compare a mostly-zero variable (rainfall) in two parts: how often it is zero, and how
    its non-zero values are distributed.

    PSI on the raw values is meaningless here: with over 90% zeros, the training quantile
    bins collapse and PSI returns inf even when the means agree (0.048 vs 0.053 mm in the
    first season-matched run). That was an artifact of the metric, not a finding.
    """
    train = train[np.isfinite(train)]
    live = live[np.isfinite(live)]
    zf_train = float(np.mean(train == 0.0)) if train.size else float("nan")
    zf_live = float(np.mean(live == 0.0)) if live.size else float("nan")
    pos_train, pos_live = train[train > 0], live[live > 0]
    bins = 5
    psi_pos = (psi(pos_train, pos_live, bins=bins)
               if pos_train.size >= 20 and pos_live.size >= 20 else None)
    # PSI has a sampling floor: two samples from the SAME distribution score about
    # (bins-1) * (1/n1 + 1/n2) on average. Two days hold only ~100-200 wet hours, which puts
    # that floor near the usual 0.10 threshold, so a fixed threshold flags pure noise. Scale
    # the thresholds by the floor: a shift must clearly exceed what chance alone produces.
    noise = (bins - 1) * (1.0 / max(pos_train.size, 1) + 1.0 / max(pos_live.size, 1))
    d_zero = abs(zf_live - zf_train)
    # Same idea for the dry-hour share: allow ~3 standard errors of a proportion.
    se_zero = float(np.sqrt(max(zf_train * (1 - zf_train), 1e-6) / max(live.size, 1)))
    major_pos = psi_pos is not None and psi_pos > max(0.25, 5 * noise)
    moderate_pos = psi_pos is not None and psi_pos > max(0.10, 3 * noise)
    if d_zero > max(0.15, 5 * se_zero) or major_pos:
        verdict = "MAJOR"
    elif d_zero > max(0.07, 3 * se_zero) or moderate_pos:
        verdict = "MODERATE"
    else:
        verdict = "ok"
    return {
        "zero_fraction_train": zf_train,
        "zero_fraction_live": zf_live,
        "zero_fraction_diff": d_zero,
        "psi_positive": psi_pos,
        "psi_noise_floor": noise,
        "positive_samples_live": int(pos_live.size),
        "verdict": verdict,
    }


def interannual_check(matched: pd.DataFrame, feature: str, live_mean: float) -> dict[str, Any]:
    """
    Is a seasonal shift weather or source skew?

    Compute the seasonal-window mean of `feature` separately for every training year. If the
    live mean falls inside the range of those yearly means, the shift is within ordinary
    year-to-year variation (weather). If it falls outside every year seen, the training source
    and the live source probably measure the variable differently (source skew).
    Needs at least 3 years; the 1-year parquet can only say "undetermined".
    """
    if "timestamp" not in matched.columns or feature not in matched.columns:
        return {"verdict": "undetermined", "reason": "no timestamp column", "years": 0}
    years = pd.DatetimeIndex(matched["timestamp"]).year
    yearly = matched[feature].groupby(years).mean().dropna()
    n = int(len(yearly))
    if n < 3:
        return {"verdict": "undetermined", "years": n,
                "reason": f"only {n} training year(s) in the seasonal window; needs the 10-year parquet"}
    lo, hi = float(yearly.min()), float(yearly.max())
    mu, sd = float(yearly.mean()), float(yearly.std(ddof=1))
    z = (live_mean - mu) / sd if sd > 0 else float("inf")
    inside = lo <= live_mean <= hi
    return {
        "verdict": "within interannual range (weather)" if inside else "outside every year seen (source skew)",
        "years": n, "yearly_min": lo, "yearly_max": hi, "yearly_mean": mu, "yearly_std": sd,
        "z_score": z,
    }


def production_lag_3h(series: np.ndarray) -> np.ndarray:
    """
    T(t) - T(t-3h) along an hourly series, exactly as routes/forecast.py
    compute_thermal_inertia_lag does it: before 3 hours of history exist it falls back to
    T(t) - T(first hour), and the first hour is 0.
    """
    out = np.zeros_like(series, dtype=float)
    for h in range(len(series)):
        if h >= 3:
            out[h] = series[h] - series[h - 3]
        elif h > 0:
            out[h] = series[h] - series[0]
    return out


def circular_doy_mask(doy_series: np.ndarray, target_doys: set[int], window: int) -> np.ndarray:
    """Rows within +/- window days of any live day-of-year, wrapping across the year boundary."""
    mask = np.zeros(doy_series.shape, dtype=bool)
    for target in target_doys:
        diff = np.abs(doy_series - target)
        circular = np.minimum(diff, 365.25 - diff)
        mask |= circular <= window
    return mask


def fetch_live_grid(hours: int) -> dict[str, Any]:
    """One batched Open-Meteo call for every training grid point."""
    import requests

    grid = build_grid()
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": ",".join(str(a) for a, _ in grid),
            "longitude": ",".join(str(o) for _, o in grid),
            "hourly": LIVE_HOURLY_VARS,
            "forecast_days": max(1, min(16, (hours + 23) // 24)),
            "timezone": "UTC",
            "wind_speed_unit": "kmh",
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        data = [data]
    return {"grid": grid, "data": data}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/jaipur_10yr_training_data.parquet")
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--doy-window", type=int, default=10,
                    help="Compare against training rows within +/- this many days of the live window")
    ap.add_argument("--anchors", action="store_true",
                    help="Also build vectors with the anchor-mean baseline (the proposed fix)")
    ap.add_argument("--dem-dir", default=None)
    args = ap.parse_args()

    from services.api.schemas import GeoLocation, HourlyForecastPoint
    from services.ml_downscaler.inference import build_inference_feature_vector, load_downscaler_model

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    in_path = ROOT_DIR / args.input if not Path(args.input).is_absolute() else Path(args.input)
    if not in_path.exists():
        alt = ROOT_DIR / "data" / "raw" / "jaipur_1yr_training_data.parquet"
        if not alt.exists():
            raise SystemExit(f"No training parquet at {in_path}")
        print(f"! {in_path.name} not found, using {alt.name}")
        print("  (champion was trained on the 10-year file; get it from Lead ML)")
        in_path = alt

    payload = load_downscaler_model()
    features = list(payload["features_list"])
    dom = float(payload["domain_mean_elevation_m"])

    print("=" * 104)
    print("PHASE 0: TRAIN / SERVE FEATURE PARITY")
    print("=" * 104)
    print(f"  training data : {in_path.name}")
    print(f"  model features: {len(features)}   domain_mean_elevation_m: {dom:.2f}")

    train_df = pd.read_parquet(in_path)
    print(f"  training rows : {len(train_df):,}")

    # Materialise the engineered features the parquet does not store. This is exactly what
    # happens at train time, so the comparison stays faithful.
    missing_before = [f for f in features if f not in train_df.columns]
    if missing_before:
        print(f"  computing {len(missing_before)} engineered feature(s) the parquet does not store,")
        print("  using the same ensure_interaction_features() the trainer calls")
        from services.ml_downscaler.train import ensure_interaction_features

        train_df = ensure_interaction_features(train_df)
    still_missing = [f for f in features if f not in train_df.columns]
    if still_missing:
        print(f"! {len(still_missing)} feature(s) still absent: {still_missing}")
    train_cols = [f for f in features if f in train_df.columns]

    # ── live side, through the real inference code path ───────────────────
    print(f"  fetching live forecast for 36 grid points, {args.hours}h ...")
    live = fetch_live_grid(args.hours)
    grid, data = live["grid"], live["data"]
    times = data[0]["hourly"]["time"][: args.hours]
    T = np.array([d["hourly"]["temperature_2m"] for d in data], dtype=float)[:, : args.hours]
    elev_om = np.array([d.get("elevation", np.nan) for d in data], dtype=float)
    anchor_mean = T.mean(axis=0)

    live_dts = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in times]
    live_doys = {dt.timetuple().tm_yday for dt in live_dts}
    print(f"  live window   : {times[0]} .. {times[-1]}  (day-of-year {sorted(live_doys)})")

    rows_live: list[pd.Series] = []
    rows_live_anchor: list[pd.Series] = []
    # thermal_inertia_lag_3h is computed by the forecast route and passed in as a covariate;
    # the first version of this script omitted it, so the feature was a constant 0.
    anchor_lag = production_lag_3h(anchor_mean)
    for i, (la, lo) in enumerate(grid):
        point_lag = production_lag_3h(T[i])
        h_src = data[i]["hourly"]
        loc = GeoLocation(latitude=la, longitude=lo,
                          elevation_m=float(elev_om[i]) if np.isfinite(elev_om[i]) else None,
                          district="Jaipur", panchayat=None)
        for h in range(len(times)):
            def mk(temp: float) -> HourlyForecastPoint:
                return HourlyForecastPoint(
                    time=live_dts[h],
                    temperature_2m_c=temp,
                    relative_humidity_2m_pct=h_src["relative_humidity_2m"][h],
                    precipitation_mm=h_src["precipitation"][h],
                    surface_pressure_hpa=h_src["surface_pressure"][h],
                    wind_speed_10m_kmh=h_src["wind_speed_10m"][h],
                    # mirrors open_meteo_client.py after the P0.3 DNI parity fix
                    solar_radiation_w_m2=h_src["direct_normal_irradiance"][h],
                    shortwave_radiation_w_m2=h_src["shortwave_radiation_instant"][h],
                    soil_temperature_0_to_7cm_c=h_src["soil_temperature_0_to_7cm"][h],
                    soil_moisture_0_to_7cm_m3m3=h_src["soil_moisture_0_to_7cm"][h],
                    et0_evapotranspiration_mm=h_src["et0_fao_evapotranspiration"][h],
                )

            rows_live.append(build_inference_feature_vector(
                loc, mk(float(T[i, h])), dom,
                covariates={"thermal_inertia_lag_3h": float(point_lag[h])}).iloc[0])
            if args.anchors:
                rows_live_anchor.append(build_inference_feature_vector(
                    loc, mk(float(anchor_mean[h])), dom,
                    covariates={"thermal_inertia_lag_3h": float(anchor_lag[h])}).iloc[0])

    live_df = pd.DataFrame(rows_live)
    anchor_df = pd.DataFrame(rows_live_anchor) if args.anchors else None
    print(f"  live vectors  : {len(live_df):,}")

    # ── seasonal matching ────────────────────────────────────────────────
    if "timestamp" in train_df.columns:
        train_doy = pd.DatetimeIndex(train_df["timestamp"]).dayofyear.to_numpy()
    elif "day_of_year" in train_df.columns:
        train_doy = train_df["day_of_year"].to_numpy()
    else:
        train_doy = None

    if train_doy is not None:
        season_mask = circular_doy_mask(train_doy.astype(float), live_doys, args.doy_window)
        matched_df = train_df.loc[season_mask]
        print(f"  seasonal match: {len(matched_df):,} training rows within "
              f"+/-{args.doy_window} days of the live window "
              f"({100.0 * len(matched_df) / max(len(train_df), 1):.1f}% of training data)")
        if len(matched_df) < 500:
            print("! seasonal subset is small; widen --doy-window for a stabler comparison")
    else:
        matched_df = train_df
        print("! no timestamp column, cannot season-match. Treat every result as confounded.")

    print()

    # ── compare ──────────────────────────────────────────────────────────
    results: list[dict[str, Any]] = []
    for f in train_cols:
        raw = train_df[f].to_numpy(dtype=float)
        seasonal = matched_df[f].to_numpy(dtype=float)
        lv = live_df[f].to_numpy(dtype=float) if f in live_df.columns else np.array([])
        if lv.size == 0:
            results.append({"feature": f, "verdict": "MISSING_LIVE", "psi_seasonal": float("nan")})
            continue

        is_calendar = f in CALENDAR_FEATURES
        entry: dict[str, Any] = {
            "feature": f,
            "calendar_feature": is_calendar,
            "train_mean_annual": float(np.nanmean(raw)),
            "train_mean_seasonal": float(np.nanmean(seasonal)) if seasonal.size else None,
            "live_mean": float(np.nanmean(lv)),
            "train_std_seasonal": float(np.nanstd(seasonal)) if seasonal.size else None,
            "live_std": float(np.nanstd(lv)),
            "train_min_annual": float(np.nanmin(raw)),
            "train_max_annual": float(np.nanmax(raw)),
            "live_min": float(np.nanmin(lv)),
            "live_max": float(np.nanmax(lv)),
            "train_distinct": int(len(set(np.round(raw[np.isfinite(raw)], 6)))),
            "live_distinct": int(len(set(np.round(lv[np.isfinite(lv)], 6)))),
            "psi_raw": psi(raw, lv),
            "psi_seasonal": psi(seasonal, lv) if seasonal.size else float("nan"),
        }
        # Out-of-range is judged against the FULL training range: anything outside it is
        # territory the trees never split on, whatever the season.
        entry["live_out_of_train_range_pct"] = float(
            100.0 * np.mean((lv < entry["train_min_annual"]) | (lv > entry["train_max_annual"]))
        )

        p = entry["psi_seasonal"]
        zero_share = float(np.mean(raw[np.isfinite(raw)] == 0.0)) if raw.size else 0.0
        entry["metric"] = "psi"
        if not is_calendar and zero_share >= ZERO_INFLATED_FRACTION:
            zi = zero_inflated_compare(seasonal if seasonal.size else raw, lv)
            entry.update({f"zi_{k}": v for k, v in zi.items() if k != "verdict"})
            entry["metric"] = "zero-inflated"
            entry["verdict"] = zi["verdict"]
        elif is_calendar:
            entry["verdict"] = "calendar"
        elif not np.isfinite(p):
            entry["verdict"] = "MAJOR"
            entry["metric"] = "psi (degenerate training distribution)"
        elif p > 0.25:
            entry["verdict"] = "MAJOR"
        elif p > 0.10:
            entry["verdict"] = "MODERATE"
        else:
            entry["verdict"] = "ok"

        if args.anchors and anchor_df is not None and f in anchor_df.columns:
            av = anchor_df[f].to_numpy(dtype=float)
            entry["psi_anchor_seasonal"] = psi(seasonal, av) if seasonal.size else float("nan")
            entry["anchor_mean"] = float(np.nanmean(av))
        results.append(entry)

    order = {"MAJOR": 0, "MODERATE": 1, "MISSING_LIVE": 2, "ok": 3, "calendar": 4}
    results.sort(key=lambda r: (order.get(r["verdict"], 9), -(r.get("psi_seasonal") or 0)))

    hdr = (f"  {'feature':<32}{'verdict':<10}{'PSI seas':>9}{'PSI raw':>9}"
           f"{'train mean':>12}{'live mean':>11}{'train sd':>10}{'live sd':>9}{'oor%':>7}")
    if args.anchors:
        hdr += f"{'PSI anchor':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        if r["verdict"] == "MISSING_LIVE":
            print(f"  {r['feature']:<32}{'MISSING':<10}")
            continue
        tm = r["train_mean_seasonal"]
        ts = r["train_std_seasonal"]
        if r.get("metric") == "zero-inflated":
            print(f"  {r['feature']:<32}{r['verdict']:<10}"
                  f"  dry hours: train {100 * r['zi_zero_fraction_train']:.0f}% vs live "
                  f"{100 * r['zi_zero_fraction_live']:.0f}%;  wet-hour PSI "
                  f"{'n/a (too few wet hours)' if r['zi_psi_positive'] is None else format(r['zi_psi_positive'], '.3f')};"
                  f"  means {tm:.3f} vs {r['live_mean']:.3f}")
            continue
        line = (f"  {r['feature']:<32}{r['verdict']:<10}"
                f"{r['psi_seasonal']:>9.3f}{r['psi_raw']:>9.3f}"
                f"{(tm if tm is not None else float('nan')):>12.3f}{r['live_mean']:>11.3f}"
                f"{(ts if ts is not None else float('nan')):>10.3f}{r['live_std']:>9.3f}"
                f"{r['live_out_of_train_range_pct']:>6.1f}%")
        if args.anchors and "psi_anchor_seasonal" in r:
            line += f"{r['psi_anchor_seasonal']:>12.3f}"
        print(line)

    print()
    print("  'train mean/sd' are the SEASONALLY MATCHED training rows. 'PSI raw' is the")
    print("  whole-year comparison, shown only so the seasonal confound stays visible.")
    print("  Calendar features carry no verdict -- see the note at the top of this file.")

    major = [r for r in results if r["verdict"] == "MAJOR"]
    moderate = [r for r in results if r["verdict"] == "MODERATE"]
    print()
    print("  " + "-" * 100)
    print(f"  {len(major)} MAJOR, {len(moderate)} MODERATE, "
          f"{sum(1 for r in results if r['verdict'] == 'ok')} ok, "
          f"{sum(1 for r in results if r['verdict'] == 'calendar')} calendar (not judged)")

    if major:
        print()
        print("  MAJOR -- the model sees these differently in production than it was fitted on,")
        print("  after controlling for season:")
        for r in major:
            print(f"    - {r['feature']}: train {r['train_mean_seasonal']:+.3f} "
                  f"(sd {r['train_std_seasonal']:.3f}) vs live {r['live_mean']:+.3f} "
                  f"(sd {r['live_std']:.3f})   PSI {r['psi_seasonal']:.3f}")

    shifted = [r for r in results if r["verdict"] in ("MAJOR", "MODERATE")
               and not r.get("calendar_feature")]
    if shifted:
        print()
        print("  WEATHER OR SOURCE SKEW? live mean vs the spread of yearly seasonal means in training:")
        for r in shifted:
            ia = interannual_check(matched_df, r["feature"], r["live_mean"])
            r["interannual"] = ia
            if ia["verdict"] == "undetermined":
                print(f"    - {r['feature']:<30} undetermined: {ia['reason']}")
            else:
                print(f"    - {r['feature']:<30} {ia['verdict']}: live {r['live_mean']:.3f}, "
                      f"yearly range [{ia['yearly_min']:.3f}, {ia['yearly_max']:.3f}] over {ia['years']} years, "
                      f"z = {ia['z_score']:+.1f}")

    oor = [r for r in results
           if r.get("live_out_of_train_range_pct", 0) > 1.0 and not r.get("calendar_feature")]
    if oor:
        print()
        print("  Live values falling OUTSIDE the training range (the trees never split there):")
        for r in sorted(oor, key=lambda x: -x["live_out_of_train_range_pct"]):
            print(f"    - {r['feature']}: {r['live_out_of_train_range_pct']:.1f}% of live values, "
                  f"live [{r['live_min']:.3f}, {r['live_max']:.3f}] "
                  f"vs training [{r['train_min_annual']:.3f}, {r['train_max_annual']:.3f}]")

    flat = [r for r in results
            if not r.get("calendar_feature")
            and r.get("live_distinct", 99) <= 2 < r.get("train_distinct", 0)]
    if flat:
        print()
        print("  Effectively CONSTANT in production but varied in training:")
        for r in flat:
            print(f"    - {r['feature']}: {r['live_distinct']} distinct live value(s) "
                  f"vs {r['train_distinct']} in training")

    if args.anchors:
        comparable = [r for r in results
                      if "psi_anchor_seasonal" in r and np.isfinite(r.get("psi_seasonal", np.nan))
                      and np.isfinite(r["psi_anchor_seasonal"]) and not r.get("calendar_feature")]
        better = [r for r in comparable if r["psi_anchor_seasonal"] < r["psi_seasonal"] - 0.02]
        worse = [r for r in comparable if r["psi_anchor_seasonal"] > r["psi_seasonal"] + 0.02]
        print()
        print(f"  ANCHOR-MEAN BASELINE: better parity on {len(better)}, worse on {len(worse)}, "
              f"unchanged on {len(comparable) - len(better) - len(worse)} of {len(comparable)} features")
        for r in sorted(better, key=lambda x: x["psi_seasonal"] - x["psi_anchor_seasonal"], reverse=True)[:8]:
            print(f"    better  {r['feature']}: {r['psi_seasonal']:.3f} -> {r['psi_anchor_seasonal']:.3f}")
        for r in sorted(worse, key=lambda x: x["psi_anchor_seasonal"] - x["psi_seasonal"], reverse=True)[:8]:
            print(f"    worse   {r['feature']}: {r['psi_seasonal']:.3f} -> {r['psi_anchor_seasonal']:.3f}")
        print()
        print("  NOTE: PSI measures distribution shape, not correctness. The anchor mean is an")
        print("  average over 36 points, so it is narrower than any single point's series and can")
        print("  score worse on PSI while still being the RIGHT input -- it is what the residual")
        print("  was trained against. Accuracy is settled by the anchor RMSE experiment, not here.")

    out = OUT_DIR / "p0_parity_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "training_file": str(in_path),
            "training_rows": int(len(train_df)),
            "seasonal_rows": int(len(matched_df)),
            "doy_window_days": args.doy_window,
            "live_window": {"start": times[0], "end": times[-1], "doys": sorted(live_doys)},
            "live_vectors": int(len(live_df)),
            "domain_mean_elevation_m": dom,
            "anchor_baseline_tested": bool(args.anchors),
            "features": results,
        }, f, indent=2, default=str)
    print()
    print("=" * 104)
    print(f"  wrote {out.relative_to(ROOT_DIR)}")
    print("=" * 104)


if __name__ == "__main__":
    main()
