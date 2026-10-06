"""
@file b2_eval_breakdown.py
@description B2a: where does the champion do well or badly, how does it compare with naive
             baselines, and which features does it really rely on? Uses the saved model only.
@module scripts/audit

1. HEADLINE    residual R2 (share of the local deviation explained), not temperature R2.
2. BREAKDOWNS  by IST hour, day/night, IMD season, month, test point, elevation band,
               source era, and on the cases that matter most: large local deviations and
               temperature extremes.
3. BASELINES   all fitted on TRAINING points only and scored on the test points:
                 coarse       residual = 0 (the baseline itself)
                 lapse        standard -6.5 C/km lapse rate
                 elev-linear  residual as a linear function of elevation difference
                 lookup       mean residual by IST hour x month x elevation band -- no ML at all
               plus, clearly marked as a DIFFERENT setting, a per-point lookup built from the
               test points' own earlier years. Registry panchayats have no training history, so
               that setting does not apply to them; it is reported only as a reference.
4. PERMUTATION importance on the test set (how much RMSE rises when a feature is shuffled),
   next to the gain-based importance stored in the artifact.

RUN
---
    python scripts/audit/b2_eval_breakdown.py
    python scripts/audit/b2_eval_breakdown.py --perm-rows 100000
Writes data/processed/audit/b2_eval_breakdown.json.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from _common import (fmt_pct, load_artifact, load_dataset, predict, score, score_frame,
                     split_by_points, write_json)

ELEV_BANDS = [-np.inf, -50, -15, 15, 50, np.inf]
ELEV_LABELS = ["valley (<-50 m)", "low (-50..-15)", "level (-15..15)", "high (15..50)", "ridge (>50 m)"]


def elevation_band(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df["delta_elevation_m"], ELEV_BANDS, labels=ELEV_LABELS)


def breakdown(test: pd.DataFrame, pred: np.ndarray, key: pd.Series | str, label: str,
              show_r2: bool = True) -> list[dict[str, Any]]:
    """Scores per group. Residual R2 is hidden where it misleads (single points): there the
    residual is mostly a constant offset, so bias is the informative number."""
    groups = test[key] if isinstance(key, str) else key
    rows = []
    for g, idx in test.groupby(groups, observed=True).indices.items():
        part = test.iloc[idx]
        r = score_frame(part, pred[idx])
        rows.append({"group": str(g), **r})
    print(f"\n  BY {label.upper()}")
    print(f"    {'group':<22}{'rows':>10}{'base RMSE':>11}{'down RMSE':>11}{'gain':>9}{'bias':>9}"
          f"{'resid R2' if show_r2 else '':>10}")
    for r in rows:
        r2 = f"{r['residual_r2']:>10.3f}" if show_r2 else ""
        print(f"    {r['group']:<22}{r['rows']:>10,}{r['baseline_rmse']:>11.3f}{r['downscaled_rmse']:>11.3f}"
              f"{fmt_pct(r['rmse_gain_pct']):>9}{r['bias']:>+9.3f}{r2}")
    return rows


def lookup_baseline(train: pd.DataFrame, test: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Mean training residual per key combination, with a coarser fallback where unseen."""
    table = train.groupby(keys, observed=True)["residual_anomaly_c"].mean()
    fallback = train.groupby(["ist_hour", "month"])["residual_anomaly_c"].mean()
    idx = pd.MultiIndex.from_frame(test[keys])
    vals = table.reindex(idx).to_numpy(dtype=float, copy=True)  # pandas 3 returns a read-only view
    miss = np.isnan(vals)
    if miss.any():
        fb = fallback.reindex(pd.MultiIndex.from_frame(test.loc[miss, ["ist_hour", "month"]])).to_numpy()
        vals[miss] = np.nan_to_num(fb, nan=float(train["residual_anomaly_c"].mean()))
    return vals


def baselines(train: pd.DataFrame, test: pd.DataFrame, pred: np.ndarray) -> dict[str, dict[str, float]]:
    train = train.assign(elev_band=elevation_band(train))
    test = test.assign(elev_band=elevation_band(test))
    slope, intercept = np.polyfit(train["delta_elevation_m"], train["residual_anomaly_c"], 1)
    candidates = {
        "coarse (residual = 0)": np.zeros(len(test)),
        "standard lapse rate": test["theoretical_lapse_delta_c"].to_numpy(),
        "elevation-linear fit": intercept + slope * test["delta_elevation_m"].to_numpy(),
        "hour x month x elevation lookup": lookup_baseline(train, test, ["ist_hour", "month", "elev_band"]),
        "CHAMPION MODEL": pred,
    }
    out = {name: score_frame(test, p) for name, p in candidates.items()}

    print("\n  NAIVE BASELINES (fitted on training points, scored on test points)")
    print(f"    {'method':<34}{'down RMSE':>11}{'gain':>9}{'resid R2':>10}")
    for name, r in out.items():
        print(f"    {name:<34}{r['downscaled_rmse']:>11.3f}{fmt_pct(r['rmse_gain_pct']):>9}{r['residual_r2']:>10.3f}")
    lk, model = out["hour x month x elevation lookup"], out["CHAMPION MODEL"]
    if model["rmse_gain_pct"] > 0:
        share = 100 * lk["rmse_gain_pct"] / model["rmse_gain_pct"]
        print(f"    -> a no-ML lookup table achieves {share:.0f}% of the champion's gain")
        out["_lookup_share_of_model_gain_pct"] = {"value": share}  # type: ignore[assignment]
    return out


def temporal_reference(test: pd.DataFrame) -> dict[str, Any] | None:
    """Per-point x hour x month lookup from the test points' own EARLIER years, scored on
    their last year. A different setting: it presumes history at the site."""
    years = sorted(test["year"].unique())
    if len(years) < 3:
        return None
    last = years[-1]
    hist, future = test[test["year"] < last], test[test["year"] == last]
    p = lookup_baseline(hist, future, ["point_id", "ist_hour", "month"])
    r = score_frame(future, p)
    print(f"\n  REFERENCE, DIFFERENT SETTING -- sites WITH history: per-point lookup from "
          f"{years[0]}-{last - 1}, scored on {last}")
    print(f"    gain {fmt_pct(r['rmse_gain_pct'])}, residual R2 {r['residual_r2']:.3f}. Not comparable "
          f"to the spatial holdout: registry panchayats have no training history.")
    return {"held_out_year": int(last), **r}


def permutation_importance(model: Any, test: pd.DataFrame, features: list[str], rows: int,
                           gain_importance: dict[str, float], seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    sample = test.sample(n=min(rows, len(test)), random_state=seed) if len(test) > rows else test
    base = score_frame(sample, predict(model, sample, features))["downscaled_rmse"]
    out = []
    for f in features:
        # Shuffle the MODEL INPUT only. baseline_temp_c is both a feature and part of the
        # target (downscaled = baseline + residual); shuffling it in the frame used for scoring
        # would corrupt the target itself and report a meaningless +9 C "importance".
        x = sample[features].copy()
        x[f] = rng.permutation(x[f].to_numpy())
        rmse = score_frame(sample, np.asarray(model.predict(x), float))["downscaled_rmse"]
        out.append({"feature": f, "rmse_increase": rmse - base, "gain_importance": gain_importance.get(f)})
    out.sort(key=lambda r: -r["rmse_increase"])
    print(f"\n  PERMUTATION IMPORTANCE on {len(sample):,} test rows (RMSE rise when shuffled)")
    print(f"    {'feature':<32}{'RMSE rise':>11}{'gain imp.':>11}")
    for r in out[:15]:
        gi = r["gain_importance"]
        print(f"    {r['feature']:<32}{r['rmse_increase']:>+11.4f}{(f'{100 * gi:.1f}%' if gi is not None else '-'):>11}")
    useless = [r["feature"] for r in out if r["rmse_increase"] <= 0.0005]
    if useless:
        print(f"    contribute ~nothing when shuffled: {useless}")
    return out


def run(df: pd.DataFrame, payload: dict[str, Any], perm_rows: int) -> dict[str, Any]:
    model, features = payload["model"], list(payload["features_list"])
    splits = split_by_points(df, payload)
    train, test = splits["train"], splits["test"]
    pred = predict(model, test, features)
    head = score_frame(test, pred)

    print(f"\n  HEADLINE on {head['rows']:,} test rows: RMSE gain {fmt_pct(head['rmse_gain_pct'])}, "
          f"temperature R2 {head['temperature_r2']:.4f}, RESIDUAL R2 {head['residual_r2']:.4f}")

    true_res = (test["temperature_2m_c"] - test["baseline_temp_c"]).to_numpy()
    big = np.abs(true_res) >= np.quantile(np.abs(true_res), 0.95)
    hot = test["temperature_2m_c"].to_numpy() >= np.quantile(test["temperature_2m_c"], 0.95)
    cold = test["temperature_2m_c"].to_numpy() <= np.quantile(test["temperature_2m_c"], 0.05)
    cases = {}
    print("\n  THE CASES THAT MATTER")
    for name, mask in [("largest 5% local deviations", big), ("hottest 5% hours", hot), ("coldest 5% hours", cold)]:
        r = score(test["temperature_2m_c"].to_numpy()[mask], test["baseline_temp_c"].to_numpy()[mask], pred[mask])
        cases[name] = r
        print(f"    {name:<30} rows {r['rows']:>9,}  base {r['baseline_rmse']:.3f}  down {r['downscaled_rmse']:.3f}"
              f"  gain {fmt_pct(r['rmse_gain_pct'])}")

    pts = test.groupby("point_id")["elevation_m"].first()
    point_key = test["point_id"].map(lambda p: f"{p} ({pts[p]:.0f} m)")
    out = {
        "headline": head,
        "cases": cases,
        "by_hour": breakdown(test, pred, "ist_hour", "IST hour"),
        "by_daytime": breakdown(test, pred, test["daytime"].map({True: "day 07-18 IST", False: "night"}), "day / night"),
        "by_season": breakdown(test, pred, "imd_season", "IMD season"),
        "by_point": breakdown(test, pred, point_key, "test point", show_r2=False),
        "by_elevation_band": breakdown(test, pred, elevation_band(test), "elevation band"),
        "by_source_era": breakdown(test, pred, "source_era", "source era"),
        "baselines": baselines(train, test, pred),
        "temporal_reference": temporal_reference(test),
        "permutation_importance": permutation_importance(
            model, test, features, perm_rows, payload.get("metrics", {}).get("feature_importances", {})),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--perm-rows", type=int, default=200_000)
    args = ap.parse_args()
    print("=" * 96)
    print("B2a: EVALUATION BREAKDOWN, BASELINES, PERMUTATION IMPORTANCE (no retraining)")
    print("=" * 96)
    payload = load_artifact()
    df, source_file = load_dataset(args.input)
    out = run(df, payload, args.perm_rows)
    out["source_file"] = source_file
    print(f"\n  wrote {write_json('b2_eval_breakdown.json', out)}")


if __name__ == "__main__":
    main()
