"""
@file b2_training_experiments.py
@description B2b: audit experiments that RETRAIN the model. Run by a person with the compute;
             nothing here touches the production artifact.
@module scripts/audit

EXPERIMENTS
-----------
learning_curve  The champion stopped at tree 1199 of a 1200 cap, so early stopping never fired
                (F8). Train with a much higher cap and patience; report where validation RMSE
                actually bottoms out and what the extra trees buy on the test points.
ablation        Retrain without each feature group -- elevation, terrain, soil, solar, lag --
                and measure the test-RMSE change. The terrain group matters most: at inference
                it is fabricated by a formula (F1, F17).
source_era      The training set mixes ERA5-Land (2015-2016) and ECMWF IFS (2017+); production
                is IFS (F21). Train on all years vs IFS years only; score both on IFS-era test
                rows. If IFS-only is as good or better, drop the ERA5-Land years.
spatial_buffer  Test points sit 15 km from training points inside one small box. Buffered
                leave-one-point-out: hold out one point at a time, and remove the training points
                within a buffer of it. A falling score as the buffer grows means spatial
                autocorrelation was flattering the headline.
seeds           The spatial split is one random draw. Repeat it with 5 seeds; report the spread.

RUN
---
    python scripts/audit/b2_training_experiments.py --gpu                       # everything
    python scripts/audit/b2_training_experiments.py --gpu --only learning_curve source_era
    python scripts/audit/b2_training_experiments.py --sample-frac 0.25          # CPU, faster

Every experiment retrains; on CPU the full 10-year set is slow. --sample-frac subsamples the
TRAINING rows only (test rows are never subsampled), which keeps comparisons fair.
Writes data/processed/audit/b2_training_experiments.json.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from _common import (ELEVATION_GROUP, IFS_ERA_START_YEAR, SOIL_GROUP, SOLAR_GROUP, TERRAIN_GROUP,
                     fmt_pct, load_artifact, load_dataset, predict, score_frame, split_by_points,
                     train_xgb, write_json)

ALL = ["learning_curve", "ablation", "source_era", "spatial_buffer", "seeds"]


def _subsample(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    return df if frac >= 1.0 else df.sample(frac=frac, random_state=seed)


def learning_curve(splits: dict[str, pd.DataFrame], features: list[str], a: argparse.Namespace) -> dict[str, Any]:
    print("\n  LEARNING CURVE: cap raised to", a.max_trees, "trees, patience", a.patience)
    train = _subsample(splits["train"], a.sample_frac, a.seed)
    model, info = train_xgb(train, splits["val"], features, n_estimators=a.max_trees,
                            early_stopping_rounds=a.patience, gpu=a.gpu, seed=a.seed)
    curve = info["val_rmse_curve"]
    test = splits["test"]
    at_cap = min(a.champion_trees, len(curve)) - 1
    best = info["best_iteration"]
    full = score_frame(test, np.asarray(model.predict(test[features], iteration_range=(0, best + 1)), float))
    capped = score_frame(test, np.asarray(model.predict(test[features], iteration_range=(0, at_cap + 1)), float))
    print(f"    best validation iteration: {best}  ({'still hitting the cap' if info['hit_the_cap'] else 'converged'})")
    if curve:
        print(f"    validation RMSE at tree {at_cap + 1}: {curve[at_cap]:.4f}   at best ({best + 1}): {curve[best]:.4f}")
    print(f"    test gain with {at_cap + 1} trees: {fmt_pct(capped['rmse_gain_pct'])}   "
          f"with {best + 1} trees: {fmt_pct(full['rmse_gain_pct'])}   "
          f"(+{full['rmse_gain_pct'] - capped['rmse_gain_pct']:.2f} points)")
    return {"best_iteration": best, "hit_the_cap": info["hit_the_cap"], "seconds": info["seconds"],
            "test_at_champion_cap": capped, "test_at_best": full,
            "val_rmse_curve_every_50": curve[::50]}


def ablation(splits: dict[str, pd.DataFrame], features: list[str], a: argparse.Namespace) -> dict[str, Any]:
    groups = {"full model": [], "- elevation": ELEVATION_GROUP, "- terrain": TERRAIN_GROUP,
              "- soil": SOIL_GROUP, "- solar": SOLAR_GROUP, "- thermal lag": ["thermal_inertia_lag_3h"]}
    train = _subsample(splits["train"], a.sample_frac, a.seed)
    out, full_rmse = {}, None
    print("\n  ABLATION: retrain without each feature group (same split, same hyperparameters)")
    print(f"    {'variant':<16}{'features':>9}{'down RMSE':>11}{'gain':>9}{'resid R2':>10}{'vs full':>10}")
    for name, drop in groups.items():
        feats = [f for f in features if f not in drop]
        model, info = train_xgb(train, splits["val"], feats, n_estimators=a.n_estimators, gpu=a.gpu, seed=a.seed)
        r = score_frame(splits["test"], predict(model, splits["test"], feats))
        full_rmse = r["downscaled_rmse"] if name == "full model" else full_rmse
        delta = r["downscaled_rmse"] - full_rmse
        out[name] = {**r, "features": len(feats), "dropped": drop, "rmse_vs_full": delta, "seconds": info["seconds"]}
        print(f"    {name:<16}{len(feats):>9}{r['downscaled_rmse']:>11.3f}{fmt_pct(r['rmse_gain_pct']):>9}"
              f"{r['residual_r2']:>10.3f}{delta:>+10.4f}")
    t = out.get("- terrain", {})
    if t and abs(t["rmse_vs_full"]) < 0.005:
        print("    -> removing terrain barely matters: the model learned little from the grid-neighbour terrain,")
        print("       so its formula-fabricated values at inference are unlikely to be doing much either.")
    return out


def source_era(df: pd.DataFrame, splits: dict[str, pd.DataFrame], features: list[str],
               a: argparse.Namespace) -> dict[str, Any] | None:
    years = splits["train"]["year"]
    if not (years < IFS_ERA_START_YEAR).any():
        print("\n  SOURCE ERA: no ERA5-Land years in this file; skipped (needs the 10-year parquet)")
        return None
    test_ifs = splits["test"][splits["test"]["year"] >= IFS_ERA_START_YEAR]
    print(f"\n  SOURCE ERA: all years vs IFS-only ({IFS_ERA_START_YEAR}+), scored on {len(test_ifs):,} IFS-era test rows")
    variants = {
        "all years (as shipped)": (splits["train"], splits["val"]),
        f"IFS only ({IFS_ERA_START_YEAR}+)": (splits["train"][years >= IFS_ERA_START_YEAR],
                                             splits["val"][splits["val"]["year"] >= IFS_ERA_START_YEAR]),
    }
    out = {}
    for name, (tr, va) in variants.items():
        model, info = train_xgb(_subsample(tr, a.sample_frac, a.seed), va, features,
                                n_estimators=a.n_estimators, gpu=a.gpu, seed=a.seed)
        r = score_frame(test_ifs, predict(model, test_ifs, features))
        out[name] = {**r, "train_rows": int(len(tr)), "seconds": info["seconds"]}
        print(f"    {name:<26} train rows {len(tr):>10,}  down RMSE {r['downscaled_rmse']:.4f}  "
              f"gain {fmt_pct(r['rmse_gain_pct'])}  resid R2 {r['residual_r2']:.3f}")
    names = list(out)
    if out[names[1]]["downscaled_rmse"] <= out[names[0]]["downscaled_rmse"] + 0.002:
        print("    -> IFS-only is at least as good on production-source data: drop the ERA5-Land years.")
    return out


def spatial_buffer(df: pd.DataFrame, features: list[str], a: argparse.Namespace) -> dict[str, Any]:
    """
    Buffered leave-one-point-out on a sample of points.

    Each sampled point is held out ALONE, and only the training points within the buffer of
    that one point are removed. Holding out a whole fold and buffering every member instead
    removes most of a 36-point grid and measures data starvation, not leakage (seen when this
    was first smoke-tested: ~20 of 24 training points removed, gain collapsing to ~0).
    """
    pts = df.groupby("point_id")[["latitude", "longitude"]].first()
    ids = np.array(sorted(pts.index))
    rng = np.random.RandomState(a.seed)
    held_out = rng.permutation(ids)[: min(a.buffer_points, len(ids))]
    print(f"\n  SPATIAL BUFFER: buffered leave-one-point-out on {len(held_out)} points, buffers {a.buffers} deg")
    print("    (grid spacing 0.14 deg: 0.15 removes a point's 4 nearest neighbours, 0.21 all 8)")
    out = {}
    for buf in a.buffers:
        gains, removed = [], []
        for k, pid in enumerate(held_out):
            here = pts.loc[pid].to_numpy()
            others = [p for p in ids if p != pid]
            far = [p for p in others if np.hypot(*(pts.loc[p].to_numpy() - here)) > buf] if buf > 0 else others
            removed.append(len(others) - len(far))
            inner = rng.permutation(far)
            n_val = max(1, len(inner) // 10)
            tr = _subsample(df[df["point_id"].isin(inner[n_val:])], a.sample_frac, a.seed + k)
            va = df[df["point_id"].isin(inner[:n_val])]
            te = df[df["point_id"] == pid]
            model, _ = train_xgb(tr, va, features, n_estimators=a.n_estimators, gpu=a.gpu, seed=a.seed + k)
            gains.append(score_frame(te, predict(model, te, features))["rmse_gain_pct"])
        out[str(buf)] = {"mean_gain_pct": float(np.mean(gains)), "std_gain_pct": float(np.std(gains)),
                         "points": [str(p) for p in held_out],
                         "mean_neighbours_removed": float(np.mean(removed))}
        g = out[str(buf)]
        print(f"    buffer {buf:>4} deg: gain {g['mean_gain_pct']:+.1f}% +/- {g['std_gain_pct']:.1f} "
              f"(~{g['mean_neighbours_removed']:.1f} neighbours removed per held-out point)")
    base, widest = out[str(a.buffers[0])]["mean_gain_pct"], out[str(a.buffers[-1])]["mean_gain_pct"]
    print(f"    -> gain falls {base - widest:.1f} points when a point's neighbours are excluded"
          + ("; the headline leans on spatial autocorrelation" if base - widest > 5 else ""))
    return out


def seeds(df: pd.DataFrame, features: list[str], a: argparse.Namespace) -> dict[str, Any]:
    from services.ml_downscaler.train import spatial_three_way_split
    print(f"\n  SPLIT VARIANCE: {len(a.seed_list)} random spatial splits")
    gains = []
    for s in a.seed_list:
        tr, va, te, *_ = spatial_three_way_split(df, seed=s)
        model, _ = train_xgb(_subsample(tr, a.sample_frac, s), va, features,
                             n_estimators=a.n_estimators, gpu=a.gpu, seed=s)
        g = score_frame(te, predict(model, te, features))["rmse_gain_pct"]
        gains.append(g)
        print(f"    seed {s:>3}: test gain {g:+.1f}%")
    print(f"    -> {np.mean(gains):+.1f}% +/- {np.std(gains):.1f} (range {min(gains):+.1f} to {max(gains):+.1f})")
    return {"seeds": a.seed_list, "gains_pct": gains, "mean": float(np.mean(gains)), "std": float(np.std(gains))}


def run(df: pd.DataFrame, payload: dict[str, Any], a: argparse.Namespace) -> dict[str, Any]:
    features = list(payload["features_list"])
    splits = split_by_points(df, payload)
    a.champion_trees = int(getattr(payload.get("model"), "n_estimators", 1200) or 1200)
    out: dict[str, Any] = {"settings": {k: v for k, v in vars(a).items() if k != "input"}}
    todo = a.only or ALL
    if "learning_curve" in todo:
        out["learning_curve"] = learning_curve(splits, features, a)
    if "ablation" in todo:
        out["ablation"] = ablation(splits, features, a)
    if "source_era" in todo:
        out["source_era"] = source_era(df, splits, features, a)
    if "spatial_buffer" in todo:
        out["spatial_buffer"] = spatial_buffer(df, features, a)
    if "seeds" in todo:
        out["seeds"] = seeds(df, features, a)
    return out


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--only", nargs="*", choices=ALL)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=1200)
    ap.add_argument("--max-trees", type=int, default=4000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--buffer-points", type=int, default=8,
                    help="How many points to hold out one at a time in the buffer experiment")
    ap.add_argument("--buffers", type=float, nargs="*", default=[0.0, 0.15, 0.21])
    ap.add_argument("--seed-list", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    return ap.parse_args(argv)


def main() -> None:
    a = parse()
    print("=" * 96)
    print("B2b: TRAINING EXPERIMENTS (retrains; production artifact untouched)")
    print("=" * 96)
    payload = load_artifact()
    df, source_file = load_dataset(a.input)
    print(f"  device: {'GPU' if a.gpu else 'CPU'}; training-row sample fraction {a.sample_frac}")
    out = run(df, payload, a)
    out["source_file"] = source_file
    print(f"\n  wrote {write_json('b2_training_experiments.json', out)}")


if __name__ == "__main__":
    main()
