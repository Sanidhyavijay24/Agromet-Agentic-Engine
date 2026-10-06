"""
@file b1_reproduce_metrics.py
@description B1: re-score the saved champion on its OWN recorded train / val / test points and
             check the stored metrics are reproducible. No retraining.
@module scripts/audit

WHAT IT CHECKS
--------------
1. Reproducibility -- does the artifact score +60.6% RMSE gain on its recorded test points?
2. The honest metric -- temperature R2 (headlined as 0.9986) vs RESIDUAL R2, the share of the
   local deviation the model actually explains.
3. Overfitting -- the train-vs-test gap.
4. Source era (F21) -- the training set mixes ERA5-Land (2015-2016) and ECMWF IFS (2017+),
   while production is IFS. How does the model score on each era's test rows?

RUN
---
    python scripts/audit/b1_reproduce_metrics.py
    python scripts/audit/b1_reproduce_metrics.py --input data/raw/jaipur_10yr_training_data.parquet
Writes data/processed/audit/b1_reproduce_metrics.json.
"""

from __future__ import annotations

import argparse

from _common import fmt_pct, load_artifact, load_dataset, predict, score_frame, split_by_points, write_json

TOLERANCE_RMSE = 0.005  # C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    args = ap.parse_args()

    print("=" * 96)
    print("B1: REPRODUCE THE CHAMPION'S METRICS")
    print("=" * 96)
    payload = load_artifact()
    model, features = payload["model"], list(payload["features_list"])
    stored = payload.get("metrics", {})
    df, source_file = load_dataset(args.input)
    splits = split_by_points(df, payload)

    results = {}
    print(f"\n  {'split':<8}{'rows':>12}{'base RMSE':>11}{'down RMSE':>11}{'RMSE gain':>11}"
          f"{'temp R2':>10}{'resid R2':>10}")
    print("  " + "-" * 73)
    for name, part in splits.items():
        r = score_frame(part, predict(model, part, features))
        results[name] = r
        print(f"  {name:<8}{r['rows']:>12,}{r['baseline_rmse']:>11.3f}{r['downscaled_rmse']:>11.3f}"
              f"{fmt_pct(r['rmse_gain_pct']):>11}{r['temperature_r2']:>10.4f}{r['residual_r2']:>10.4f}")

    t = results["test"]
    print()
    print("  REPRODUCIBILITY against the artifact's stored test metrics:")
    checks = [("baseline_rmse", "baseline_rmse_c"), ("downscaled_rmse", "downscaled_rmse_c"),
              ("rmse_gain_pct", "rmse_improvement_pct"), ("residual_r2", "residual_r2")]
    reproduced = True
    for mine, theirs in checks:
        if theirs in stored:
            d = t[mine] - float(stored[theirs])
            ok = abs(d) < (TOLERANCE_RMSE if "rmse" in mine and "pct" not in mine else 0.5 if "pct" in mine else 0.005)
            reproduced &= ok
            print(f"    {mine:<18} stored {float(stored[theirs]):>9.4f}   now {t[mine]:>9.4f}   "
                  f"{'match' if ok else 'MISMATCH'}")
    if "1yr" in source_file:
        verdict = "NOT COMPARABLE (1-year file; the champion was trained on the 10-year file)"
    else:
        verdict = "REPRODUCED" if reproduced else "NOT REPRODUCED"
    print(f"    verdict: {verdict}")

    print()
    print("  HEADLINE METRIC CHECK:")
    print(f"    temperature R2 {t['temperature_r2']:.4f} -- mostly the diurnal/seasonal cycle the baseline")
    print(f"    already carries. Residual R2 {t['residual_r2']:.4f} is the share of the LOCAL deviation")
    print(f"    the model explains; report that one.")

    if "train" in results:
        gap = results["train"]["rmse_gain_pct"] - t["rmse_gain_pct"]
        print()
        print(f"  OVERFITTING: train gain {results['train']['rmse_gain_pct']:+.1f}% vs test "
              f"{t['rmse_gain_pct']:+.1f}% (gap {gap:.1f} points)")

    print()
    print("  BY SOURCE ERA on test points (production is served by ECMWF IFS):")
    era = {}
    test = splits["test"]
    for name, part in test.groupby("source_era", observed=True):
        r = score_frame(part, predict(model, part, features))
        era[name] = r
        print(f"    {name:<12} rows {r['rows']:>10,}  gain {fmt_pct(r['rmse_gain_pct'])}  "
              f"residual R2 {r['residual_r2']:.4f}  down RMSE {r['downscaled_rmse']:.3f}")
    if len(era) == 1:
        print("    (only one era present in this file)")

    path = write_json("b1_reproduce_metrics.json", {
        "source_file": source_file, "verdict": verdict, "splits": results,
        "stored_metrics": {k: stored.get(k) for k in ("baseline_rmse_c", "downscaled_rmse_c",
                                                       "rmse_improvement_pct", "residual_r2", "best_iteration")},
        "by_source_era_test": era,
    })
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
