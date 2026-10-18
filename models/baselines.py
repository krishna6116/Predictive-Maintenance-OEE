"""Phase 4: baseline models on NASA's official FD001 test set.

Run:  python -m models.baselines        (about 10-15 minutes on a laptop CPU)

Protocol (the test set is used once, at the end):
  1. Choose settings on the 20 validation engines only (Ridge alpha, XGBoost depth and tree count).
  2. Refit each model on all 100 training engines with those settings.
  3. Score each model once on the 100 test engines, at each engine's last observed cycle.
XGBoost is trained with 5 random seeds to show how much results move by chance.
Outputs: models/results_phase4.json, models/predictions_phase4.csv, models/importance_phase4.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from features.build import RUL_CAP, feature_columns
from models.data import test_frame, train_frame
from models.metrics import bootstrap_ci, mae, paired_bootstrap, phm_score, rmse
from pipeline.config import SENSORS

OUT = Path("models")
SEEDS = [0, 1, 2, 3, 4]
ALL = feature_columns()
FEATURE_SETS = {
    "all 168 features": ALL,
    "no FFT (154)": [c for c in ALL if not c.endswith("_lowfreq")],
    "rolling only, no EMA/FFT (140)": [c for c in ALL if not c.endswith(("_lowfreq", "_ema"))],
    "without s9 and s14 (144)": [c for c in ALL if not (c in ("s9", "s14") or c.startswith(("s9_", "s14_")))],
    "raw sensors only (14)": list(SENSORS),
}
XGB_FIXED = dict(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                 tree_method="hist", n_jobs=4)


def clip(p):
    # float64 so scored values and the saved CSV are identical (XGBoost returns float32)
    return np.clip(np.asarray(p, dtype=np.float64), 0, RUL_CAP)


def score(y, p, y_uncapped) -> dict:
    lo, hi = bootstrap_ci(y, p)
    return {"rmse": rmse(y, p), "rmse_ci95": [lo, hi], "mae": mae(y, p), "phm": phm_score(y, p),
            "rmse_vs_uncapped_truth": rmse(y_uncapped, p), "phm_vs_uncapped_truth": phm_score(y_uncapped, p),
            "late_share": float(np.mean(p > y))}


def tune_ridge(tr, va, cols):
    """Pick the penalty on validation engines. Also returns the whole curve, so a flat curve is visible."""
    curve = {a: rmse(va.RUL, clip(Ridge(alpha=a).fit(tr[cols], tr.RUL).predict(va[cols])))
             for a in (0.01, 0.1, 1, 10, 100, 1000)}
    best = min(curve, key=curve.get)
    return {"alpha": best, "val_rmse": curve[best], "val_rmse_by_alpha": {str(a): v for a, v in curve.items()}}


def xgb_val(tr, va, cols, depth, seed):
    m = XGBRegressor(n_estimators=3000, max_depth=depth, random_state=seed, early_stopping_rounds=100,
                     eval_metric="rmse", **XGB_FIXED)
    m.fit(tr[cols], tr.RUL, eval_set=[(va[cols], va.RUL)], verbose=False)
    return m.best_iteration + 1, rmse(va.RUL, clip(m.predict(va[cols], iteration_range=(0, m.best_iteration + 1))))


def main() -> int:
    full, te = train_frame(), test_frame()
    tr, va = full[full.split == "train"], full[full.split == "val"]
    y, y_raw = te.RUL.to_numpy(), te.RUL_true.to_numpy()
    print(f"train {len(tr):,} rows / 80 engines | val {len(va):,} rows / 20 engines | test {len(te)} engines")
    results, preds = {}, pd.DataFrame({"unit": te.unit, "RUL_true": y_raw, "RUL": y})

    # 1. Fleet average: one number for every engine
    avg = float(full.RUL.mean())
    p = np.full(len(te), avg)
    results["Fleet average"] = {"features": "none", "settings": {"constant": avg},
                                "val_rmse": rmse(va.RUL, np.full(len(va), float(tr.RUL.mean()))), **score(y, p, y_raw)}
    preds["Fleet average"] = p

    # 2. Ridge regression (linear) on raw sensors and on all features
    for name, cols in (("Ridge, raw sensors (14)", SENSORS), ("Ridge, all 168 features", ALL)):
        t = tune_ridge(tr, va, cols)
        p = clip(Ridge(alpha=t["alpha"]).fit(full[cols], full.RUL).predict(te[cols]))
        results[name] = {"features": f"{len(cols)} columns",
                         "settings": {"alpha": t["alpha"], "val_rmse_by_alpha": t["val_rmse_by_alpha"]},
                         "val_rmse": t["val_rmse"], **score(y, p, y_raw)}
        preds[name] = p
        print(f"{name}: alpha={t['alpha']} val {t['val_rmse']:.2f} test {results[name]['rmse']:.2f}")

    # 3. XGBoost: pick tree depth on validation (seed 0, all features), then use it for every feature set
    depth_val = {d: xgb_val(tr, va, ALL, d, 0) for d in (3, 5, 7)}
    depth = min(depth_val, key=lambda d: depth_val[d][1])
    print("XGBoost depth search (val RMSE):", {d: round(v[1], 3) for d, v in depth_val.items()}, "-> depth", depth)

    importance = None
    for set_name, cols in FEATURE_SETS.items():
        runs = []
        for seed in SEEDS:
            n_trees, v = xgb_val(tr, va, cols, depth, seed)
            m = XGBRegressor(n_estimators=n_trees, max_depth=depth, random_state=seed, **XGB_FIXED)
            m.fit(full[cols], full.RUL)
            p = clip(m.predict(te[cols]))
            runs.append({"seed": seed, "trees": n_trees, "val_rmse": v, "test_rmse": rmse(y, p), "pred": p})
            if set_name == "all 168 features" and seed == 0:
                gain = m.get_booster().get_score(importance_type="gain")
                importance = pd.Series(gain).reindex(cols).fillna(0).sort_values(ascending=False)
        name = f"XGBoost, {set_name}"
        p0 = runs[0]["pred"]  # seed 0 is the published prediction; the other seeds measure stability
        tests = [r["test_rmse"] for r in runs]
        results[name] = {"features": f"{len(cols)} columns", "settings": {"max_depth": depth, **XGB_FIXED,
                         "trees_per_seed": [r["trees"] for r in runs]},
                         "val_rmse": float(np.mean([r["val_rmse"] for r in runs])), **score(y, p0, y_raw),
                         "seed_test_rmse": tests, "seed_test_rmse_mean": float(np.mean(tests)),
                         "seed_test_rmse_std": float(np.std(tests, ddof=1))}
        preds[name] = p0
        print(f"{name}: val {results[name]['val_rmse']:.2f} test {results[name]['rmse']:.2f} "
              f"(seeds {np.mean(tests):.2f} +/- {np.std(tests, ddof=1):.2f})")

    # 4. Is a gap real or luck? Paired bootstrap on the same test engines
    main_model = "XGBoost, all 168 features"
    comparisons = {}
    pairs = [(main_model, o) for o in ["Fleet average", "Ridge, raw sensors (14)", "Ridge, all 168 features"] +
             [f"XGBoost, {s}" for s in FEATURE_SETS if s != "all 168 features"]]
    pairs.append(("Ridge, all 168 features", "Ridge, raw sensors (14)"))  # value of feature engineering alone
    for a, b in pairs:
        comparisons[f"{a} vs {b}"] = {"rmse": paired_bootstrap(y, preds[a], preds[b]),
                                      "phm": paired_bootstrap(y, preds[a], preds[b], metric=phm_score)}

    out = {"protocol": "settings chosen on 20 validation engines; refit on all 100 training engines; "
                       "test set scored once at each engine's last observed cycle; true RUL capped at 125",
           "xgb_depth_search_val_rmse": {str(d): v[1] for d, v in depth_val.items()},
           "models": results, "paired_comparisons": comparisons}
    (OUT / "results_phase4.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    preds.to_csv(OUT / "predictions_phase4.csv", index=False)
    importance.rename("gain").to_csv(OUT / "importance_phase4.csv", index_label="feature")
    print("\nWrote models/results_phase4.json, predictions_phase4.csv, importance_phase4.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
