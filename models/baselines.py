"""Phase 4: baseline models on NASA's official FD001 test set.

Run:  python -m models.baselines        (about 10-15 minutes on a laptop CPU)

Protocol (the test set is used once, at the end):
  1. Choose settings on the 20 validation engines only (Ridge alpha, XGBoost depth and tree count).
  2. Refit each model on all 100 training engines with those settings.
  3. Score each model once on the 100 test engines, at each engine's last observed cycle.
XGBoost is trained with 5 random seeds to show how much results move by chance.
The published model is chosen on validation only (test-like rows and near failure); the test set confirms it.
Outputs: models/results_phase4.json, predictions_phase4.csv (test), val_predictions_phase4.csv, importance_phase4.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from features.build import RUL_CAP, feature_columns
from models.data import test_frame, train_frame
from models.metrics import bootstrap_ci, holm, mae, paired_bootstrap, phm_score, rmse
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
# n_jobs=1: XGBoost's results depend on the thread count (1 vs 4 threads moved test predictions by up to
# 10 cycles on the same machine), so a fixed single thread keeps results independent of the CPU.
XGB_FIXED = dict(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                 tree_method="hist", n_jobs=1)


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
    pv = clip(m.predict(va[cols], iteration_range=(0, m.best_iteration + 1)))
    return m.best_iteration + 1, rmse(va.RUL, pv), pv


def test_like(va: pd.DataFrame) -> pd.Series:
    """Validation rows that look like the test set: at least 31 cycles observed, true RUL at most 145.

    The test engines are observed for 31-303 cycles and have true RUL 7-145, so this score is closer to
    what the test set measures than the all-rows validation score (which includes many easy cap rows).
    """
    return (va.cycle >= 31) & (va.RUL_true <= 145)


def age_model(rows: pd.DataFrame):
    """RUL predicted from age alone: the mean capped RUL of training readings at the same cycle."""
    table = rows.groupby("cycle").RUL.mean()
    table = table.reindex(range(1, int(table.index.max()) + 1)).ffill()
    return lambda cycles: clip(table.reindex(np.minimum(cycles, table.index.max())).to_numpy())


def main() -> int:
    full, te = train_frame(), test_frame()
    tr, va = full[full.split == "train"], full[full.split == "val"]
    y, y_raw = te.RUL.to_numpy(), te.RUL_true.to_numpy()
    print(f"train {len(tr):,} rows / 80 engines | val {len(va):,} rows / 20 engines | test {len(te)} engines")
    results, preds = {}, pd.DataFrame({"unit": te.unit, "RUL_true": y_raw, "RUL": y})
    vpreds = va[["unit", "cycle", "RUL", "RUL_true"]].reset_index(drop=True)  # 80-engine fits scored on validation
    tl = test_like(vpreds).to_numpy()

    def val_scores(p):
        p = np.asarray(p, float)
        return {"val_rmse": rmse(vpreds.RUL, p), "val_rmse_testlike": rmse(vpreds.RUL[tl], p[tl])}

    # 1. Fleet average: one number for every engine
    avg = float(full.RUL.mean())
    p = np.full(len(te), avg)
    vpreds["Fleet average"] = np.full(len(va), float(tr.RUL.mean()))
    results["Fleet average"] = {"features": "none", "settings": {"constant": avg},
                                **val_scores(vpreds["Fleet average"]), **score(y, p, y_raw)}
    preds["Fleet average"] = p

    # 1b. Age only: how much does knowing just the cycle count tell you?
    age = "Age only (cycle count)"
    vpreds[age] = age_model(tr)(va.cycle.to_numpy())
    p = age_model(full)(te.cycle.to_numpy())
    results[age] = {"features": "cycle", "settings": {"rule": "mean capped RUL of training readings at the same cycle"},
                    **val_scores(vpreds[age]), **score(y, p, y_raw)}
    preds[age] = p
    print(f"{age}: val {results[age]['val_rmse']:.2f} test {results[age]['rmse']:.2f}")

    # 2. Ridge regression (linear) on raw sensors and on all features
    for name, cols in (("Ridge, raw sensors (14)", SENSORS), ("Ridge, all 168 features", ALL)):
        t = tune_ridge(tr, va, cols)
        vpreds[name] = clip(Ridge(alpha=t["alpha"]).fit(tr[cols], tr.RUL).predict(va[cols]))
        p = clip(Ridge(alpha=t["alpha"]).fit(full[cols], full.RUL).predict(te[cols]))
        results[name] = {"features": f"{len(cols)} columns",
                         "settings": {"alpha": t["alpha"], "val_rmse_by_alpha": t["val_rmse_by_alpha"]},
                         **val_scores(vpreds[name]), **score(y, p, y_raw)}
        preds[name] = p
        print(f"{name}: alpha={t['alpha']} val {t['val_rmse']:.2f} test {results[name]['rmse']:.2f}")

    # 3. XGBoost: pick tree depth on validation (seed 0, all features), then use it for every feature set
    depth_val = {d: xgb_val(tr, va, ALL, d, 0) for d in (3, 5, 7)}
    depth = min(depth_val, key=lambda d: depth_val[d][1])
    depth_testlike = {d: rmse(vpreds.RUL[tl], v[2][tl]) for d, v in depth_val.items()}
    print("XGBoost depth search (val RMSE):", {d: round(v[1], 3) for d, v in depth_val.items()}, "-> depth", depth,
          "| test-like val:", {d: round(v, 3) for d, v in depth_testlike.items()})

    importance = None
    for set_name, cols in FEATURE_SETS.items():
        runs = []
        for seed in SEEDS:
            n_trees, v, pv = xgb_val(tr, va, cols, depth, seed)
            m = XGBRegressor(n_estimators=n_trees, max_depth=depth, random_state=seed, **XGB_FIXED)
            m.fit(full[cols], full.RUL)
            p = clip(m.predict(te[cols]))
            runs.append({"seed": seed, "trees": n_trees, "val_rmse": v, "test_rmse": rmse(y, p), "pred": p, "vpred": pv})
            if set_name == "all 168 features" and seed == 0:
                gain = m.get_booster().get_score(importance_type="gain")
                importance = pd.Series(gain).reindex(cols).fillna(0).sort_values(ascending=False)
        name = f"XGBoost, {set_name}"
        p0 = runs[0]["pred"]  # seed 0 is the published prediction; the other seeds measure stability
        tests = [r["test_rmse"] for r in runs]
        vpreds[name] = runs[0]["vpred"]
        results[name] = {"features": f"{len(cols)} columns", "settings": {"max_depth": depth, **XGB_FIXED,
                         "trees_per_seed": [r["trees"] for r in runs]},
                         **val_scores(runs[0]["vpred"]),
                         "val_rmse_mean_over_seeds": float(np.mean([r["val_rmse"] for r in runs])),
                         **score(y, p0, y_raw),
                         "seed_test_rmse": tests, "seed_test_rmse_mean": float(np.mean(tests)),
                         "seed_test_rmse_std": float(np.std(tests, ddof=1))}
        preds[name] = p0
        print(f"{name}: val {results[name]['val_rmse']:.2f} test {results[name]['rmse']:.2f} "
              f"(seeds {np.mean(tests):.2f} +/- {np.std(tests, ddof=1):.2f})")

    # 4. Model choice from VALIDATION only: test-like rows and near failure, resampling whole engines
    main_model = "XGBoost, all 168 features"
    near = (vpreds.RUL_true <= 50).to_numpy()
    selection = {}
    for other in ["Ridge, all 168 features", age]:
        selection[f"{main_model} vs {other}"] = {
            "val_near_failure_mae": paired_bootstrap(vpreds.RUL[near], vpreds[main_model][near], vpreds[other][near],
                                                     metric=mae, groups=vpreds.unit[near]),
            "val_testlike_rmse": paired_bootstrap(vpreds.RUL[tl], vpreds[main_model][tl], vpreds[other][tl],
                                                  groups=vpreds.unit[tl])}
    for k, v in selection.items():
        print("validation:", k, {m: (round(c["diff"], 2), [round(x, 2) for x in c["ci"]]) for m, c in v.items()})

    # 5. Is a gap real or luck? Paired bootstrap on the same test engines, Holm-corrected across comparisons
    comparisons = {}
    pairs = [(main_model, o) for o in ["Fleet average", age, "Ridge, raw sensors (14)", "Ridge, all 168 features"] +
             [f"XGBoost, {s}" for s in FEATURE_SETS if s != "all 168 features"]]
    pairs.append(("Ridge, all 168 features", "Ridge, raw sensors (14)"))  # value of feature engineering alone
    for a, b in pairs:
        comparisons[f"{a} vs {b}"] = {"rmse": paired_bootstrap(y, preds[a], preds[b]),
                                      "phm": paired_bootstrap(y, preds[a], preds[b], metric=phm_score)}
    for metric in ("rmse", "phm"):
        adjusted = holm({k: c[metric]["p"] for k, c in comparisons.items()})
        for k in comparisons:
            comparisons[k][metric].update(adjusted[k])

    out = {"protocol": "settings chosen on 20 validation engines; refit on all 100 training engines; "
                       "test set scored once at each engine's last observed cycle; true RUL capped at 125",
           "xgb_depth_search_val_rmse": {str(d): v[1] for d, v in depth_val.items()},
           "xgb_depth_search_val_rmse_testlike": {str(d): v for d, v in depth_testlike.items()},
           "test_like_validation": f"validation rows with cycle >= 31 and true RUL <= 145 ({int(tl.sum()):,} of {len(tl):,})",
           "model_selection_on_validation": selection,
           "models": results, "paired_comparisons": comparisons,
           # XGBoost's row/column sampling draws through the C++ standard library, whose output differs between
           # Windows and Linux; the verifier demands identical predictions only on the OS recorded here
           "environment": {"os": sys.platform, "xgboost": xgboost.__version__, "n_jobs": XGB_FIXED["n_jobs"]}}
    (OUT / "results_phase4.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    preds.to_csv(OUT / "predictions_phase4.csv", index=False)
    vpreds.to_csv(OUT / "val_predictions_phase4.csv", index=False)
    importance.rename("gain").to_csv(OUT / "importance_phase4.csv", index_label="feature")
    print("\nWrote models/results_phase4.json, predictions_phase4.csv, val_predictions_phase4.csv, importance_phase4.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
