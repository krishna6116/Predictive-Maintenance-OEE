"""Independent verification of Phase 4 (run: python -m models.verify_baselines).

Rechecks the published results with different code paths: plain-Python file reading, pure-Python metric
loops, a closed-form Ridge solution in NumPy, and an exact XGBoost re-run. Exits 1 if any check fails.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features.build import RUL_CAP
from pipeline.config import RAW, SENSORS

RESULTS = json.loads(Path("models/results_phase4.json").read_text())
PREDS = pd.read_csv("models/predictions_phase4.csv")
SCALER = json.loads(Path("features/scaler.json").read_text())
COLS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
checks: list[tuple[bool, str]] = []


def check(ok, what):
    checks.append((bool(ok), what))
    print(("PASS " if ok else "FAIL ") + what)


def read_plain(path: Path) -> list[dict]:
    """Parse a C-MAPSS file with plain Python (no pandas)."""
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                vals = line.split()
                rows.append({c: float(v) for c, v in zip(COLS, vals)})
    return rows


def py_rmse(y, p):
    return math.sqrt(sum((b - a) ** 2 for a, b in zip(y, p)) / len(y))


def py_mae(y, p):
    return sum(abs(b - a) for a, b in zip(y, p)) / len(y)


def py_phm(y, p):
    total = 0.0
    for a, b in zip(y, p):
        d = b - a
        total += math.exp(-d / 13) - 1 if d < 0 else math.exp(d / 10) - 1
    return total


def main() -> int:
    train_rows, test_rows = read_plain(RAW / "train_FD001.txt"), read_plain(RAW / "test_FD001.txt")
    truth = [int(l) for l in (RAW / "RUL_FD001.txt").read_text().split()]

    print("== Test targets")
    check(list(PREDS.unit) == list(range(1, 101)), "100 test engines, in order 1-100")
    check(list(PREDS.RUL_true) == truth, "RUL_true equals NASA's RUL file line by line")
    check(all(PREDS.RUL == [min(t, RUL_CAP) for t in truth]), "scored target = NASA RUL capped at 125")

    print("\n== Predictions")
    models = list(RESULTS["models"])
    check(set(models) <= set(PREDS.columns), f"a prediction column for each of the {len(models)} models")
    vals = PREDS[models].to_numpy()
    check(np.isfinite(vals).all() and vals.min() >= 0 and vals.max() <= RUL_CAP, "all predictions finite and within 0-125")

    print("\n== Every published metric, recomputed with plain Python")
    worst = 0.0
    for m in models:
        r, y, yr, p = RESULTS["models"][m], list(PREDS.RUL), list(PREDS.RUL_true), list(PREDS[m])
        for key, val in (("rmse", py_rmse(y, p)), ("mae", py_mae(y, p)), ("phm", py_phm(y, p)),
                         ("rmse_vs_uncapped_truth", py_rmse(yr, p)), ("phm_vs_uncapped_truth", py_phm(yr, p))):
            worst = max(worst, abs(val - r[key]) / max(1.0, abs(val)))
        lo, hi = r["rmse_ci95"]
        if not lo <= r["rmse"] <= hi:
            worst = float("inf")
    check(worst < 1e-9, f"RMSE, MAE, PHM (capped and uncapped) for {len(models)} models match; max relative diff {worst:.1e}; every RMSE inside its 95% interval")

    print("\n== Fleet average from the raw training file")
    last = {}
    for r in train_rows:
        last[r["unit"]] = max(last.get(r["unit"], 0), r["cycle"])
    caps = [min(last[r["unit"]] - r["cycle"], RUL_CAP) for r in train_rows]
    avg = sum(caps) / len(caps)
    check(abs(avg - RESULTS["models"]["Fleet average"]["settings"]["constant"]) < 1e-9 and
          np.allclose(PREDS["Fleet average"], avg), f"constant = mean capped training RUL = {avg:.4f} over {len(caps):,} readings")

    print("\n== Ridge on raw sensors, re-solved in closed form (NumPy normal equations)")
    def scaled(rows):
        return np.array([[(r[s] - SCALER[s]["mean"]) / SCALER[s]["std"] for s in SENSORS] for r in rows])
    X, yv = scaled(train_rows), np.array(caps, float)
    alpha = RESULTS["models"]["Ridge, raw sensors (14)"]["settings"]["alpha"]
    xm, ym = X.mean(0), yv.mean()           # intercept is not penalised: centre, solve, recover it
    Xc = X - xm
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (yv - ym))
    b = ym - xm @ w
    last_test = {}
    for r in test_rows:
        if r["cycle"] >= last_test.get(r["unit"], {"cycle": -1})["cycle"]:
            last_test[r["unit"]] = r
    Xt = scaled([last_test[u] for u in sorted(last_test)])
    p = np.clip(Xt @ w + b, 0, RUL_CAP)
    diff = float(np.abs(p - PREDS["Ridge, raw sensors (14)"]).max())
    check(diff < 1e-6, f"closed-form Ridge (alpha={alpha}) reproduces the predictions; max diff {diff:.1e}")
    check(sorted(int(r["cycle"]) for r in last_test.values()) == sorted(
        pd.read_csv(RAW / "test_FD001.txt", sep=r"\s+", header=None).groupby(0)[1].max().tolist()),
        "each test engine is scored at its last observed cycle")

    print("\n== XGBoost re-run from scratch (deterministic: same data + seed must give the same predictions)")
    from xgboost import XGBRegressor
    from models.baselines import FEATURE_SETS, XGB_FIXED
    from models.data import test_frame, train_frame
    full, te = train_frame(), test_frame()
    check(len(full) == len(train_rows) and set(full.unit) == set(range(1, 101)),
          f"model trained on exactly the {len(train_rows):,} training readings (no test data)")
    r = RESULTS["models"]["XGBoost, all 168 features"]
    m = XGBRegressor(n_estimators=r["settings"]["trees_per_seed"][0], max_depth=r["settings"]["max_depth"],
                     random_state=0, **XGB_FIXED).fit(full[FEATURE_SETS["all 168 features"]], full.RUL)
    p = np.clip(m.predict(te[FEATURE_SETS["all 168 features"]]), 0, RUL_CAP)
    diff = float(np.abs(p - PREDS["XGBoost, all 168 features"]).max())
    check(diff < 1e-3, f"XGBoost (all features, seed 0) reproduces its predictions; max diff {diff:.1e}")

    print("\n== Claims about which model is better")
    comp = RESULTS["paired_comparisons"]
    ok = True
    for name, c in comp.items():
        a, b = name.split(" vs ")
        for metric, fn in (("rmse", py_rmse), ("phm", py_phm)):
            expected = fn(PREDS.RUL, PREDS[a]) - fn(PREDS.RUL, PREDS[b])
            lo, hi = c[metric]["ci"]
            ok &= abs(expected - c[metric]["diff"]) < 1e-6 * max(1, abs(expected)) and lo <= expected <= hi
    check(ok, f"{len(comp)} paired comparisons: each observed gap recomputes exactly and lies inside its interval")

    failed = [w for ok, w in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
