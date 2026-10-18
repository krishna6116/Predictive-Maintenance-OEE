"""Independent verification of Phase 4 (run: python -m models.verify_baselines).

Rechecks the published results with different code paths: plain-Python file reading, pure-Python metric
loops, a closed-form Ridge solution in NumPy, and an exact XGBoost re-run. Exits 1 if any check fails.
"""
import json
import os
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
    if not ok and os.environ.get("GITHUB_ACTIONS"):
        print(f"::error title={Path(__file__).stem} check failed::{what}")  # visible on the run page


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

    print("\n== XGBoost re-run from scratch (same data, seeds and settings)")
    import xgboost
    from xgboost import XGBRegressor
    from models.baselines import FEATURE_SETS, SEEDS, XGB_FIXED
    from models.data import test_frame, train_frame
    full, te = train_frame(), test_frame()
    check(len(full) == len(train_rows) and set(full.unit) == set(range(1, 101)),
          f"model trained on exactly the {len(train_rows):,} training readings (no test data)")
    r = RESULTS["models"]["XGBoost, all 168 features"]
    env = RESULTS["environment"]
    check(all(r["settings"][k] == v for k, v in XGB_FIXED.items()) and env["n_jobs"] == XGB_FIXED["n_jobs"] == 1,
          "published XGBoost settings equal the code's, with a fixed single thread (thread count changes results)")
    cols = FEATURE_SETS["all 168 features"]
    reruns = []
    for seed, trees in zip(SEEDS, r["settings"]["trees_per_seed"]):
        m = XGBRegressor(n_estimators=trees, max_depth=r["settings"]["max_depth"], random_state=seed,
                         **XGB_FIXED).fit(full[cols], full.RUL)
        reruns.append(np.clip(m.predict(te[cols]).astype(np.float64), 0, RUL_CAP))
    if env["os"] == sys.platform and env["xgboost"] == xgboost.__version__:
        # Same OS and library as the published run: predictions must be identical
        diff = float(np.abs(reruns[0] - PREDS["XGBoost, all 168 features"]).max())
        check(diff < 1e-3, f"XGBoost (all features, seed 0) reproduces its predictions; max diff {diff:.1e}")
    else:
        # XGBoost's row/column sampling draws through the C++ standard library, which differs between OSes
        # (e.g. Windows vs Linux), so exact predictions cannot be expected here. Instead the 5 re-run seeds
        # must agree with the 5 published ones: the published seed spread (std 0.20) puts the 5-seed mean
        # within about +/-0.12, so 0.6 (~5 standard errors) never fails by chance yet catches real errors
        # such as wrong features, leakage or wrong tree counts, which move RMSE by a point or more.
        rerun_mean = float(np.mean([py_rmse(list(PREDS.RUL), list(p)) for p in reruns]))
        gap = rerun_mean - r["seed_test_rmse_mean"]
        print(f"note: published on {env['os']} with xgboost {env['xgboost']}; this is {sys.platform} with "
              f"xgboost {xgboost.__version__}, so the exact check is replaced by a 5-seed comparison")
        check(abs(gap) <= 0.6, f"XGBoost 5-seed mean test RMSE {rerun_mean:.3f} vs published "
                               f"{r['seed_test_rmse_mean']:.3f} (gap {gap:+.3f}, allowed 0.6)")

    print("\n== Age-only baseline from the raw training file (plain Python)")
    by_cycle = {}
    for row, c in zip(train_rows, caps):
        by_cycle.setdefault(int(row["cycle"]), []).append(c)
    table = {k: sum(v) / len(v) for k, v in by_cycle.items()}
    top = max(table)

    def age_pred(cycle):
        k = min(int(cycle), top)
        while k not in table:  # carry the last known value forward over any gap
            k -= 1
        return min(max(table[k], 0), RUL_CAP)
    p = np.array([age_pred(last_test[u]["cycle"]) for u in sorted(last_test)])
    diff = float(np.abs(p - PREDS["Age only (cycle count)"]).max())
    check(diff < 1e-9, f"age-only predictions recomputed from per-cycle means; max diff {diff:.1e}")

    print("\n== Settings were chosen on validation, never on test")
    vp = pd.read_csv("models/val_predictions_phase4.csv")
    split = json.loads(Path("features/split.json").read_text())
    check(sorted(vp.unit.unique()) == sorted(split["val"]), "validation predictions cover exactly the 20 validation engines")
    curve = RESULTS["models"]["Ridge, raw sensors (14)"]["settings"]["val_rmse_by_alpha"]
    check(float(min(curve, key=curve.get)) == alpha, "Ridge alpha = the minimum of its validation curve")
    dv = RESULTS["xgb_depth_search_val_rmse"]
    check(int(min(dv, key=dv.get)) == r["settings"]["max_depth"], "XGBoost depth = the minimum of the validation depth search")
    tr80 = [row for row, c in zip(train_rows, caps) if int(row["unit"]) in split["train"]]
    y80 = np.array([c for row, c in zip(train_rows, caps) if int(row["unit"]) in split["train"]])
    X80 = scaled(tr80)
    xm, ym = X80.mean(0), y80.mean()
    w = np.linalg.solve((X80 - xm).T @ (X80 - xm) + alpha * np.eye(X80.shape[1]), (X80 - xm).T @ (y80 - ym))
    val_rows = sorted((row for row in train_rows if int(row["unit"]) in split["val"]), key=lambda q: (q["unit"], q["cycle"]))
    pv = np.clip(scaled(val_rows) @ w + (ym - xm @ w), 0, RUL_CAP)
    vps = vp.sort_values(["unit", "cycle"])
    diff = float(np.abs(pv - vps["Ridge, raw sensors (14)"].to_numpy()).max())
    check(diff < 1e-6, f"validation predictions come from a fit on the 80 training engines only (closed-form Ridge, diff {diff:.1e})")
    tl = (vps.cycle >= 31) & (vps.RUL_true <= 145)
    got = RESULTS["models"]["XGBoost, all 168 features"]["val_rmse_testlike"]
    check(abs(py_rmse(list(vps.RUL[tl]), list(vps["XGBoost, all 168 features"][tl])) - got) < 1e-9,
          f"test-like validation RMSE recomputes ({got:.2f} on {int(tl.sum()):,} rows)")
    sel = RESULTS["model_selection_on_validation"]["XGBoost, all 168 features vs Ridge, all 168 features"]
    near = vps.RUL_true <= 50
    gap = py_mae(list(vps.RUL[near]), list(vps["XGBoost, all 168 features"][near])) - \
        py_mae(list(vps.RUL[near]), list(vps["Ridge, all 168 features"][near]))
    check(abs(gap - sel["val_near_failure_mae"]["diff"]) < 1e-9 and sel["val_near_failure_mae"]["ci"][1] < 0,
          f"on validation, XGBoost beats Ridge near failure (MAE gap {gap:+.2f}, interval excludes 0): choice made without the test set")

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
    ok = True
    for metric in ("rmse", "phm"):
        ps = sorted((c[metric]["p"], name) for name, c in comp.items())
        m_, running = len(ps), 0.0
        for rank, (p_, name) in enumerate(ps):  # Holm by hand: running max of p x tests left
            running = max(running, min(1.0, p_ * (m_ - rank)))
            ok &= abs(comp[name][metric]["p_holm"] - running) < 1e-12 and comp[name][metric]["significant"] == (running < 0.05)
    check(ok, "Holm-corrected p-values and verdicts recompute for both metrics")

    print("\n== The published document matches the results")
    from models.report_phase4 import DOC, render
    check(DOC.read_text(encoding="utf-8") == render(), "docs/MODEL_METRICS.md equals a fresh rendering of the results")
    # The README's headline numbers are hand-written prose, so each one is recomputed here and must appear in it
    readme = Path("README.md").read_text(encoding="utf-8")
    ms = RESULTS["models"]
    x = ms["XGBoost, all 168 features"]
    near = [i for i, t in enumerate(PREDS.RUL) if t <= 50]
    near_mae = {m: sum(abs(PREDS[m][i] - PREDS.RUL[i]) for i in near) / len(near)
                for m in ("XGBoost, all 168 features", "Ridge, all 168 features")}
    quoted = [f"{x['rmse']:.2f} RMSE", f"{x['rmse_ci95'][0]:.1f}-{x['rmse_ci95'][1]:.1f}",
              f"{ms['Fleet average']['rmse']:.1f}", f"{ms['Age only (cycle count)']['rmse']:.1f}",
              f"{near_mae['XGBoost, all 168 features']:.1f} cycles", f"{near_mae['Ridge, all 168 features']:.1f}"]
    missing = [q for q in quoted if q not in readme]
    check(not missing, f"README quotes the published numbers ({', '.join(quoted)})"
                       + (f"; missing: {', '.join(missing)}" if missing else ""))

    failed = [w for ok, w in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # a crash must be as visible on CI as a failed check
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::error title={Path(__file__).stem} crashed::{type(exc).__name__}: {exc}")
        raise
