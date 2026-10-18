"""Independent verification of Phase 5b, the FD003 confirmation (run: python -m models.verify_fd003).

Rechecks the published FD003 results with different code paths: a plain-Python scaler and window builder,
pure-Python metrics, quantiles and Holm, interval coverage recomputed from the saved MC passes, bootstrap
intervals re-drawn with a different random seed, and a re-training check (exact on the producing OS;
statistical elsewhere). Exits 1 if any check fails.
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features.build import RUL_CAP
from models.verify_deep import py_mae, py_phm, py_quantile, py_rmse, py_std, read_plain
from pipeline.config import RAW, SENSORS

DS = "FD003"
R = json.loads(Path("models/results_phase5b.json").read_text())
P5 = json.loads(Path("models/results_phase5.json").read_text())
P4 = json.loads(Path("models/results_phase4.json").read_text())
RULE = json.loads(Path("models/calibration_rule.json").read_text())
PREDS = pd.read_csv("models/predictions_phase5b.csv", float_precision="round_trip")
VPREDS = pd.read_csv("models/val_predictions_phase5b.csv", float_precision="round_trip")
DRAWS = pd.read_csv("models/mc_draws_test_phase5b.csv", float_precision="round_trip")
T1 = pd.read_csv("models/predictions_phase5.csv", float_precision="round_trip")
V1 = pd.read_csv("models/val_predictions_phase5.csv", float_precision="round_trip")
SPLIT = json.loads(Path(f"features/split_{DS}.json").read_text())
SCALER = json.loads(Path(f"features/scaler_{DS}.json").read_text())
XGB = "XGBoost, all 168 features"
SEED_SD_MULTIPLE = 4.0  # same cross-OS band as Phase 5
checks: list[tuple[bool, str]] = []


def check(ok, what):
    checks.append((bool(ok), what))
    print(("PASS " if ok else "FAIL ") + what)
    if not ok and os.environ.get("GITHUB_ACTIONS"):
        print(f"::error title={Path(__file__).stem} check failed::{what}")


def windows(rows, length, last_only=False):
    """Plain-Python causal windows with the FD003 scaler; left padding repeats the first reading."""
    by_unit = {}
    for r in rows:
        by_unit.setdefault(int(r["unit"]), []).append(r)
    out = []
    for unit in sorted(by_unit):
        seq = sorted(by_unit[unit], key=lambda r: r["cycle"])
        scaled = [[(r[s] - SCALER[s]["mean"]) / SCALER[s]["std"] for s in SENSORS] for r in seq]
        for t in ([len(seq) - 1] if last_only else range(len(seq))):
            out.append((unit, int(seq[t]["cycle"]), [scaled[max(0, i)] for i in range(t - length + 1, t + 1)]))
    return out


def coverage(y, lo, hi, mask=None):
    idx = [i for i in range(len(y)) if mask is None or mask[i]]
    return sum(lo[i] <= y[i] <= hi[i] for i in idx) / len(idx)


def zone(y, mean, sd, cycle, rul_true):
    """Rule E by hand: 90th percentiles of |error|/sd on test-like readings, split by predicted RUL <= 50."""
    near, far = [], []
    for a, m, s, c, t in zip(y, mean, sd, cycle, rul_true):
        if c >= RULE["test_like"]["min_cycle"] and t <= RULE["test_like"]["max_true_rul"]:
            (near if m <= RULE["near_zone_cycles"] else far).append(abs(a - m) / max(s, 1e-6))
    return py_quantile(near, RULE["nominal"]), py_quantile(far, RULE["nominal"])


def interval(mean, sd, kn, kf):
    lo = [max(m - (kn if m <= RULE["near_zone_cycles"] else kf) * s, 0.0) for m, s in zip(mean, sd)]
    hi = [m + (kn if m <= RULE["near_zone_cycles"] else kf) * s for m, s in zip(mean, sd)]
    return lo, hi


def boot(groups, metric, n=2000, seed=12345):
    """Second bootstrap implementation (Python's random module), engines resampled within each dataset."""
    import random
    rnd = random.Random(seed)
    diffs = []
    for _ in range(n):
        ys, a_, b_ = [], [], []
        for y, a, b in groups:
            idx = [rnd.randrange(len(y)) for _ in range(len(y))]
            ys += [y[i] for i in idx]
            a_ += [a[i] for i in idx]
            b_ += [b[i] for i in idx]
        diffs.append(metric(ys, a_) - metric(ys, b_))
    return py_quantile(diffs, 0.025), py_quantile(diffs, 0.975)


def main() -> int:
    M = R["models"]
    print(f"== Data and preparation ({DS})")
    sums = dict(reversed(l.split()) for l in Path("data/checksums.sha256").read_text().splitlines() if l.strip())
    files = [f"train_{DS}.txt", f"test_{DS}.txt", f"RUL_{DS}.txt"]
    check(all(hashlib.sha256((RAW / f).read_bytes()).hexdigest() == sums[f] for f in files), "FD003 files match the pinned SHA-256 checksums")
    train_rows, test_rows = read_plain(RAW / f"train_{DS}.txt"), read_plain(RAW / f"test_{DS}.txt")
    truth = [int(x) for x in (RAW / f"RUL_{DS}.txt").read_text().split()]
    units = sorted({int(r["unit"]) for r in train_rows})
    check(sorted(SPLIT["train"] + SPLIT["val"]) == units and len(SPLIT["val"]) == 20 and not set(SPLIT["train"]) & set(SPLIT["val"]),
          "engine split: 80 training + 20 validation engines, disjoint, covering all 100")
    tr = [r for r in train_rows if int(r["unit"]) in set(SPLIT["train"])]
    worst = max(max(abs(sum(r[s] for r in tr) / len(tr) - SCALER[s]["mean"]),
                    abs(py_std([r[s] for r in tr]) - SCALER[s]["std"])) for s in SENSORS)
    check(worst < 1e-9, f"scaler = mean and sd of the 80 training engines only, recomputed by hand (max diff {worst:.1e})")
    check(list(PREDS.unit) == list(range(1, 101)) and list(PREDS.RUL_true) == truth and
          list(PREDS.RUL) == [min(t, RUL_CAP) for t in truth], "100 test engines in order; truth = NASA's RUL file; scored target capped at 125")

    print("\n== Frozen from FD001 (nothing re-chosen on FD003)")
    fr = R["frozen_from_fd001"]
    check(fr["deep_settings"] == P5["final_settings"] and all(M[a]["settings"] == P5["final_settings"][a] for a in ("LSTM", "CNN")),
          "LSTM and CNN use the Phase 5 settings")
    check(M[XGB]["settings"]["max_depth"] == P4["models"][XGB]["settings"]["max_depth"] == fr["xgb_depth"],
          f"XGBoost uses the Phase 4 depth ({fr['xgb_depth']})")
    check(RULE["chosen"] == "E" and R["mc_dropout"]["rule"].startswith("E"), "intervals use the pre-registered rule E")

    from models.sequences import datasets
    W = M["LSTM"]["settings"]["window"]
    d = datasets(W, DS)
    (X_all, i_all), (X_te, i_te) = d["all"], d["test"]
    pw = windows(train_rows, W)
    diff = float(np.abs(np.array([w for _, _, w in pw], dtype=np.float64) - X_all).max())
    check([(u, c) for u, c, _ in pw] == list(zip(i_all.unit, i_all.cycle)) and diff < 1e-4,
          f"all {len(pw):,} FD003 training windows match an independent rebuild (max diff {diff:.1e})")
    pt = windows(test_rows, W, last_only=True)
    diff = float(np.abs(np.array([w for _, _, w in pt], dtype=np.float64) - X_te).max())
    check([u for u, _, _ in pt] == list(range(1, 101)) and diff < 1e-4, f"100 FD003 test windows match (max diff {diff:.1e})")
    check(sorted(VPREDS.unit.unique()) == sorted(SPLIT["val"]), "validation predictions cover exactly the 20 validation engines")

    print("\n== Every published FD003 metric, recomputed with plain Python")
    y, yr = list(PREDS.RUL), list(PREDS.RUL_true)
    near = [t <= 50 for t in yr]
    worst = 0.0
    for name, m in M.items():
        p = list(PREDS[name])
        got = [py_rmse(y, p), py_mae(y, p), py_phm(y, p), py_rmse(yr, p), py_phm(yr, p), sum(a > b for a, b in zip(p, y)) / len(y),
               py_mae([a for a, n in zip(y, near) if n], [b for b, n in zip(p, near) if n])]
        pub = [m["rmse"], m["mae"], m["phm"], m["rmse_vs_uncapped_truth"], m["phm_vs_uncapped_truth"], m["late_share"], m["near_failure_mae"]]
        worst = max(worst, max(abs(a - b) for a, b in zip(got, pub)))
        if "seed_test_predictions" in m:
            rm = [py_rmse(y, s) for s in m["seed_test_predictions"]]
            worst = max(worst, max(abs(a - b) for a, b in zip(rm, m["seed_test_rmse"])),
                        abs(sum(rm) / len(rm) - m["seed_test_rmse_mean"]), abs(py_std(rm) - m["seed_test_rmse_std"]),
                        max(abs(a - b) for a, b in zip(m["seed_test_predictions"][0], p)))
    check(worst < 1e-9, f"RMSE, MAE, PHM08 (capped, uncapped), late share, near-failure MAE and seed statistics for {len(M)} models (max diff {worst:.1e})")
    vy = list(VPREDS.RUL)
    tl = [c >= 31 and t <= 145 for c, t in zip(VPREDS.cycle, VPREDS.RUL_true)]
    worst = max(max(abs(py_rmse(vy, list(VPREDS[n])) - M[n]["val_rmse"]),
                    abs(py_rmse([a for a, k in zip(vy, tl) if k], [b for b, k in zip(VPREDS[n], tl) if k]) - M[n]["val_rmse_testlike"]))
                for n in M)
    check(worst < 1e-9, f"validation RMSE (all and test-like) for every model (max diff {worst:.1e})")

    print("\n== Intervals (rule E), recomputed from the saved MC passes")
    mc = R["mc_dropout"]
    kn, kf = zone(vy, list(VPREDS["MC mean"]), list(VPREDS["MC std"]), list(VPREDS.cycle), list(VPREDS.RUL_true))
    passes = [c for c in DRAWS.columns if c.startswith("pass_")]
    draws = DRAWS[passes].to_numpy().tolist()
    mean = [sum(r) / len(r) for r in draws]
    sd = [py_std(r) for r in draws]
    lo, hi = interval(mean, sd, kn, kf)
    f3 = mc["FD003"]
    got = [kn, kf, coverage(y, lo, hi), coverage(y, lo, hi, near), sum(h - l for l, h in zip(lo, hi)) / len(lo)]
    pub = [f3["k_near"], f3["k_far"], f3["test_coverage"], f3["test_coverage_near_failure"], f3["test_mean_width"]]
    check(len(passes) == mc["passes"] and max(abs(a - b) for a, b in zip(got, pub)) < 1e-9,
          f"FD003: k {kn:.2f}/{kf:.2f}, test coverage {got[2]:.0%} ({got[3]:.0%} near failure), recomputed")
    check(max(abs(a - b) for a, b in zip(mean, PREDS["MC mean"])) < 1e-9, "saved MC passes reproduce the published MC mean")
    y1 = list(T1.RUL)
    k1n, k1f = zone(list(V1.RUL), list(V1["MC mean"]), list(V1["MC std"]), list(V1.cycle), list(V1.RUL_true))
    lo1, hi1 = interval(list(T1["MC mean"]), list(T1["MC std"]), k1n, k1f)
    f1 = mc["FD001_post_hoc"]
    got = [k1n, k1f, coverage(y1, lo1, hi1), coverage(y1, lo1, hi1, [t <= 50 for t in T1.RUL_true])]
    check(max(abs(a - b) for a, b in zip(got, [f1["k_near"], f1["k_far"], f1["test_coverage"], f1["test_coverage_near_failure"]])) < 1e-9,
          f"FD001 (post hoc): rule E coverage {got[2]:.0%} ({got[3]:.0%} near failure), recomputed")

    print("\n== Pre-registered tests and verdict")
    T = R["tests"]
    n1 = [t <= 50 for t in T1.RUL_true]
    sel = lambda v, m: [a for a, k in zip(v, m) if k]
    groups = {
        "H1": ([(y1, list(T1.LSTM), list(T1[XGB])), (y, list(PREDS.LSTM), list(PREDS[XGB]))], py_rmse),
        "H2": ([(sel(y1, n1), sel(T1.LSTM, n1), sel(T1[XGB], n1)), (sel(y, near), sel(PREDS.LSTM, near), sel(PREDS[XGB], near))], py_mae),
    }
    for arch, h in (("LSTM", 3), ("CNN", 6)):
        a, b = list(PREDS[arch]), list(PREDS[XGB])
        groups[f"H{h}"] = ([(y, a, b)], py_rmse)
        groups[f"H{h + 1}"] = ([(y, a, b)], py_phm)
        groups[f"H{h + 2}"] = ([(sel(y, near), sel(a, near), sel(b, near))], py_mae)
    key = {k.split()[0]: k for k in T}
    check(sorted(key) == sorted(groups) and len(T) == 8, "exactly the 8 pre-registered tests H1-H8 were run")
    worst, ci_gap = 0.0, 0.0
    for h, (g, metric) in groups.items():
        ys, a_, b_ = (sum((list(x[i]) for x in g), []) for i in range(3))
        worst = max(worst, abs(metric(ys, a_) - metric(ys, b_) - T[key[h]]["diff"]))
        if metric is not py_phm:  # PHM08 is dominated by a few late engines; its interval is checked via Holm only
            lo_, hi_ = boot(g, metric)
            ci_gap = max(ci_gap, abs(lo_ - T[key[h]]["ci"][0]), abs(hi_ - T[key[h]]["ci"][1]))
    check(worst < 1e-9, f"observed gaps of all 8 tests recomputed (max diff {worst:.1e})")
    check(ci_gap < 0.6, f"bootstrap intervals re-drawn with a different generator and seed agree (largest endpoint gap {ci_gap:.2f} cycles)")
    order = sorted(T, key=lambda k: T[k]["p"])
    running, ok = 0.0, True
    for rank, k in enumerate(order):
        running = max(running, min(1.0, T[k]["p"] * (len(order) - rank)))
        ok &= abs(running - T[k]["p_holm"]) < 1e-12 and (running < 0.05) == T[k]["significant"]
    check(ok, "Holm adjustment over the 8 tests recomputed by hand")
    pooled = R["pooled_fd001_fd003"]
    pl = lambda n: py_rmse(y1 + y, list(T1[n]) + list(PREDS[n]))
    pn = lambda n: py_mae(sel(y1, n1) + sel(y, near), sel(T1[n], n1) + sel(PREDS[n], near))
    check(max(abs(pl(n) - pooled[n]["rmse"]) + abs(pn(n) - pooled[n]["near_failure_mae"]) for n in pooled) < 1e-9,
          "pooled FD001+FD003 RMSE and near-failure MAE recomputed")
    h1 = T[key["H1"]]
    wins = h1["significant"] and h1["diff"] < 0 and pooled["LSTM"]["near_failure_mae"] <= pooled[XGB]["near_failure_mae"]
    check(wins == R["verdict"]["deep_wins"] and R["verdict"]["published_winner"] == ("LSTM" if wins else XGB),
          f"verdict follows the pre-registered rule: published winner = {R['verdict']['published_winner']}")
    check(T1.LSTM.tolist() == pd.read_csv("models/predictions_phase5.csv", float_precision="round_trip").LSTM.tolist(),
          "FD001 half of the pooled tests uses Phase 5's published predictions")

    print("\n== Re-training (seed 0)")
    import torch
    from models.deep import predict, train_fixed
    env, lstm = R["environment"], M["LSTM"]
    model, secs = train_fixed(lstm["settings"], 0, X_all, i_all.RUL.to_numpy(float), lstm["epochs_per_seed"][0])
    p = predict(model, X_te)
    same = env["os"] == sys.platform and env["torch"] == torch.__version__
    if same:
        diff = float(np.abs(p - np.array(lstm["seed_test_predictions"][0])).max())
        check(diff == 0.0, f"exact LSTM re-run on {sys.platform}: identical test predictions (max diff {diff:.1e}); {secs:.0f}s")
    else:
        tol = SEED_SD_MULTIPLE * lstm["seed_test_rmse_std"] * math.sqrt(1 + 1 / len(lstm["seed_test_rmse"]))
        got = py_rmse(y, list(p))
        check(abs(got - lstm["seed_test_rmse_mean"]) <= tol, f"LSTM re-trained on {sys.platform}: test RMSE {got:.2f} vs published "
              f"5-seed mean {lstm['seed_test_rmse_mean']:.2f} (allowed {tol:.2f}); {secs:.0f}s")
    from xgboost import XGBRegressor
    from models.baselines import XGB_FIXED, clip
    from models.data import test_frame, train_frame
    full, te = train_frame(DS), test_frame(DS)
    xs = M[XGB]["settings"]
    from features.build import feature_columns
    cols = feature_columns()
    m = XGBRegressor(n_estimators=xs["trees_per_seed"][0], max_depth=xs["max_depth"], random_state=0, **XGB_FIXED)
    px = clip(m.fit(full[cols], full.RUL).predict(te[cols]))
    if env["os"] == sys.platform:
        diff = float(np.abs(px - PREDS[XGB].to_numpy()).max())
        check(diff == 0.0, f"exact XGBoost re-run on {sys.platform}: identical test predictions (max diff {diff:.1e})")
    else:
        tol = SEED_SD_MULTIPLE * M[XGB]["seed_test_rmse_std"] * math.sqrt(1 + 1 / len(M[XGB]["seed_test_rmse"]))
        got = py_rmse(y, list(px))
        check(abs(got - M[XGB]["seed_test_rmse_mean"]) <= max(tol, 0.6), f"XGBoost re-trained on {sys.platform}: test RMSE {got:.2f} vs "
              f"published 5-seed mean {M[XGB]['seed_test_rmse_mean']:.2f} (allowed {max(tol, 0.6):.2f})")

    print("\n== Published documents match the results")
    from models.report_phase5b import DOC, render
    check(DOC.exists() and DOC.read_text(encoding="utf-8") == render(), f"{DOC} equals a fresh rendering of the results")
    readme = Path("README.md").read_text(encoding="utf-8")
    h1 = T[key["H1"]]
    quoted = [f"{M['LSTM']['rmse']:.2f} RMSE", f"{M[XGB]['rmse']:.2f}", f"{abs(h1['diff']):.2f} cycles RMSE",
              f"{-h1['ci'][1]:.2f}-{-h1['ci'][0]:.2f}", f"Holm p {h1['p_holm']:.3f}", f"{pooled['LSTM']['near_failure_mae']:.2f} vs "
              f"{pooled[XGB]['near_failure_mae']:.2f}", f"{mc['FD003']['test_coverage']:.0%} of FD003"]
    missing = [q for q in quoted if q not in readme]
    winner = "the LSTM is now the published model" if R["verdict"]["deep_wins"] else "XGBoost"
    check(not missing and winner in readme, f"README quotes the Phase 5b numbers and the published model ({', '.join(quoted)})"
          + (f"; missing: {', '.join(missing)}" if missing else ""))

    failed = [w for ok, w in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::error title={Path(__file__).stem} crashed::{type(exc).__name__}: {exc}")
        raise
