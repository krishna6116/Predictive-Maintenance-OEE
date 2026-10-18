"""Independent verification of Phase 5 (run: python -m models.verify_deep).

Rechecks the published deep-learning results with different code paths: a plain-Python window builder,
pure-Python metrics and quantiles, MC-dropout intervals and coverage recomputed from the saved passes, Holm by
hand, and a re-training check (exact on the producing OS; statistical elsewhere). Exits 1 if any check fails.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features.build import RUL_CAP
from pipeline.config import RAW, SENSORS

RESULTS = json.loads(Path("models/results_phase5.json").read_text())
# round_trip: pandas' default fast float parser can be off in the last binary digit, which would break exact checks
CSV = {"float_precision": "round_trip"}
PREDS = pd.read_csv("models/predictions_phase5.csv", **CSV)
VPREDS = pd.read_csv("models/val_predictions_phase5.csv", **CSV)
DRAWS = pd.read_csv("models/mc_draws_test_phase5.csv", **CSV)
SCALER = json.loads(Path("features/scaler.json").read_text())
SPLIT = json.loads(Path("features/split.json").read_text())
COLS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
XGB = "XGBoost, all 168 features"
checks: list[tuple[bool, str]] = []

# Cross-OS tolerance for one re-trained seed's test RMSE vs the published 5-seed mean (see main for the reasoning)
SEED_SD_MULTIPLE = 4.0


def check(ok, what):
    checks.append((bool(ok), what))
    print(("PASS " if ok else "FAIL ") + what)
    if not ok and os.environ.get("GITHUB_ACTIONS"):
        print(f"::error title={Path(__file__).stem} check failed::{what}")


def read_plain(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append({c: float(v) for c, v in zip(COLS, line.split())})
    return rows


def py_rmse(y, p):
    return math.sqrt(sum((b - a) ** 2 for a, b in zip(y, p)) / len(y))


def py_mae(y, p):
    return sum(abs(b - a) for a, b in zip(y, p)) / len(y)


def py_phm(y, p):
    return sum(math.exp(-(b - a) / 13) - 1 if b - a < 0 else math.exp((b - a) / 10) - 1 for a, b in zip(y, p))


def py_quantile(values, q):
    """Linear interpolation between order statistics (numpy's default), written out."""
    v = sorted(values)
    h = (len(v) - 1) * q
    lo = math.floor(h)
    return v[lo] if lo + 1 >= len(v) else v[lo] + (h - lo) * (v[lo + 1] - v[lo])


def py_std(values):
    m = sum(values) / len(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def plain_windows(rows, length, last_only=False):
    """Second, independent window builder: dictionaries and lists only."""
    by_unit = {}
    for r in rows:
        by_unit.setdefault(int(r["unit"]), []).append(r)
    out = []
    for unit in sorted(by_unit):
        seq = sorted(by_unit[unit], key=lambda r: r["cycle"])
        scaled = [[(r[s] - SCALER[s]["mean"]) / SCALER[s]["std"] for s in SENSORS] for r in seq]
        ends = [len(seq) - 1] if last_only else range(len(seq))
        for t in ends:
            win = [scaled[max(0, i)] for i in range(t - length + 1, t + 1)]  # index < 0 -> first reading
            out.append((unit, int(seq[t]["cycle"]), win))
    return out


def main() -> int:
    R, M = RESULTS, RESULTS["models"]
    pub = R["published_deep_model"]
    cfg = M[pub]["settings"]
    W = cfg["window"]
    train_rows, test_rows = read_plain(RAW / "train_FD001.txt"), read_plain(RAW / "test_FD001.txt")
    truth = [int(x) for x in (RAW / "RUL_FD001.txt").read_text().split()]

    print("== Test targets")
    check(list(PREDS.unit) == list(range(1, 101)) and list(PREDS.RUL_true) == truth, "100 test engines in order; truth = NASA's RUL file")
    check(all(PREDS.RUL == [min(t, RUL_CAP) for t in truth]), "scored target = NASA RUL capped at 125")

    print(f"\n== Windows (published model: {pub}, window {W}) rebuilt by an independent plain-Python builder")
    from models.sequences import datasets
    d = datasets(W)
    (X_all, i_all), (X_te, i_te) = d["all"], d["test"]
    pw = plain_windows(train_rows, W)
    keys_ok = [(u, c) for u, c, _ in pw] == list(zip(i_all.unit, i_all.cycle))
    diff = float(np.abs(np.array([w for _, _, w in pw], dtype=np.float64) - X_all).max())
    check(keys_ok and diff < 1e-4, f"all {len(pw):,} training windows match value by value (max diff {diff:.1e}, float32 storage)")
    pt = plain_windows(test_rows, W, last_only=True)
    diff = float(np.abs(np.array([w for _, _, w in pt], dtype=np.float64) - X_te).max())
    check([u for u, _, _ in pt] == list(range(1, 101)) and diff < 1e-4,
          f"100 test windows end at each engine's last observed cycle and match (max diff {diff:.1e})")
    # Causality: a window ending at cycle t must not change when every later reading is rewritten
    cut = [r for r in train_rows if not (int(r["unit"]) == 1 and r["cycle"] > 50)] + \
          [{**r, **{s: r[s] + 999 for s in SENSORS}} for r in train_rows if int(r["unit"]) == 1 and r["cycle"] > 50]
    pc = plain_windows(cut, W)
    same = all(a[2] == b[2] for a, b in zip(pw, pc) if a[0] == 1 and a[1] <= 50)
    check(same, "causal: rewriting engine 1's readings after cycle 50 leaves its windows up to cycle 50 unchanged")
    first = next(w for u, c, w in pw if u == 1 and c == 1)
    check(all(row == first[-1] for row in first), "left padding repeats the engine's first reading (cycle 1 window)")
    short = [r for r in test_rows if int(r["unit"]) == 1 and r["cycle"] <= 10]
    w10 = plain_windows(short, W, last_only=True)[0][2]
    check(all(row == w10[0] for row in w10[:W - 10]), f"an engine with 10 readings is padded with its first reading ({W - 10} copies)")
    check(len(X_all) == len(train_rows) and set(i_all.unit) == set(range(1, 101)),
          "models are trained on windows from the training file only (no test data)")
    check(sorted(VPREDS.unit.unique()) == sorted(SPLIT["val"]), "validation predictions cover exactly the 20 validation engines")

    print("\n== Every published metric, recomputed with plain Python")
    y, yr = list(PREDS.RUL), list(PREDS.RUL_true)
    near = [i for i, t in enumerate(PREDS.RUL_true) if t <= 50]
    worst = 0.0
    for name, r in M.items():
        p = list(PREDS[name])
        for key, val in (("rmse", py_rmse(y, p)), ("mae", py_mae(y, p)), ("phm", py_phm(y, p)),
                         ("rmse_vs_uncapped_truth", py_rmse(yr, p)), ("phm_vs_uncapped_truth", py_phm(yr, p)),
                         ("near_failure_mae", py_mae([y[i] for i in near], [p[i] for i in near]))):
            worst = max(worst, abs(val - r[key]) / max(1.0, abs(val)))
        seeds = r["seed_test_predictions"]
        rm = [py_rmse(y, s) for s in seeds]
        mean = sum(rm) / len(rm)
        worst = max(worst, abs(mean - r["seed_test_rmse_mean"]), abs(py_std(rm) - r["seed_test_rmse_std"]),
                    max(abs(a - b) for a, b in zip(seeds[0], p)))
        if not r["rmse_ci95"][0] <= r["rmse"] <= r["rmse_ci95"][1]:
            worst = float("inf")
    check(worst < 1e-9, f"RMSE, MAE, PHM08 (capped, uncapped), near-failure MAE, seed mean/sd and seed-0 = published for "
                        f"{len(M)} models; max relative diff {worst:.1e}")

    print("\n== Every choice was made on validation")
    ok = True
    for arch, final in R["final_settings"].items():
        rows = [s for s in R["search"] if s["arch"] == arch]
        best = min(rows, key=lambda s: s["val_rmse"])
        ok &= all(best[k] == final[k] for k in final)
        a_best = min((s for s in rows if s["stage"] == "A"), key=lambda s: s["val_rmse"])
        ok &= all(s["window"] == a_best["window"] and s["hidden"] == a_best["hidden"] for s in rows if s["stage"] == "B")
    check(ok, "each model's final setting is the validation minimum of its search; stage B tunes stage A's winner")
    means = {a: sum(M[a]["seed_val_rmse"]) / len(M[a]["seed_val_rmse"]) for a in ("LSTM", "CNN")}
    check(pub == min(means, key=means.get) and all(abs(means[a] - R["val_rmse_5seed_mean"][a]) < 1e-12 for a in means),
          f"published deep model = lower 5-seed validation RMSE ({', '.join(f'{a} {v:.2f}' for a, v in means.items())})")
    vp = VPREDS.sort_values(["unit", "cycle"])
    tl = (vp.cycle >= 31) & (vp.RUL_true <= 145)
    got = M[pub]["val_rmse_testlike"]
    check(abs(py_rmse(list(vp.RUL[tl]), list(vp[pub][tl])) - got) < 1e-9 and abs(py_rmse(list(vp.RUL), list(vp[pub])) - M[pub]["val_rmse"]) < 1e-9,
          f"validation RMSE (all rows {M[pub]['val_rmse']:.2f}, test-like {got:.2f}) recomputes from the saved validation predictions")

    print("\n== MC dropout: intervals and coverage recomputed from the saved passes")
    mc = R["mc_dropout"]
    passes = [c for c in DRAWS.columns if c.startswith("pass_")]
    check(len(passes) == mc["passes"] and list(DRAWS.unit) == list(range(1, 101)), f"{len(passes)} saved passes for each of the 100 test engines")
    q_lo, q_hi = (1 - mc["nominal"]) / 2, 1 - (1 - mc["nominal"]) / 2
    worst = 0.0
    lo_raw, hi_raw, mean_t, std_t = [], [], [], []
    for _, row in DRAWS.iterrows():
        v = [row[c] for c in passes]
        mean_t.append(sum(v) / len(v)); std_t.append(py_std(v))
        lo_raw.append(py_quantile(v, q_lo)); hi_raw.append(py_quantile(v, q_hi))
    for col, val in (("MC mean", mean_t), ("MC std", std_t), ("MC 90% low (raw)", lo_raw), ("MC 90% high (raw)", hi_raw)):
        worst = max(worst, max(abs(a - b) for a, b in zip(PREDS[col], val)))
    k = mc["scale_k"]
    clo = [min(max(m - k * s, 0), RUL_CAP) for m, s in zip(mean_t, std_t)]
    chi = [min(max(m + k * s, 0), RUL_CAP) for m, s in zip(mean_t, std_t)]
    worst = max(worst, max(abs(a - b) for a, b in zip(PREDS["MC 90% low (calibrated)"], clo)),
                max(abs(a - b) for a, b in zip(PREDS["MC 90% high (calibrated)"], chi)))
    check(worst < 1e-9, f"test MC mean, sd, raw 5-95% interval and calibrated interval (k={k:.3f}) recompute; max diff {worst:.1e}")

    def cover(t, lo, hi, idx=None):
        idx = range(len(t)) if idx is None else idx
        return sum(lo[i] <= t[i] <= hi[i] for i in idx) / len(idx)
    yv, vm, vs = list(vp.RUL), list(vp["MC mean"]), list(vp["MC std"])
    z = [abs(a - m) / max(s, 1e-6) for a, m, s in zip(yv, vm, vs)]
    k_re = py_quantile(z, mc["nominal"])
    units = list(vp.unit)
    covered = 0  # leave-engines-out: each validation engine is scored with k learned on the other 19
    for u in sorted(set(units)):
        others = [zz for zz, uu in zip(z, units) if uu != u]
        k_u = py_quantile(others, mc["nominal"])
        covered += sum(1 for zz, uu in zip(z, units) if uu == u and zz <= k_u)
    loo_cov = covered / len(z)
    raw_val = cover(yv, list(vp["MC 90% low (raw)"]), list(vp["MC 90% high (raw)"]))
    raw_test = cover(y, lo_raw, hi_raw)
    cal_test = cover(y, clo, chi)
    nt = [i for i, t in enumerate(PREDS.RUL_true) if t <= 50]
    ok = (abs(k_re - k) < 1e-9 and abs(loo_cov - mc["calibrated"]["val_coverage_leave_engines_out"]) < 1e-12 and
          abs(raw_val - mc["raw"]["val_coverage"]) < 1e-12 and abs(raw_test - mc["raw"]["test_coverage"]) < 1e-12 and
          abs(cal_test - mc["calibrated"]["test_coverage"]) < 1e-12 and
          abs(cover(y, clo, chi, nt) - mc["calibrated"]["test_coverage_near_failure"]) < 1e-12)
    check(ok, f"coverage of the nominal 90% interval recomputes: validation raw {raw_val:.1%} -> calibrated "
              f"(leave-engines-out) {loo_cov:.1%}; test raw {raw_test:.1%} -> calibrated {cal_test:.1%}")
    check(abs(loo_cov - mc["nominal"]) <= 0.05, f"calibrated validation coverage {loo_cov:.1%} is within 5 points of the nominal 90%")

    print("\n== Comparisons, Holm correction and the verdict")
    comp = R["paired_comparisons"]
    ok = True
    for name, c in comp.items():
        a, b = name.split(" vs ")
        for metric, fn, idx in (("rmse", py_rmse, None), ("phm", py_phm, None), ("near_failure_mae", py_mae, nt)):
            idx = list(range(100)) if idx is None else idx
            expected = fn([y[i] for i in idx], [PREDS[a][i] for i in idx]) - fn([y[i] for i in idx], [PREDS[b][i] for i in idx])
            lo, hi = c[metric]["ci"]
            ok &= abs(expected - c[metric]["diff"]) < 1e-6 * max(1, abs(expected)) and lo <= expected <= hi
    check(ok, f"{len(comp)} comparisons x 3 metrics: every gap recomputes and lies inside its interval")
    ps = sorted((c[m]["p"], f"{n}|{m}") for n, c in comp.items() for m in c)
    running, ok = 0.0, True
    for rank, (p_, key) in enumerate(ps):
        running = max(running, min(1.0, p_ * (len(ps) - rank)))
        n, m = key.split("|")
        ok &= abs(comp[n][m]["p_holm"] - running) < 1e-12 and comp[n][m]["significant"] == (running < 0.05)
    check(ok, f"Holm over all {len(ps)} tests (one family) recomputes, with verdicts")
    c = comp[f"{pub} vs {XGB}"]
    xg_near = py_mae([y[i] for i in nt], [PREDS[XGB][i] for i in nt])
    wins = c["rmse"]["significant"] and c["rmse"]["diff"] < 0 and M[pub]["near_failure_mae"] <= xg_near
    check(wins == R["verdict"]["deep_wins"] and R["verdict"]["published_winner"] == (pub if wins else XGB),
          f"verdict follows the rule: deep model wins = {wins}; published winner = {R['verdict']['published_winner']}")
    p4 = pd.read_csv("models/predictions_phase4.csv", **CSV)
    check(np.array_equal(p4[XGB].to_numpy(), PREDS[XGB].to_numpy()), "XGBoost predictions compared are exactly Phase 4's published ones")

    print("\n== Re-training the published model (seed 0)")
    import torch
    from models.deep import THREADS, mc_dropout, predict, train_fixed
    env = R["environment"]
    r = M[pub]
    model, secs = train_fixed(cfg, 0, X_all, i_all.RUL.to_numpy(float), r["epochs_per_seed"][0])
    p = predict(model, X_te)
    check(env["threads"] == THREADS == 1 and env["deterministic"], "published run used one fixed thread and deterministic algorithms")
    if env["os"] == sys.platform and env["torch"] == torch.__version__:
        diff = float(np.abs(p - np.array(r["seed_test_predictions"][0])).max())
        draws = mc_dropout(model, X_te, mc["passes"], seed=0)
        ddiff = float(np.abs(draws.T - DRAWS[passes].to_numpy()).max())
        check(diff == 0.0 and ddiff == 0.0, f"exact re-run on {sys.platform}: identical test predictions (max diff {diff:.1e}) "
                                             f"and identical MC passes (max diff {ddiff:.1e}); refit took {secs:.0f}s")
    else:
        # Different OS or library: floating-point kernels differ, so an exact match cannot be expected. One
        # re-trained seed is compared with the published seed distribution. A single seed sits within
        # SEED_SD_MULTIPLE published seed sds of the 5-seed mean by chance with overwhelming probability. The band
        # is loose, so input bugs are left to the exact window rebuild above (see models/tolerance_phase5.py).
        sd, mean = r["seed_test_rmse_std"], r["seed_test_rmse_mean"]
        tol = SEED_SD_MULTIPLE * sd * math.sqrt(1 + 1 / len(r["seed_test_rmse"]))
        got = py_rmse(y, list(p))
        print(f"note: published on {env['os']} with torch {env['torch']}; this is {sys.platform} with torch "
              f"{torch.__version__}, so the exact check is replaced by a statistical one")
        check(abs(got - mean) <= tol, f"re-trained seed 0 test RMSE {got:.3f} vs published 5-seed mean {mean:.3f} "
                                      f"(gap {got - mean:+.3f}, allowed {tol:.3f}); refit took {secs:.0f}s")

    # Epoch consistency: the refit must use the epoch count the validation search found for this setting (seed 0)
    row = next(s for s in R["search"] if all(s[k] == cfg[k] for k in ("arch", "window", "hidden", "dropout", "lr")))
    check(r["epochs_per_seed"][0] == row["best_epoch"] and abs(r["seed_val_rmse"][0] - row["val_rmse"]) < 1e-12,
          f"seed 0 refit uses the validated epoch count ({row['best_epoch']}) and its validation RMSE matches the search")

    # Which check catches which deliberate bug (measured by models.tolerance_phase5 on validation engines)
    tolf = json.loads(Path("models/tolerance_phase5.json").read_text())
    tol_pub = SEED_SD_MULTIPLE * r["seed_test_rmse_std"] * math.sqrt(1 + 1 / len(r["seed_test_rmse"]))
    uncaught = [k for k, v in tolf["bugs"].items() if not v["caught_by"]]
    check(tolf["model"] == pub and tolf["settings"] == cfg and abs(tolf["band"] - tol_pub) < 1e-12
          and abs(tolf["correct_validation_rmse"] - r["val_rmse"]) < 1e-9 and not uncaught,
          f"every one of {len(tolf['bugs'])} deliberate bugs is caught by at least one check that also works on other operating systems"
          + (f"; uncaught: {', '.join(uncaught)}" if uncaught else ""))

    print("\n== Published documents match the results")
    from models.report_phase5 import DOC, render
    check(DOC.read_text(encoding="utf-8") == render(), f"{DOC} equals a fresh rendering of the results")
    readme = Path("README.md").read_text(encoding="utf-8")
    quoted = [f"{r['rmse']:.2f} RMSE", f"{r['rmse_ci95'][0]:.1f}-{r['rmse_ci95'][1]:.1f}",
              f"{mc['calibrated']['test_coverage']:.0%}", f"{mc['raw']['val_coverage']:.0%}"]
    missing = [q for q in quoted if q not in readme]
    check(not missing, f"README quotes the Phase 5 numbers ({', '.join(quoted)})" + (f"; missing: {', '.join(missing)}" if missing else ""))

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
