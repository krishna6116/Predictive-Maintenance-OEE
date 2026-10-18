"""Phase 5: train, select and score the deep sequence models (LSTM, 1D-CNN) on NASA FD001.

Run:  python -m models.run_deep            (about 2-3 hours on one CPU thread)
      python -m models.run_deep --quick    (tiny smoke test of the whole script, ~2 minutes; writes nothing)

Protocol (identical to Phase 4; the test set is scored once, at the end, and decides nothing):
  A. Seed 0, early stopping on the 20 validation engines: model x window x hidden size.
  B. For each model's best A setting: dropout x learning rate.
  C. Each model's final setting with 5 seeds: early stopping on the 80 training engines gives the epoch count;
     refit on all 100 training engines for that many epochs; score the 100 test engines.
  D. Ablation: the better model on windows of the 168 Phase 3 features, 5 seeds.
The published deep model is the one with the lower 5-seed validation RMSE. MC-dropout intervals are calibrated
on validation (leave-engines-out) and then applied to the test set.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features.build import RUL_CAP, SCALER_FILE, SPLIT_FILE, build_features, feature_columns, load_json
from models.deep import THREADS, mc_dropout, predict, train_early_stopping, train_fixed
from models.metrics import bootstrap_ci, holm, mae, paired_bootstrap, phm_score, rmse
from models.sequences import datasets, make_windows
from pipeline.check_raw_data import load
from pipeline.config import RAW

OUT = Path("models")
SEEDS = [0, 1, 2, 3, 4]
WINDOWS, HIDDENS = (20, 30, 40), (32, 64)
DROPOUTS, LRS = (0.1, 0.2, 0.3), (1e-3, 3e-3)
BASE = {"dropout": 0.2, "lr": 1e-3}
MAX_EPOCHS, PATIENCE = 60, 8
MC_PASSES, NOMINAL = 100, 0.90
XGB = "XGBoost, all 168 features"
RIDGE = "Ridge, all 168 features"


def test_like(idx):
    return ((idx.cycle >= 31) & (idx.RUL_true <= 145)).to_numpy()


def score(y, p, y_uncapped) -> dict:
    lo, hi = bootstrap_ci(y, p)
    return {"rmse": rmse(y, p), "rmse_ci95": [lo, hi], "mae": mae(y, p), "phm": phm_score(y, p),
            "rmse_vs_uncapped_truth": rmse(y_uncapped, p), "phm_vs_uncapped_truth": phm_score(y_uncapped, p),
            "late_share": float(np.mean(p > y))}


def feature_windows(length: int):
    """Windows over the 168 Phase 3 features (already causal and scaled), same engines and order as sensors."""
    from models.data import train_frame
    tr = train_frame().sort_values(["unit", "cycle"]).reset_index(drop=True)
    split = load_json(SPLIT_FILE)
    cols = feature_columns()
    X_all, i_all = make_windows(tr, length, cols=cols)
    te = build_features(load(RAW / "test_FD001.txt"), load_json(SCALER_FILE))
    X_te, i_te = make_windows(te, length, cols=cols, last_only=True)
    return {"train": X_all[i_all.unit.isin(split["train"]).to_numpy()],
            "val": X_all[i_all.unit.isin(split["val"]).to_numpy()], "all": X_all, "test": X_te}, i_te


def interval_scale(y, mean, std, groups, nominal=NOMINAL):
    """Factor k so that mean +/- k*std covers `nominal` of validation truths, and a leave-engines-out estimate
    of how well that factor generalises (each engine is scored with k learned on the others)."""
    z = np.abs(y - mean) / np.maximum(std, 1e-6)
    k = float(np.quantile(z, nominal))
    covered = np.zeros(len(y), bool)
    for g in np.unique(groups):
        out = groups == g
        k_g = np.quantile(z[~out], nominal)
        covered[out] = z[out] <= k_g
    return k, float(covered.mean())


def run(quick: bool) -> dict:
    max_epochs, patience = (2, 1) if quick else (MAX_EPOCHS, PATIENCE)
    seeds = SEEDS[:2] if quick else SEEDS
    windows, hiddens = ((20,), (16,)) if quick else (WINDOWS, HIDDENS)
    dropouts, lrs = ((0.2,), (1e-3,)) if quick else (DROPOUTS, LRS)
    passes = 5 if quick else MC_PASSES
    data = {w: datasets(w) for w in windows}
    t_start = time.perf_counter()
    search = []

    def val_run(cfg, seed, d=None):
        d = d or data[cfg["window"]]
        (Xtr, itr), (Xva, iva) = d["train"], d["val"]
        model, info = train_early_stopping(cfg, seed, Xtr, itr.RUL.to_numpy(float), Xva, iva.RUL.to_numpy(float),
                                           max_epochs=max_epochs, patience=patience)
        pv = predict(model, Xva)
        tl = test_like(iva)
        return model, {**info, "val_rmse_testlike": rmse(iva.RUL[tl], pv[tl])}, pv

    # A + B: architecture search on validation, seed 0
    best_cfg = {}
    for arch in ("LSTM", "CNN"):
        rows = []
        for w in windows:
            for h in hiddens:
                cfg = {"arch": arch, "window": w, "hidden": h, **BASE}
                _, info, _ = val_run(cfg, 0)
                rows.append({"stage": "A", **cfg, **{k: info[k] for k in ("val_rmse", "val_rmse_testlike", "best_epoch", "epochs_run", "seconds")}})
                print(f"A {arch} W={w} h={h}: val {info['val_rmse']:.2f} (test-like {info['val_rmse_testlike']:.2f}) epoch {info['best_epoch']} {info['seconds']:.0f}s", flush=True)
        a_best = min(rows, key=lambda r: r["val_rmse"])
        for dr in dropouts:
            for lr in lrs:
                if dr == BASE["dropout"] and lr == BASE["lr"]:
                    continue
                cfg = {"arch": arch, "window": a_best["window"], "hidden": a_best["hidden"], "dropout": dr, "lr": lr}
                _, info, _ = val_run(cfg, 0)
                rows.append({"stage": "B", **cfg, **{k: info[k] for k in ("val_rmse", "val_rmse_testlike", "best_epoch", "epochs_run", "seconds")}})
                print(f"B {arch} dropout={dr} lr={lr}: val {info['val_rmse']:.2f} epoch {info['best_epoch']} {info['seconds']:.0f}s", flush=True)
        final = min(rows, key=lambda r: r["val_rmse"])
        best_cfg[arch] = {k: final[k] for k in ("arch", "window", "hidden", "dropout", "lr")}
        search += rows

    # C: 5 seeds per model on its final setting; refit on all 100 engines for the validated epoch count
    def five_seeds(name, cfg, d):
        (Xva, iva), (Xall, iall), (Xte, ite) = d["val"], d["all"], d["test"]
        runs = []
        for seed in seeds:
            model80, info, pv = val_run(cfg, seed, d)
            model, secs = train_fixed(cfg, seed, Xall, iall.RUL.to_numpy(float), info["best_epoch"])
            pt = predict(model, Xte)
            runs.append({"seed": seed, "epochs": info["best_epoch"], "val_rmse": info["val_rmse"],
                         "val_rmse_testlike": info["val_rmse_testlike"], "test_rmse": rmse(ite.RUL, pt),
                         "seconds_val_run": info["seconds"], "seconds_refit": secs,
                         "val_pred": pv, "test_pred": pt, "model80": model80, "model": model})
            # the smoke test never shows test scores, so nothing can be tuned on them by accident
            shown = "" if quick else f" test {runs[-1]['test_rmse']:.2f}"
            print(f"C {name} seed {seed}: epochs {info['best_epoch']} val {info['val_rmse']:.2f}{shown}", flush=True)
        return runs

    seed_runs = {arch: five_seeds(arch, best_cfg[arch], data[best_cfg[arch]["window"]]) for arch in ("LSTM", "CNN")}
    val_mean = {a: float(np.mean([r["val_rmse"] for r in seed_runs[a]])) for a in seed_runs}
    published = min(val_mean, key=val_mean.get)
    print("published deep model (lower 5-seed validation RMSE):", published, {a: round(v, 3) for a, v in val_mean.items()})

    # D: ablation, the published model's setting on windows of the 168 Phase 3 features
    fcfg = best_cfg[published]
    fx, fi_te = feature_windows(fcfg["window"])
    base = data[fcfg["window"]]
    assert fi_te.unit.tolist() == base["test"][1].unit.tolist()
    fd = {"train": (fx["train"], base["train"][1]), "val": (fx["val"], base["val"][1]),
          "all": (fx["all"], base["all"][1]), "test": (fx["test"], base["test"][1])}
    abl_name = f"{published}, 168-feature windows"
    seed_runs[abl_name] = five_seeds(abl_name, fcfg, fd)

    # MC dropout on the published model, seed 0: calibrate on validation (80-engine model), apply to test (refit)
    d = data[fcfg["window"]]
    (Xva, iva), (Xte, ite) = d["val"], d["test"]
    r0 = seed_runs[published][0]
    vd = mc_dropout(r0["model80"], Xva, passes, seed=0)
    td = mc_dropout(r0["model"], Xte, passes, seed=0)
    yv, yt = iva.RUL.to_numpy(float), ite.RUL.to_numpy(float)
    lo_q, hi_q = (1 - NOMINAL) / 2, 1 - (1 - NOMINAL) / 2
    v_mean, v_std, v_lo, v_hi = vd.mean(0), vd.std(0, ddof=1), np.quantile(vd, lo_q, axis=0), np.quantile(vd, hi_q, axis=0)
    t_mean, t_std, t_lo, t_hi = td.mean(0), td.std(0, ddof=1), np.quantile(td, lo_q, axis=0), np.quantile(td, hi_q, axis=0)
    k, k_cv_cov = interval_scale(yv, v_mean, v_std, iva.unit.to_numpy())
    cal = lambda m, s: (np.clip(m - k * s, 0, RUL_CAP), np.clip(m + k * s, 0, RUL_CAP))
    v_clo, v_chi = cal(v_mean, v_std)
    t_clo, t_chi = cal(t_mean, t_std)
    tlv, near_v, near_t = test_like(iva), iva.RUL_true.to_numpy() <= 50, ite.RUL_true.to_numpy() <= 50
    cover = lambda y, lo, hi, m=None: float(np.mean(((y >= lo) & (y <= hi))[m] if m is not None else (y >= lo) & (y <= hi)))
    mc = {"passes": passes, "nominal": NOMINAL, "model": published, "seed": 0, "scale_k": k,
          "raw": {"val_coverage": cover(yv, v_lo, v_hi), "val_coverage_testlike": cover(yv, v_lo, v_hi, tlv),
                  "test_coverage": cover(yt, t_lo, t_hi), "test_coverage_near_failure": cover(yt, t_lo, t_hi, near_t),
                  "val_mean_width": float(np.mean(v_hi - v_lo)), "test_mean_width": float(np.mean(t_hi - t_lo))},
          "calibrated": {"val_coverage_in_sample": cover(yv, v_clo, v_chi), "val_coverage_leave_engines_out": k_cv_cov,
                         "val_coverage_testlike": cover(yv, v_clo, v_chi, tlv),
                         "val_coverage_near_failure": cover(yv, v_clo, v_chi, near_v),
                         "test_coverage": cover(yt, t_clo, t_chi), "test_coverage_near_failure": cover(yt, t_clo, t_chi, near_t),
                         "val_mean_width": float(np.mean(v_chi - v_clo)), "test_mean_width": float(np.mean(t_chi - t_clo))},
          "mc_mean_test_rmse": rmse(yt, t_mean), "point_test_rmse": rmse(yt, r0["test_pred"])}
    shown = "" if quick else f", test raw {mc['raw']['test_coverage']:.1%} -> calibrated {mc['calibrated']['test_coverage']:.1%}"
    print(f"MC dropout: raw val coverage {mc['raw']['val_coverage']:.1%}, k={k:.2f}, calibrated leave-engines-out "
          f"{k_cv_cov:.1%}{shown}", flush=True)

    # Scores, seed spread, comparisons (Phase 4 predictions on the same 100 test engines)
    p4 = pd.read_csv(OUT / "predictions_phase4.csv", float_precision="round_trip")
    v4 = pd.read_csv(OUT / "val_predictions_phase4.csv", float_precision="round_trip").sort_values(["unit", "cycle"]).reset_index(drop=True)
    assert p4.unit.tolist() == ite.unit.tolist() and (p4.RUL.to_numpy() == yt).all()
    preds = p4[["unit", "RUL_true", "RUL", XGB, RIDGE]].copy()
    vpreds = iva[["unit", "cycle", "RUL", "RUL_true"]].copy()
    assert vpreds[["unit", "cycle"]].equals(v4[["unit", "cycle"]])
    vpreds[XGB], vpreds[RIDGE] = v4[XGB].to_numpy(), v4[RIDGE].to_numpy()
    models = {}
    for name, runs in seed_runs.items():
        p0, tests = runs[0]["test_pred"], [r["test_rmse"] for r in runs]
        preds[name], vpreds[name] = p0, runs[0]["val_pred"]
        cfg = fcfg if name == abl_name else best_cfg[name]
        models[name] = {"settings": cfg, "inputs": "168 features per cycle" if name == abl_name else "14 scaled sensors per cycle",
                        "epochs_per_seed": [r["epochs"] for r in runs],
                        "val_rmse": runs[0]["val_rmse"], "val_rmse_testlike": runs[0]["val_rmse_testlike"],
                        "val_rmse_mean_over_seeds": float(np.mean([r["val_rmse"] for r in runs])),
                        **score(yt, p0, ite.RUL_true.to_numpy(float)),
                        "near_failure_mae": mae(yt[near_t], p0[near_t]),
                        "seed_test_rmse": tests, "seed_test_rmse_mean": float(np.mean(tests)),
                        "seed_test_rmse_std": float(np.std(tests, ddof=1)) if len(tests) > 1 else 0.0,
                        "seed_val_rmse": [r["val_rmse"] for r in runs],
                        "seconds_val_runs": [r["seconds_val_run"] for r in runs], "seconds_refits": [r["seconds_refit"] for r in runs],
                        "seed_test_predictions": [r["test_pred"].tolist() for r in runs]}
    preds["MC mean"], preds["MC std"] = t_mean, t_std
    preds["MC 90% low (raw)"], preds["MC 90% high (raw)"] = t_lo, t_hi
    preds["MC 90% low (calibrated)"], preds["MC 90% high (calibrated)"] = t_clo, t_chi
    vpreds["MC mean"], vpreds["MC std"], vpreds["MC 90% low (raw)"], vpreds["MC 90% high (raw)"] = v_mean, v_std, v_lo, v_hi

    # Validation-based comparison with XGBoost (whole engines resampled), as in Phase 4's model choice
    tl, nv = test_like(iva), near_v
    selection = {f"{published} vs {XGB}": {
        "val_testlike_rmse": paired_bootstrap(yv[tl], vpreds[published][tl], vpreds[XGB][tl], groups=iva.unit[tl]),
        "val_near_failure_mae": paired_bootstrap(yv[nv], vpreds[published][nv], vpreds[XGB][nv], metric=mae, groups=iva.unit[nv])}}

    # Test comparisons, one Holm family (RMSE, PHM08 and near-failure MAE together)
    pairs = [("LSTM", XGB), ("CNN", XGB), ("LSTM", RIDGE), ("CNN", RIDGE), ("LSTM", "CNN"), (abl_name, published)]
    comparisons, pvals = {}, {}
    for a, b in pairs:
        c = {"rmse": paired_bootstrap(yt, preds[a], preds[b]), "phm": paired_bootstrap(yt, preds[a], preds[b], metric=phm_score),
             "near_failure_mae": paired_bootstrap(yt[near_t], preds[a][near_t], preds[b][near_t], metric=mae)}
        comparisons[f"{a} vs {b}"] = c
        for metric, v in c.items():
            pvals[(f"{a} vs {b}", metric)] = v["p"]
    adjusted = holm({f"{k[0]}|{k[1]}": v for k, v in pvals.items()})
    for (name, metric) in pvals:
        comparisons[name][metric].update(adjusted[f"{name}|{metric}"])

    xg_near = mae(yt[near_t], preds[XGB].to_numpy()[near_t])
    c = comparisons[f"{published} vs {XGB}"]
    wins = bool(c["rmse"]["significant"] and c["rmse"]["diff"] < 0 and models[published]["near_failure_mae"] <= xg_near)
    p4res = json.loads((OUT / "results_phase4.json").read_text())
    return {
        "protocol": "settings chosen on 20 validation engines (stages A, B); 5 seeds refit on all 100 training engines for "
                    "the validated epoch count; test set scored once at each engine's last observed cycle; true RUL capped at 125",
        "search_space": {"windows": list(windows), "hiddens": list(hiddens), "dropouts": list(dropouts), "lrs": list(lrs),
                         "base": BASE, "max_epochs": max_epochs, "patience": patience, "batch": 256, "optimizer": "Adam",
                         "loss": "MSE on RUL/125", "padding": "left, copies of the engine's first reading"},
        "search": search, "final_settings": best_cfg, "val_rmse_5seed_mean": val_mean,
        "published_deep_model": published, "models": models, "mc_dropout": mc,
        "model_selection_on_validation": selection, "paired_comparisons": comparisons,
        "baseline": {"name": XGB, "rmse": p4res["models"][XGB]["rmse"], "phm": p4res["models"][XGB]["phm"],
                     "near_failure_mae": xg_near, "seed_test_rmse_mean": p4res["models"][XGB]["seed_test_rmse_mean"]},
        "verdict": {"deep_wins": wins, "published_winner": published if wins else XGB,
                    "rule": "deep model wins only if its paired RMSE gap vs XGBoost is significant after Holm AND its "
                            "near-failure MAE (true RUL <= 50) is not worse than XGBoost's"},
        "environment": {"os": sys.platform, "platform": platform.platform(), "python": platform.python_version(),
                        "torch": torch.__version__, "threads": THREADS, "deterministic": torch.are_deterministic_algorithms_enabled()},
        "seconds_total": time.perf_counter() - t_start,
    }, preds, vpreds, td


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="smoke test with tiny settings; writes nothing")
    a = p.parse_args(argv)
    res, preds, vpreds, draws = run(a.quick)
    if a.quick:
        print(f"quick run OK in {res['seconds_total']:.0f}s (test scores deliberately not shown)")
        return 0
    (OUT / "results_phase5.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    preds.to_csv(OUT / "predictions_phase5.csv", index=False)
    vpreds.to_csv(OUT / "val_predictions_phase5.csv", index=False)
    pd.DataFrame(draws.T, columns=[f"pass_{i}" for i in range(draws.shape[0])]).assign(unit=preds.unit.to_numpy()) \
        .to_csv(OUT / "mc_draws_test_phase5.csv", index=False)
    print(f"\nWrote results_phase5.json, predictions_phase5.csv, val_predictions_phase5.csv, mc_draws_test_phase5.csv "
          f"in {res['seconds_total'] / 60:.0f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
