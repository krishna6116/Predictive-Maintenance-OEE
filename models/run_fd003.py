"""Phase 5b: confirm the FD001 models on NASA FD003, a fresh dataset nobody has scored yet.

Run:  python -m models.run_fd003          (about 45-60 minutes on one CPU thread)

Why: FD001's 100 test engines were already used, and they were too few to separate the LSTM from XGBoost.
FD003 adds 100 new test engines and a second fault mode (fan degradation as well as HPC degradation).

Frozen from FD001 (nothing is re-chosen on FD003): the 14 sensors, the 168 Phase 3 features, RUL cap 125,
an 80/20 engine split (same function and seed), XGBoost depth 5 and its other settings, the LSTM and CNN
settings found by the Phase 5 search, and the interval rule in models/calibration_rule.json (committed before
this script was run). Learned on FD003's own 20 validation engines, with the same procedures as on FD001:
the scaler (training engines only), XGBoost tree count and network epoch counts (early stopping), the Ridge
penalty, and the two interval scale factors.

Pre-registered tests (one Holm family; the test set is scored once, at the end):
  H1 (primary)  LSTM vs XGBoost, RMSE, FD001 + FD003 pooled (200 engines; engines resampled within each dataset)
  H2            LSTM vs XGBoost, near-failure MAE (true RUL <= 50), pooled
  H3-H5         LSTM vs XGBoost on FD003 alone: RMSE, PHM08, near-failure MAE
  H6-H8         CNN vs XGBoost on FD003 alone: RMSE, PHM08, near-failure MAE
Verdict rule (same as Phase 5, applied to the pooled evidence): the LSTM replaces XGBoost as the published
model only if H1 is significant after Holm with LSTM better, AND its pooled near-failure MAE is not worse.
Interval check: the 90% interval from rule E is scored on FD003 test (pre-registered) and, labelled post hoc,
on FD001 test (which had already been seen).

Outputs: models/results_phase5b.json, predictions_phase5b.csv (FD003 test), val_predictions_phase5b.csv,
mc_draws_test_phase5b.csv; features/split_FD003.json, features/scaler_FD003.json.
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from features.build import (RUL_CAP, add_rul, feature_columns, fit_scaler, load_json, save_json, scaler_file,
                            split_engines, split_file)
from models.baselines import XGB_FIXED, age_model, clip, score, tune_ridge, xgb_val
from models.calibration import NEAR, test_like, zone_interval, zone_scales
from models.data import test_frame, train_frame
from models.deep import THREADS, mc_dropout, predict, train_early_stopping, train_fixed
from models.metrics import holm, mae, paired_bootstrap, phm_score, rmse
from models.sequences import datasets
from pipeline.check_raw_data import load
from pipeline.config import RAW

DATASET = "FD003"
OUT = Path("models")
SEEDS = [0, 1, 2, 3, 4]
MAX_EPOCHS, PATIENCE = 60, 8   # Phase 5 values
MC_PASSES = 100
XGB = "XGBoost, all 168 features"
RIDGE = "Ridge, all 168 features"
ALL = feature_columns()


def ensure_split_and_scaler(dataset: str) -> dict:
    """Phase 3 recipe for a new dataset: engine split (seed 42), scaler fit on training engines only."""
    raw = load(RAW / f"train_{dataset}.txt")
    if not split_file(dataset).exists():
        split = split_engines(raw.unit.unique())
        save_json(split, split_file(dataset))
        save_json(fit_scaler(raw, split["train"]), scaler_file(dataset))
    return load_json(split_file(dataset))


def stratified_paired_bootstrap(groups: list[tuple], metric=rmse, n: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap over several datasets at once: engines are resampled WITHIN each dataset, so every
    resample keeps 100 FD001 and 100 FD003 engines. groups = [(y, pred_a, pred_b), ...]."""
    rng = np.random.default_rng(seed)
    data = [tuple(np.asarray(v, float) for v in g) for g in groups]
    cat = lambda parts: tuple(np.concatenate(x) for x in zip(*parts))
    y, a, b = cat(data)
    diffs = []
    for _ in range(n):
        ys, as_, bs = cat([(g[0][i], g[1][i], g[2][i]) for g in data for i in [rng.integers(0, len(g[0]), len(g[0]))]])
        diffs.append(metric(ys, as_) - metric(ys, bs))
    diffs = np.array(diffs)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    wins = float((diffs < 0).mean())
    return {"diff": metric(y, a) - metric(y, b), "ci": (float(lo), float(hi)), "a_wins": wins, "p": float(max(2 * min(wins, 1 - wins), 1 / n))}


def main() -> int:
    t_start = time.perf_counter()
    p5 = json.loads((OUT / "results_phase5.json").read_text())
    p4 = json.loads((OUT / "results_phase4.json").read_text())
    rule = json.loads((OUT / "calibration_rule.json").read_text())
    assert rule["chosen"] == "E" and rule["near_zone_cycles"] == NEAR
    split = ensure_split_and_scaler(DATASET)

    # ---- Baselines on the 168 features (Phase 4 code, frozen settings)
    full, te = train_frame(DATASET), test_frame(DATASET)
    tr, va = full[full.split == "train"], full[full.split == "val"]
    assert sorted(va.unit.unique().tolist()) == sorted(split["val"])
    y, y_raw = te.RUL.to_numpy(float), te.RUL_true.to_numpy(float)
    print(f"{DATASET}: train {len(tr):,} rows / {tr.unit.nunique()} engines | val {len(va):,} rows / "
          f"{va.unit.nunique()} engines | test {len(te)} engines", flush=True)
    preds = pd.DataFrame({"unit": te.unit, "RUL_true": y_raw, "RUL": y})
    vpreds = va[["unit", "cycle", "RUL", "RUL_true"]].sort_values(["unit", "cycle"]).reset_index(drop=True)
    va = va.sort_values(["unit", "cycle"]).reset_index(drop=True)
    tl = test_like(vpreds.cycle, vpreds.RUL_true)
    val_scores = lambda p: {"val_rmse": rmse(vpreds.RUL, p), "val_rmse_testlike": rmse(vpreds.RUL[tl], np.asarray(p)[tl])}
    models = {}

    avg = float(full.RUL.mean())
    vpreds["Fleet average"], preds["Fleet average"] = float(tr.RUL.mean()), avg
    models["Fleet average"] = {"settings": {"constant": avg}, **val_scores(vpreds["Fleet average"]), **score(y, preds["Fleet average"].to_numpy(), y_raw)}
    age = "Age only (cycle count)"
    vpreds[age], preds[age] = age_model(tr)(va.cycle.to_numpy()), age_model(full)(te.cycle.to_numpy())
    models[age] = {"settings": {"rule": "mean capped RUL of training readings at the same cycle"},
                   **val_scores(vpreds[age]), **score(y, preds[age].to_numpy(), y_raw)}

    t = tune_ridge(tr, va, ALL)
    vpreds[RIDGE] = clip(Ridge(alpha=t["alpha"]).fit(tr[ALL], tr.RUL).predict(va[ALL]))
    preds[RIDGE] = clip(Ridge(alpha=t["alpha"]).fit(full[ALL], full.RUL).predict(te[ALL]))
    models[RIDGE] = {"settings": {"alpha": t["alpha"], "val_rmse_by_alpha": t["val_rmse_by_alpha"]},
                     **val_scores(vpreds[RIDGE]), **score(y, preds[RIDGE].to_numpy(), y_raw)}
    print(f"{RIDGE}: alpha {t['alpha']}, val {models[RIDGE]['val_rmse']:.2f}", flush=True)

    depth = p4["models"][XGB]["settings"]["max_depth"]
    runs = []
    for seed in SEEDS:
        n_trees, v, pv = xgb_val(tr, va, ALL, depth, seed)
        m = XGBRegressor(n_estimators=n_trees, max_depth=depth, random_state=seed, **XGB_FIXED).fit(full[ALL], full.RUL)
        runs.append({"trees": n_trees, "val_rmse": v, "vpred": pv, "pred": clip(m.predict(te[ALL]))})
        print(f"{XGB} seed {seed}: {n_trees} trees, val {v:.2f}", flush=True)
    vpreds[XGB], preds[XGB] = runs[0]["vpred"], runs[0]["pred"]
    tests = [rmse(y, r["pred"]) for r in runs]
    models[XGB] = {"settings": {"max_depth": depth, **XGB_FIXED, "trees_per_seed": [r["trees"] for r in runs]},
                   **val_scores(vpreds[XGB]), "val_rmse_mean_over_seeds": float(np.mean([r["val_rmse"] for r in runs])),
                   **score(y, preds[XGB].to_numpy(), y_raw), "seed_test_rmse": tests,
                   "seed_test_rmse_mean": float(np.mean(tests)), "seed_test_rmse_std": float(np.std(tests, ddof=1)),
                   "seed_test_predictions": [r["pred"].tolist() for r in runs]}

    # ---- Deep models with the Phase 5 settings (no search on FD003)
    deep_runs = {}
    for arch in ("LSTM", "CNN"):
        cfg = p5["final_settings"][arch]
        d = datasets(cfg["window"], DATASET)
        (Xtr, itr), (Xva, iva), (Xall, iall), (Xte, ite) = d["train"], d["val"], d["all"], d["test"]
        assert iva[["unit", "cycle"]].equals(vpreds[["unit", "cycle"]]) and ite.unit.tolist() == preds.unit.tolist()
        runs = []
        for seed in SEEDS:
            m80, info = train_early_stopping(cfg, seed, Xtr, itr.RUL.to_numpy(float), Xva, iva.RUL.to_numpy(float),
                                             MAX_EPOCHS, PATIENCE)
            model, secs = train_fixed(cfg, seed, Xall, iall.RUL.to_numpy(float), info["best_epoch"])
            runs.append({"epochs": info["best_epoch"], "val_rmse": info["val_rmse"], "vpred": predict(m80, Xva),
                         "pred": predict(model, Xte), "m80": m80, "model": model,
                         "seconds": info["seconds"] + secs})
            print(f"{arch} seed {seed}: {info['best_epoch']} epochs, val {info['val_rmse']:.2f} "
                  f"({runs[-1]['seconds']:.0f}s)", flush=True)  # test scores are not printed during training
        deep_runs[arch] = (cfg, runs, Xva, Xte)
        vpreds[arch], preds[arch] = runs[0]["vpred"], runs[0]["pred"]
        tests = [rmse(y, r["pred"]) for r in runs]
        models[arch] = {"settings": cfg, "inputs": "14 scaled sensors per cycle", "epochs_per_seed": [r["epochs"] for r in runs],
                        **val_scores(vpreds[arch]), "seed_val_rmse": [r["val_rmse"] for r in runs],
                        "val_rmse_mean_over_seeds": float(np.mean([r["val_rmse"] for r in runs])),
                        **score(y, preds[arch].to_numpy(), y_raw), "seed_test_rmse": tests,
                        "seed_test_rmse_mean": float(np.mean(tests)), "seed_test_rmse_std": float(np.std(tests, ddof=1)),
                        "seed_test_predictions": [r["pred"].tolist() for r in runs],
                        "seconds_per_seed": [r["seconds"] for r in runs]}

    near = y_raw <= 50
    for name in models:
        models[name]["near_failure_mae"] = mae(y[near], preds[name].to_numpy()[near])

    # ---- Intervals: rule E, scale factors learned on FD003 validation (80-engine model), applied to the refit model
    cfg, runs, Xva, Xte = deep_runs["LSTM"]
    vd, td = mc_dropout(runs[0]["m80"], Xva, MC_PASSES, seed=0), mc_dropout(runs[0]["model"], Xte, MC_PASSES, seed=0)
    v_mean, v_std, t_mean, t_std = vd.mean(0), vd.std(0, ddof=1), td.mean(0), td.std(0, ddof=1)
    vpreds["MC mean"], vpreds["MC std"] = v_mean, v_std
    k_near, k_far = zone_scales(vpreds.RUL, v_mean, v_std, vpreds.cycle, vpreds.RUL_true)
    lo, hi = zone_interval(t_mean, t_std, k_near, k_far)
    zv = np.abs(vpreds.RUL.to_numpy() - v_mean) / np.maximum(v_std, 1e-6)
    k_a = float(np.quantile(zv, rule["nominal"]))  # Phase 5 rule, for contrast
    cov = lambda lo_, hi_, m=slice(None): float(np.mean(((y >= lo_) & (y <= hi_))[m]))
    preds["MC mean"], preds["MC std"], preds["MC 90% low (rule E)"], preds["MC 90% high (rule E)"] = t_mean, t_std, lo, hi

    # Post hoc (labelled): rule E on FD001 test, from the saved Phase 5 validation and test MC passes
    v1 = pd.read_csv(OUT / "val_predictions_phase5.csv", float_precision="round_trip")
    t1 = pd.read_csv(OUT / "predictions_phase5.csv", float_precision="round_trip")
    k1n, k1f = zone_scales(v1.RUL, v1["MC mean"], v1["MC std"], v1.cycle, v1.RUL_true)
    lo1, hi1 = zone_interval(t1["MC mean"], t1["MC std"], k1n, k1f)
    y1 = t1.RUL.to_numpy(float)
    c1 = lambda m=slice(None): float(np.mean(((y1 >= lo1) & (y1 <= hi1))[m]))
    mc = {"rule": "E (models/calibration_rule.json)", "passes": MC_PASSES, "model": "LSTM", "seed": 0,
          "FD003": {"k_near": k_near, "k_far": k_far, "test_coverage": cov(lo, hi), "test_coverage_near_failure": cov(lo, hi, near),
                    "test_mean_width": float(np.mean(hi - lo)),
                    "phase5_rule_k": k_a, "phase5_rule_test_coverage": cov(np.maximum(t_mean - k_a * t_std, 0), t_mean + k_a * t_std),
                    "phase5_rule_test_coverage_near_failure": cov(np.maximum(t_mean - k_a * t_std, 0), t_mean + k_a * t_std, near),
                    "raw_test_coverage": cov(np.quantile(td, 0.05, axis=0), np.quantile(td, 0.95, axis=0))},
          "FD001_post_hoc": {"note": "FD001 test coverage had already been seen in Phase 5, so this is not a confirmation",
                             "k_near": k1n, "k_far": k1f, "test_coverage": c1(), "test_coverage_near_failure": c1(t1.RUL_true.to_numpy() <= 50),
                             "test_mean_width": float(np.mean(hi1 - lo1))}}

    # ---- Pre-registered comparisons (one Holm family)
    n1 = t1.RUL_true.to_numpy() <= 50
    pooled = lambda col, mask1=slice(None), mask3=slice(None): [
        (y1[mask1], t1["LSTM"].to_numpy()[mask1], t1[XGB].to_numpy()[mask1]),
        (y[mask3], preds[col].to_numpy()[mask3], preds[XGB].to_numpy()[mask3])]
    tests_ = {
        "H1 LSTM vs XGBoost, RMSE, FD001+FD003": stratified_paired_bootstrap(pooled("LSTM")),
        "H2 LSTM vs XGBoost, near-failure MAE, FD001+FD003": stratified_paired_bootstrap(pooled("LSTM", n1, near), metric=mae),
    }
    for arch, h in (("LSTM", 3), ("CNN", 6)):
        a, b = preds[arch].to_numpy(), preds[XGB].to_numpy()
        tests_[f"H{h} {arch} vs XGBoost, RMSE, FD003"] = paired_bootstrap(y, a, b)
        tests_[f"H{h + 1} {arch} vs XGBoost, PHM08, FD003"] = paired_bootstrap(y, a, b, metric=phm_score)
        tests_[f"H{h + 2} {arch} vs XGBoost, near-failure MAE, FD003"] = paired_bootstrap(y[near], a[near], b[near], metric=mae)
    adj = holm({k: v["p"] for k, v in tests_.items()})
    for k in tests_:
        tests_[k].update(adj[k])
    pooled_scores = {name: {"rmse": rmse(np.r_[y1, y], np.r_[t1[name], preds[name]]),
                            "near_failure_mae": mae(np.r_[y1[n1], y[near]], np.r_[t1[name].to_numpy()[n1], preds[name].to_numpy()[near]])}
                     for name in ("LSTM", "CNN", XGB)}
    h1 = tests_["H1 LSTM vs XGBoost, RMSE, FD001+FD003"]
    wins = h1["significant"] and h1["diff"] < 0 and pooled_scores["LSTM"]["near_failure_mae"] <= pooled_scores[XGB]["near_failure_mae"]
    verdict = {"deep_wins": bool(wins), "published_winner": "LSTM" if wins else XGB,
               "rule": "LSTM replaces XGBoost only if H1 is significant after Holm with LSTM better AND its pooled "
                       "near-failure MAE is not worse"}

    out = {"dataset": DATASET, "protocol": __doc__.split("Pre-registered")[0].strip(),
           "split": {"train": len(split["train"]), "val": len(split["val"])},
           "frozen_from_fd001": {"xgb_depth": depth, "deep_settings": p5["final_settings"], "calibration_rule": "E"},
           "models": models, "mc_dropout": mc, "tests": tests_, "pooled_fd001_fd003": pooled_scores, "verdict": verdict,
           "environment": {"os": sys.platform, "platform": platform.platform(), "python": platform.python_version(),
                           "torch": torch.__version__, "xgboost": xgboost.__version__, "threads": THREADS, "deterministic": True},
           "seconds_total": time.perf_counter() - t_start}
    (OUT / "results_phase5b.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    preds.to_csv(OUT / "predictions_phase5b.csv", index=False)
    vpreds.to_csv(OUT / "val_predictions_phase5b.csv", index=False)
    pd.DataFrame(td.T, columns=[f"pass_{i}" for i in range(MC_PASSES)]).assign(unit=te.unit.to_numpy()) \
        .to_csv(OUT / "mc_draws_test_phase5b.csv", index=False)
    print(f"Done in {out['seconds_total'] / 60:.0f} min. Verdict: {verdict['published_winner']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
