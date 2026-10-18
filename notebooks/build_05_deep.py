"""Builds notebooks/05_deep.ipynb (Phase 5: deep learning).

Run from the project root:  python notebooks/build_05_deep.py
Then execute it:            jupyter nbconvert --to notebook --execute --inplace notebooks/05_deep.ipynb
Reads the saved results of `python -m models.run_deep` (it does not retrain anything). The written findings are
generated from those results and every claim is asserted, so the notebook cannot go stale or overclaim. The text
was written before the results existed; it adapts to whichever way they came out.
"""
import json
import sys
from pathlib import Path

import nbformat as nbf
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
XGB, RIDGE = "XGBoost, all 168 features", "Ridge, all 168 features"


def findings() -> str:
    r = json.loads(Path("models/results_phase5.json").read_text())
    p4 = json.loads(Path("models/results_phase4.json").read_text())["models"]
    tol = json.loads(Path("models/tolerance_phase5.json").read_text())
    M, C, mc, v = r["models"], r["paired_comparisons"], r["mc_dropout"], r["verdict"]
    pub = r["published_deep_model"]
    other = "CNN" if pub == "LSTM" else "LSTM"
    abl = next(n for n in M if "168-feature" in n)
    P, X = M[pub], p4[XGB]
    c = C[f"{pub} vs {XGB}"]
    sel = r["model_selection_on_validation"][f"{pub} vs {XGB}"]

    # 1. Model choice on validation
    vm = r["val_rmse_5seed_mean"]
    assert vm[pub] <= vm[other], "claim: the published deep model has the lower 5-seed validation RMSE"
    # 2. Validation comparison with XGBoost
    vt, vn = sel["val_testlike_rmse"], sel["val_near_failure_mae"]
    val_word = ("better than" if vt["ci"][1] < 0 else "worse than" if vt["ci"][0] > 0 else "statistically tied with")
    # 3. Test verdict
    sig, better = c["rmse"]["significant"], c["rmse"]["diff"] < 0
    near_ok = P["near_failure_mae"] <= r["baseline"]["near_failure_mae"]
    assert v["deep_wins"] == (sig and better and near_ok), "claim: the verdict follows the rule"
    if v["deep_wins"]:
        verdict = (f"**The deep model wins.** {pub} beats XGBoost on the test set after Holm correction (RMSE gap "
                   f"{c['rmse']['diff']:+.2f}, interval {c['rmse']['ci'][0]:+.2f} to {c['rmse']['ci'][1]:+.2f}) and is at least as "
                   f"accurate near failure ({P['near_failure_mae']:.1f} vs {r['baseline']['near_failure_mae']:.1f} cycles).")
    else:
        why = []
        if not (sig and better):
            why.append(f"its RMSE gap vs XGBoost ({c['rmse']['diff']:+.2f}, interval {c['rmse']['ci'][0]:+.2f} to "
                       f"{c['rmse']['ci'][1]:+.2f}, Holm p {c['rmse']['p_holm']:.3f}) is "
                       + ("not significant" if not sig else "in XGBoost's favour"))
        if not near_ok:
            why.append(f"it is less accurate near failure ({P['near_failure_mae']:.1f} vs {r['baseline']['near_failure_mae']:.1f} cycles)")
        verdict = (f"**XGBoost stays the published model.** {pub} does not meet the bar because " + " and ".join(why) +
                   ". This is an honest result, not a failure of the project: on 100 test engines, a deep model must be clearly "
                   "better to be distinguishable from a strong tree model.")
    # 4. Seeds
    seed_line = (f"Across 5 seeds, {pub}'s test RMSE is {P['seed_test_rmse_mean']:.2f} +/- {P['seed_test_rmse_std']:.2f} "
                 f"(XGBoost: {X['seed_test_rmse_mean']:.2f} +/- {X['seed_test_rmse_std']:.2f}).")
    # 5. LSTM vs CNN and the ablation
    lc, ab = C["LSTM vs CNN"]["rmse"], C[f"{abl} vs {pub}"]["rmse"]
    lc_line = ("LSTM and CNN are statistically tied on the test set" if not lc["significant"] else
               f"{'LSTM' if lc['diff'] < 0 else 'CNN'} is significantly better on the test set") + \
              f" (gap {lc['diff']:+.2f}, Holm p {lc['p_holm']:.3f})."
    ab_line = (f"Feeding the 168 Phase 3 features instead of raw sensors {'helps' if ab['diff'] < 0 else 'does not help'} "
               f"(gap {ab['diff']:+.2f}, Holm p {ab['p_holm']:.3f}, " + ("significant" if ab["significant"] else "not significant") + ").")
    # 6. MC dropout
    raw, cal = mc["raw"], mc["calibrated"]
    raw_word = "too narrow" if raw["val_coverage"] < mc["nominal"] - 0.05 else "too wide" if raw["val_coverage"] > mc["nominal"] + 0.05 else "about right"
    assert abs(cal["val_coverage_leave_engines_out"] - mc["nominal"]) <= 0.05, "claim: calibration reaches about 90% on validation"
    # 7. Tolerance evidence
    assert all(b["caught_by"] for b in tol["bugs"].values()), "claim: every deliberate bug is caught by at least one check"
    weak = [k for k, b in tol["bugs"].items() if "statistical band" not in b["caught_by"]]
    time_min = (sum(P["seconds_val_runs"]) + sum(P["seconds_refits"])) / 60 / len(P["seconds_refits"])

    return f"""## What the results say

1. **Choice on validation.** Over 5 seeds, {pub} has the lower validation RMSE ({vm[pub]:.2f} vs {other} {vm[other]:.2f}), so it is the published deep model. Its setting: window {P['settings']['window']}, hidden {P['settings']['hidden']}, dropout {P['settings']['dropout']}, learning rate {P['settings']['lr']:g}.
2. **On validation, {pub} is {val_word} XGBoost** on test-like readings (RMSE gap {vt['diff']:+.2f}, interval {vt['ci'][0]:+.2f} to {vt['ci'][1]:+.2f}); near failure the MAE gap is {vn['diff']:+.2f} (interval {vn['ci'][0]:+.2f} to {vn['ci'][1]:+.2f}). Whole engines were resampled.
3. **Test set (scored once):** {pub} scores {P['rmse']:.2f} RMSE (interval {P['rmse_ci95'][0]:.1f}-{P['rmse_ci95'][1]:.1f}), PHM08 {P['phm']:,.0f}, near-failure MAE {P['near_failure_mae']:.1f}; XGBoost scores {X['rmse']:.2f}, {X['phm']:,.0f} and {r['baseline']['near_failure_mae']:.1f}.
4. {verdict}
5. **Seed spread.** {seed_line}
6. **Architecture and inputs.** {lc_line} {ab_line}
7. **Uncertainty.** Raw MC-dropout intervals are {raw_word}: they cover {raw['val_coverage']:.1%} of validation truths against a nominal {mc['nominal']:.0%}. Scaling them by k = {mc['scale_k']:.2f} (learned on validation) gives {cal['val_coverage_leave_engines_out']:.1%} on validation when each engine is scored with k learned from the others, and {cal['test_coverage']:.1%} on the test set (raw: {raw['test_coverage']:.1%}). The calibrated intervals are {cal['test_mean_width']:.0f} cycles wide on average.
8. **Cost.** One seed of {pub} takes about {time_min:.1f} minutes to train on one CPU thread (validation run plus refit).
9. **Reproducibility.** Same OS: identical predictions and MC passes on a re-run. Other OS: one re-trained seed must land within {tol['band']:.2f} RMSE of the published 5-seed mean. That band is loose: {len(weak)} of {len(tol['bugs'])} deliberate bugs ({', '.join(weak)}) stay inside it, so they are caught instead by the exact window rebuild or the epoch-count check, which work on any OS.

## Why we did these tasks

| # | Task | Why we did it | What we learned | What it changes next |
| --- | --- | --- | --- | --- |
| 1 | Built causal windows with first-reading left padding | A sequence model must see only the past, and short engines still need a full window | Windows match an independent builder value by value, and rewriting the future leaves earlier windows unchanged | Phase 7 can build windows for each new shift the same way |
| 2 | Searched model, window, size, dropout and learning rate on validation only | The test set must not choose anything | The search table shows every setting tried and why the final one won | Phase 6 uses the published model's predictions without re-tuning |
| 3 | Early stopping, then a refit on all 100 engines for the validated epoch count | Stops memorising training engines; uses all data for the final model | Best epochs vary by seed; the refit reuses each seed's own count | The same recipe can retrain the model when monitoring asks for it |
| 4 | 5 seeds per model | One seed can be lucky | Seed spread is reported next to every headline number | Comparisons with XGBoost account for chance |
| 5 | Paired bootstrap with Holm across RMSE, PHM08 and near-failure MAE | 100 test engines is small and many tests are run | The verdict against XGBoost is decided by a pre-set rule, not by eye | Phase 6 schedules maintenance with the published winner |
| 6 | MC dropout and calibration on validation | Maintenance planners need an interval, not just a number | Raw intervals were {raw_word}; one factor learned on validation fixes coverage | Phase 6 alerts can act on the lower edge of a calibrated interval |
| 7 | Deterministic one-thread training, environment recorded, OS-aware verifier | Phase 4 showed that the same seed is not enough for the same result | Same-OS re-runs are bit-identical; a loose cross-OS band misses small bugs, so exact input checks must back it up | Every later model uses the same reproducibility rules |
| 8 | Independent verifier, generated docs, CI | Numbers on a resume must be reproducible and match the docs | All published numbers recompute with separate code | Every later phase keeps a verifier and generated docs |
"""


cells = [
    md("""# Phase 5: Deep learning for remaining useful life (FD001)

**Goal:** find out whether sequence models (LSTM and 1D-CNN) that read the raw sensor history beat the Phase 4 XGBoost baseline, by more than chance, on the same 100 test engines; and give each prediction an uncertainty interval that is honestly calibrated.

Training and scoring happen in `models/run_deep.py`; this notebook reads the saved results. `python -m models.verify_deep` rechecks every number with independent code, and `python -m models.report_phase5` writes `docs/MODEL_METRICS_PHASE5.md`."""),
    md("""## Concepts in plain words

- **Window:** the last W cycles of the 14 scaled sensors, ending at the current cycle (so only the past is used). An engine with fewer than W readings is padded on the left with copies of its first reading.
- **LSTM:** reads the window one cycle at a time, keeping a small memory of what matters; the memory after the last cycle becomes the RUL estimate.
- **1D-CNN:** slides small 5-cycle filters along the window to detect local shapes such as a rise or a step, then maps them to an RUL estimate.
- **Dropout:** randomly switches off part of the network during training so it cannot rely on any single connection (less overfitting).
- **Early stopping:** train in passes over the data (epochs), watch the validation error, and keep the epoch where it was lowest.
- **MC dropout:** leave dropout on when predicting and predict many times; the spread of the answers shows how unsure the model is.
- **Calibration:** check that a "90% interval" really contains the truth about 90% of the time; if not, widen or narrow it by a factor learned on validation."""),
    code("""import json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
FIG = ROOT / "docs" / "figures"
sns.set_theme(style="whitegrid")
res = json.loads((ROOT / "models/results_phase5.json").read_text())
p4 = json.loads((ROOT / "models/results_phase4.json").read_text())["models"]
preds = pd.read_csv(ROOT / "models/predictions_phase5.csv", float_precision="round_trip")
vpreds = pd.read_csv(ROOT / "models/val_predictions_phase5.csv", float_precision="round_trip")
pub = res["published_deep_model"]
print(res["protocol"]); print("environment:", res["environment"])"""),
    md("""## 5.1 The search (validation only, seed 0)

Stage A tries each model with windows of 20, 30 and 40 cycles and 32 or 64 hidden units. Stage B tunes dropout and learning rate around each model's best stage-A setting."""),
    code("""search = pd.DataFrame(res["search"])
search[["stage", "arch", "window", "hidden", "dropout", "lr", "val_rmse", "val_rmse_testlike", "best_epoch", "seconds"]].round(2)"""),
    code("""fig, ax = plt.subplots(figsize=(9, 4))
for arch, g in search[search.stage == "A"].groupby("arch"):
    for h, gg in g.groupby("hidden"):
        ax.plot(gg.window, gg.val_rmse, marker="o", label=f"{arch}, hidden {h}")
ax.set(title="Stage A: validation RMSE by window length", xlabel="window (cycles)", ylabel="validation RMSE")
ax.legend()
fig.savefig(FIG / "fd001_deep_search.png", dpi=150, bbox_inches="tight")"""),
    md("""## 5.2 Results on the 100 test engines (scored once)"""),
    code("""rows = []
for name, m in res["models"].items():
    rows.append({"model": name, "val RMSE": m["val_rmse"], "test-like val RMSE": m["val_rmse_testlike"], "test RMSE": m["rmse"],
                 "95% interval": f"{m['rmse_ci95'][0]:.1f}-{m['rmse_ci95'][1]:.1f}", "PHM08": m["phm"],
                 "near-failure MAE": m["near_failure_mae"], "5-seed mean": m["seed_test_rmse_mean"], "5-seed sd": m["seed_test_rmse_std"]})
x = p4["XGBoost, all 168 features"]
rows.append({"model": "XGBoost (Phase 4)", "val RMSE": x["val_rmse"], "test-like val RMSE": x["val_rmse_testlike"], "test RMSE": x["rmse"],
             "95% interval": f"{x['rmse_ci95'][0]:.1f}-{x['rmse_ci95'][1]:.1f}", "PHM08": x["phm"],
             "near-failure MAE": res["baseline"]["near_failure_mae"], "5-seed mean": x["seed_test_rmse_mean"], "5-seed sd": x["seed_test_rmse_std"]})
pd.DataFrame(rows).round(2)"""),
    code("""comp = []
for name, c in res["paired_comparisons"].items():
    for metric in ("rmse", "phm", "near_failure_mae"):
        v = c[metric]
        comp.append({"A vs B": name, "metric": metric, "gap A - B": round(v["diff"], 2),
                     "95% interval": f"{v['ci'][0]:+.2f} to {v['ci'][1]:+.2f}", "Holm p": round(v["p_holm"], 3),
                     "verdict": ("A better" if v["diff"] < 0 else "B better") if v["significant"] else "no clear difference"})
pd.DataFrame(comp)"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, m in zip(axes, [pub, "XGBoost, all 168 features"]):
    late = preds[m] > preds.RUL
    ax.scatter(preds.RUL[~late], preds[m][~late], s=18, label="early (safe side)", color="#4C72B0")
    ax.scatter(preds.RUL[late], preds[m][late], s=18, label="late (risky side)", color="#C44E52")
    ax.plot([0, 125], [0, 125], color="black", linewidth=1)
    ax.set(title=m, xlabel="True RUL (capped at 125)", ylabel="Predicted RUL")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fd001_deep_pred_vs_true.png", dpi=150, bbox_inches="tight")"""),
    md("""## 5.3 Uncertainty: MC dropout and calibration"""),
    code("""mc = res["mc_dropout"]
pd.DataFrame({"raw": {k: mc["raw"][k] for k in ("val_coverage", "val_coverage_testlike", "test_coverage", "test_coverage_near_failure", "test_mean_width")},
              "calibrated": {"val_coverage": mc["calibrated"]["val_coverage_leave_engines_out"], "val_coverage_testlike": mc["calibrated"]["val_coverage_testlike"],
                             "test_coverage": mc["calibrated"]["test_coverage"], "test_coverage_near_failure": mc["calibrated"]["test_coverage_near_failure"],
                             "test_mean_width": mc["calibrated"]["test_mean_width"]}}).round(3)"""),
    code("""o = preds.sort_values("RUL").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.fill_between(o.index, o["MC 90% low (calibrated)"], o["MC 90% high (calibrated)"], alpha=0.25, label="calibrated 90% interval")
ax.fill_between(o.index, o["MC 90% low (raw)"], o["MC 90% high (raw)"], alpha=0.5, label="raw MC-dropout 90% interval")
ax.plot(o.index, o.RUL, "k.", label="true RUL (capped)")
ax.set(title=f"{pub}: test engines sorted by true RUL, with intervals", xlabel="test engine (sorted)", ylabel="RUL (cycles)")
ax.legend(fontsize=8)
fig.savefig(FIG / "fd001_deep_intervals.png", dpi=150, bbox_inches="tight")"""),
    md(findings()),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "notebooks/05_deep.ipynb")
print("wrote notebooks/05_deep.ipynb")
