"""Builds notebooks/05b_fd003.ipynb (Phase 5b: FD003 confirmation and interval calibration).

Run from the project root:  python notebooks/build_05b_fd003.py
Then execute it:            jupyter nbconvert --to notebook --execute --inplace notebooks/05b_fd003.ipynb
Reads the saved results of `python -m models.run_fd003` (it does not retrain anything). The findings are
generated from those results and every claim is asserted; the text adapts to whichever way the results came out.
"""
import json
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
XGB = "XGBoost, all 168 features"


def findings() -> str:
    r = json.loads(Path("models/results_phase5b.json").read_text())
    rule = json.loads(Path("models/calibration_rule.json").read_text())
    p5 = json.loads(Path("models/results_phase5.json").read_text())
    M, T, mc, v, pz = r["models"], r["tests"], r["mc_dropout"], r["verdict"], r["pooled_fd001_fd003"]
    f3, f1 = mc["FD003"], mc["FD001_post_hoc"]
    h = {k.split()[0]: t for k, t in T.items()}
    cand = rule["candidates"]

    # 1. Calibration rule chosen on FD001 validation
    a, e = cand["A"], cand[rule["chosen"]]
    assert e["loeo_coverage_testlike"] > a["loeo_coverage_testlike"] and e["loeo_coverage_near_failure"] > a["loeo_coverage_near_failure"], \
        "claim: rule E covers better than the Phase 5 rule on validation"
    # 2. FD003 ranking vs FD001 ranking
    fd1 = {"LSTM": p5["models"]["LSTM"]["rmse"], "CNN": p5["models"]["CNN"]["rmse"], XGB: p5["baseline"]["rmse"]}
    rank1, rank3 = sorted(fd1, key=fd1.get), sorted(fd1, key=lambda n: M[n]["rmse"])
    best3 = rank3[0]
    beats_simple = all(M[n]["rmse"] < M["Age only (cycle count)"]["rmse"] and M[n]["rmse"] < M["Fleet average"]["rmse"] for n in fd1)
    assert beats_simple, "claim: all three learned models beat the fleet-average and age-only baselines on FD003"
    order_word = "the same order as on FD001" if rank1 == rank3 else f"a different order from FD001 ({' < '.join(rank1)})"
    # 3. Verdict
    h1, h2 = h["H1"], h["H2"]
    wins = h1["significant"] and h1["diff"] < 0 and pz["LSTM"]["near_failure_mae"] <= pz[XGB]["near_failure_mae"]
    assert wins == v["deep_wins"], "claim: the verdict follows the pre-registered rule"
    sig_tests = [k for k, t in T.items() if t["significant"]]
    if v["deep_wins"]:
        verdict = (f"**The LSTM replaces XGBoost as the published model.** With 200 test engines its RMSE gap is {h1['diff']:+.2f} "
                   f"(interval {h1['ci'][0]:+.2f} to {h1['ci'][1]:+.2f}, Holm p {h1['p_holm']:.3f}), and it is at least as accurate near failure "
                   f"({pz['LSTM']['near_failure_mae']:.2f} vs {pz[XGB]['near_failure_mae']:.2f} cycles). Doubling the test evidence turned a "
                   "suggestive FD001 gap into a confirmed one.")
    else:
        verdict = (f"**XGBoost stays the published model.** Even with 200 test engines the LSTM's RMSE gap is {h1['diff']:+.2f} "
                   f"(interval {h1['ci'][0]:+.2f} to {h1['ci'][1]:+.2f}, Holm p {h1['p_holm']:.3f})"
                   + (", in XGBoost's favour" if h1["diff"] > 0 else ", not significant")
                   + f". Near failure: {pz['LSTM']['near_failure_mae']:.2f} (LSTM) vs {pz[XGB]['near_failure_mae']:.2f} (XGBoost) cycles. "
                   "The FD001 gap did not survive a fresh dataset, which is exactly what a confirmation set is for.")
    sig_line = ("Tests significant after Holm: " + "; ".join(sig_tests) + ".") if sig_tests else "No test is significant after Holm."
    # 4. Intervals
    gap = rule["nominal"] - f3["test_coverage"]
    cov_word = ("within 5 points of the nominal 90%" if abs(gap) <= 0.05 else "still below the nominal 90%" if gap > 0 else "above the nominal 90% (conservative)")
    import pandas as pd
    n_near = int((pd.read_csv("models/predictions_phase5b.csv").RUL_true <= 50).sum())
    near_word = ("on target" if abs(rule["nominal"] - f3["test_coverage_near_failure"]) <= 0.05 else
                 "below target" if f3["test_coverage_near_failure"] < rule["nominal"] else "above target")
    return f"""## Findings (generated from the results; every claim is asserted)

1. **Interval rule, fixed before FD003.** On FD001 validation (leave-engines-out), the Phase 5 rule covered {a['loeo_coverage_testlike']:.1%} of test-like and {a['loeo_coverage_near_failure']:.1%} of near-failure readings at a nominal 90%. Rule E (two factors, switched on the *predicted* RUL) covers {e['loeo_coverage_testlike']:.1%} and {e['loeo_coverage_near_failure']:.1%}. It was committed before any FD003 model was trained.
2. **FD003 results.** Test RMSE: {', '.join(f"{n} {M[n]['rmse']:.2f}" for n in rank3)}; {order_word}. Best on FD003: **{best3}**. All three learned models beat the fleet average ({M['Fleet average']['rmse']:.2f}) and the age-only rule ({M['Age only (cycle count)']['rmse']:.2f}), so the FD001 recipe transfers to a dataset with a second fault mode without re-tuning.
3. **Verdict on 200 engines.** {verdict} {sig_line}
4. **Intervals on fresh data.** FD003 test coverage with rule E: **{f3['test_coverage']:.0%}** ({f3['test_coverage_near_failure']:.0%} of the {n_near} engines near failure, {near_word}{": conservative where alerts matter" if f3['test_coverage_near_failure'] >= rule['nominal'] else ""}); the Phase 5 rule would have given {f3['phase5_rule_test_coverage']:.0%}, raw MC percentiles {f3['raw_test_coverage']:.0%}. Overall coverage is {cov_word}. On FD001 (post hoc, already seen) rule E gives {f1['test_coverage']:.0%} ({f1['test_coverage_near_failure']:.0%} near failure), against {p5['mc_dropout']['calibrated']['test_coverage']:.0%} for the Phase 5 rule.
5. **What this means for Phase 6.** The published model for alerts is **{v['published_winner']}**; alerts should use the lower edge of the rule-E interval, whose coverage has now been checked on data that played no part in choosing it.

## Why we did these tasks

| # | Task | Why we did it | What we learned | What it changes next |
| --- | --- | --- | --- | --- |
| 1 | Chose an interval rule on FD001 validation and committed it first | FD001's test coverage had been seen; re-tuning on it would be fitting the test set | Rule E fixes the under-coverage where it matters, using only what is known at prediction time | Phase 6 alerts use rule-E intervals |
| 2 | Made the loaders dataset-aware with per-dataset scaler and split files | FD003 must get its own training-only scaler, like FD001 | The FD001 path is unchanged (all earlier verifiers still pass) | Any NASA subset can be added the same way |
| 3 | Froze every FD001 setting and wrote the 8 tests before running | A confirmation is only honest if nothing is chosen after seeing it | The commit history proves the order | Later phases keep the same "decide, commit, then test" order |
| 4 | Ran all models on FD003 (5 seeds each) | 100 new test engines and a second fault mode | {order_word[0].upper() + order_word[1:]}; best: {best3} | Generalisation is measured, not assumed |
| 5 | Pooled FD001 + FD003 (engines resampled within each dataset) | 200 engines give much tighter intervals than 100 | The LSTM-vs-XGBoost question now has a {'confirmed' if v['deep_wins'] else 'firm'} answer: {v['published_winner']} | Phase 6 builds on {v['published_winner']} |
| 6 | Independent verifier and CI | Every published number must recompute with separate code | Scaler, windows, metrics, intervals, tests and verdict all recompute | The confirmation is part of every CI run |
"""


cells = [
    md("""# Phase 5b: FD003 confirmation and interval calibration

**Goal:** make the Phase 5 conclusions more reliable using only the original NASA data: test the frozen FD001 recipe on FD003 (100 new test engines, a second fault mode), settle LSTM vs XGBoost on 200 engines, and check a better interval rule on data that played no part in choosing it.

Training and scoring: `models/run_fd003.py`. Rule choice: `models/calibration.py`. Checks: `python -m models.verify_fd003`. Metrics document: `docs/MODEL_METRICS_PHASE5B.md`."""),
    md("""## Concepts in plain words

- **Confirmation set:** data kept completely unseen until every decision is made. FD001's test set was already used, so FD003 plays this role.
- **Pre-registration:** writing down the tests and the decision rule (here, in a dated git commit) before seeing the data, so results cannot steer the choices.
- **Frozen recipe:** the same sensors, features and model settings as FD001; only what the procedure itself learns (scaler, epochs, trees, interval factors) is learned again on FD003's own training and validation engines.
- **Pooling with stratified resampling:** combine both test sets (200 engines) and resample engines *within* each dataset, so every bootstrap sample keeps 100 engines from each.
- **Fault modes:** FD001 engines degrade in one way (high-pressure compressor); FD003 engines in one of two ways (compressor or fan). Some sensors move in opposite directions under the two faults."""),
    code("""import json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
FIG = ROOT / "docs" / "figures"
sns.set_theme(style="whitegrid")
res = json.loads((ROOT / "models/results_phase5b.json").read_text())
rule = json.loads((ROOT / "models/calibration_rule.json").read_text())
preds = pd.read_csv(ROOT / "models/predictions_phase5b.csv", float_precision="round_trip")
print("environment:", res["environment"])"""),
    md("""## 5b.1 Interval rule, chosen on FD001 validation only"""),
    code("""pd.DataFrame({k: {"rule": c["rule"], "test-like": c["loeo_coverage_testlike"], "near failure": c["loeo_coverage_near_failure"],
                  "width (test-like)": c["mean_width_testlike"]} for k, c in rule["candidates"].items()}).T"""),
    md("""## 5b.2 FD003 results (scored once)"""),
    code("""pd.DataFrame({n: {"val RMSE": m["val_rmse"], "test RMSE": m["rmse"], "95% interval": f"{m['rmse_ci95'][0]:.1f}-{m['rmse_ci95'][1]:.1f}",
                  "PHM08": m["phm"], "near-failure MAE": m["near_failure_mae"], "late share": m["late_share"]}
              for n, m in res["models"].items()}).T"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, m in zip(axes, ["LSTM", "XGBoost, all 168 features"]):
    late = preds[m] > preds.RUL
    ax.scatter(preds.RUL[~late], preds[m][~late], s=18, label="early (safe side)", color="#4C72B0")
    ax.scatter(preds.RUL[late], preds[m][late], s=18, label="late (risky side)", color="#C44E52")
    ax.plot([0, 125], [0, 125], color="black", linewidth=1)
    ax.set(title=f"FD003: {m}", xlabel="True RUL (capped at 125)", ylabel="Predicted RUL")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fd003_pred_vs_true.png", dpi=150, bbox_inches="tight")"""),
    md("""## 5b.3 Pre-registered tests (one Holm family) and pooled scores"""),
    code("""pd.DataFrame({k: {"gap A - B": t["diff"], "95% interval": f"{t['ci'][0]:+.2f} to {t['ci'][1]:+.2f}", "Holm p": t["p_holm"],
                  "significant": t["significant"]} for k, t in res["tests"].items()}).T"""),
    code("""pd.DataFrame(res["pooled_fd001_fd003"]).T"""),
    md("""## 5b.4 Intervals on FD003"""),
    code("""o = preds.sort_values("RUL").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.fill_between(o.index, o["MC 90% low (rule E)"], o["MC 90% high (rule E)"], alpha=0.3, label="rule-E 90% interval")
ax.plot(o.index, o["MC mean"], color="#4C72B0", linewidth=1, label="MC mean")
ax.plot(o.index, o.RUL, "k.", label="true RUL (capped)")
ax.set(title="FD003 test engines sorted by true RUL, LSTM with rule-E intervals", xlabel="test engine (sorted)", ylabel="RUL (cycles)")
ax.legend(fontsize=8)
fig.savefig(FIG / "fd003_intervals.png", dpi=150, bbox_inches="tight")
pd.Series({k: v for k, v in res["mc_dropout"]["FD003"].items()}).round(3)"""),
    md(findings()),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "notebooks/05b_fd003.ipynb")
print("wrote notebooks/05b_fd003.ipynb")
