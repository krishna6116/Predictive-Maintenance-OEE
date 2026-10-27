"""Builds notebooks/07_monitoring.ipynb (Phase 7: orchestration and monitoring).

Run from the project root:  python notebooks/build_07_monitoring.py
Then execute it:            jupyter nbconvert --to notebook --execute --inplace notebooks/07_monitoring.ipynb
Reads the saved replay (results/phase7/phase7_*.csv, results_phase7.json, phase7_checks.json, monitoring_rule.json); it
re-runs nothing. Findings are generated from those files and every claim is asserted.
"""
import json
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def findings() -> str:
    import pandas as pd
    rule = json.loads(Path("pipeline/monitoring_rule.json").read_text())
    chk = json.loads(Path("results/phase7/phase7_checks.json").read_text())
    res = json.loads(Path("results/phase7/results_phase7.json").read_text())
    runs = pd.read_csv("results/phase7/phase7_runs.csv")
    fc, rp = chk["flow_check"], chk["replay"]
    assert (runs.flow_status == "complete").all() and len(runs) == res["runs"], "claim: every run completed"
    pub = {"FD001": json.loads(Path("models/results_phase5.json").read_text())["models"]["LSTM"]["rmse"],
           "FD003": json.loads(Path("models/results_phase5b.json").read_text())["models"]["LSTM"]["rmse"]}
    for ds, r in res["datasets"].items():
        assert abs(r["final_cum"]["rmse"] - pub[ds]) < 1e-9, "claim: final RMSE equals the published RMSE"
    assert fc["plus5pct"]["drifted"] == [fc["sensor"]] and fc["plus5pct"]["needs_retraining"] and fc["clean_runs_flagged"] == 0, \
        "claim: the guide's +5% check passes and the clean runs before it set nothing"
    val_fa = rule["drift"]["false_alarm_chosen"]
    test_fa = rp["drift_false_alarm_rate"]
    flags = {ds: r["flagged_runs"] for ds, r in res["datasets"].items()}
    reasons = {ds: r["flag_reasons"] for ds, r in res["datasets"].items()}
    drift_flags = sum(v for r in reasons.values() for k, v in r.items() if k.startswith("drift"))
    health_flags = sum(v for r in reasons.values() for k, v in r.items() if not k.startswith("drift"))
    assert drift_flags == rp["runs_with_a_drifted_sensor"], "claim: drift-triggered flags = runs with a drifted sensor"
    rows = [
        f"1. **The flow ran every shift.** {res['runs']} runs ({res['shifts']} shifts x 2 datasets), all complete, "
        f"{res['forecasts']:,} forecasts ({res['invalid_forecasts']} invalid). Re-running, retrying and resuming are proven in "
        "`tests/test_flow.py`.",
        "2. **Monitoring is tied to the published results.** Once all 100 engines of a dataset are labelled, the flow's RMSE "
        f"equals the published test RMSE ({', '.join(f'{ds} {pub[ds]:.2f}' for ds in pub)}).",
        f"3. **The guide's drift check passes.** +5% on {fc['sensor']}: the validator quarantined all {fc['plus5pct']['readings']} "
        f"readings, drift still flagged {fc['sensor']}, and the retraining flag was set; the {fc['clean_runs']} clean runs before it set nothing.",
        f"4. **False alarms on clean test data:** {test_fa:.2%} of runs had a drifted sensor ({rp['runs_with_a_drifted_sensor']} of "
        f"{rp['runs_with_drift_test']}), against {val_fa:.2%} on clean validation streams when the rule was chosen."
        + (" That is within the 5% target." if test_fa <= 0.05 else " That is ABOVE the 5% target: reported, not re-tuned."),
        f"5. **Retraining flags on the clean test replay:** FD001 {flags['FD001']} runs, FD003 {flags['FD003']} runs; "
        f"{health_flags} come from the health rules (labelled engines) and {drift_flags} from drift."
        + (" None of these is a deliberate fault: they show what the pre-registered rule does on real test data, and the "
           "limits were not moved after seeing them." if health_flags + drift_flags else ""),
    ]
    dd, hd = chk["drift_diagnosis"], chk["health_diagnosis"]
    assert dd["all_drifted_sensors_in_top_wear"], "claim: every drifted sensor is a wear sensor"
    assert all(v["drift_runs"]["median_rul_test"] > v["drift_runs"]["median_rul_reference"] for v in dd["by_dataset"].values()
               if "drift_runs" in v), "claim: in flagged windows the test engines are healthier than the reference"
    assert all(v["first_window"]["rmse"] > v["other_engines"]["rmse"] and v["first_window"]["median_life"] < v["other_engines"]["median_life"]
               for v in hd.values()), "claim: the first labelled engines are shorter-lived and harder to predict"
    f3 = dd["by_dataset"]["FD003"]["drift_runs"]
    rows += [
        f"6. **Why the clean runs were flagged (traced, not tuned):** the drift reference matches age but not wear. Test engines "
        f"still reporting late are healthier than training engines of the same age (FD003 flagged windows: median remaining life "
        f"{f3['median_rul_test']:.0f} vs {f3['median_rul_reference']:.0f}), and every sensor that 'drifted' is one of the "
        f"{dd['top_wear_n']} most wear-related sensors. The first labels come from the shortest-lived engines (FD001 first 20: "
        f"RMSE {hd['FD001']['first_window']['rmse']:.1f} vs {hd['FD001']['other_engines']['rmse']:.1f} for the rest), which is "
        "what tripped the RMSE rule. A fix must be chosen on validation and confirmed on fresh data (FD002/FD004)."]
    return "## What the results say\n\n" + "\n".join(rows)


WHY = """## Why we did these tasks

| # | Task | Why we did it | What we learned | What it changes next |
| --- | --- | --- | --- | --- |
| 1 | Prefect flow, one run per shift, retries and fixed run IDs | A pipeline someone runs by hand is not a pipeline; re-runs must not duplicate | A forced database failure is retried, a failed shift resumes, a re-run changes nothing | Phase 8 dashboards can trust one row per run |
| 2 | Published model only, proven identical before use | Decisions must come from the model whose numbers were published | Batch size changed predictions in the 5th decimal; fixed batches of 100 make them exact | Any new model must pass the same identity check |
| 3 | Rules fixed on validation and committed first | A threshold chosen on test results is tuned to them | Reading-level KS raised false alarms on clean data; per-engine tests and an RMSE rule of 1.45x (not 1.3x) were needed | The rule file is the contract for monitoring |
| 4 | Labels only after failure | In a plant, the true remaining life is known only when the engine fails | Health is blind until the first failures; the earliest labels come from the shortest-lived engines | Censoring must be planned for if services are executed |
| 5 | Drift on everything received | The validator removes big faults before the model sees them | The +5% check works only because drift looks at quarantined rows too | Dashboards should show quarantine and drift together |
| 6 | Independent verifier and a fresh replay in CI | Numbers must be reproducible on another machine | KS, health, flags and rows recompute exactly | Phase 8 reads the verified tables |
| 7 | Fixed a pinned dependency (SQLAlchemy 2.0.55) | .serve() created no runs with 2.1.5 | A library version can silently break scheduling | Keep the server check in the run instructions |
"""


def build():
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md("# Phase 7: Orchestration and monitoring\n\n**Goal:** run the whole pipeline by itself every shift, record its "
           "health, and raise a retraining flag when the data or the model's errors say so.\n\n> The forecasts are real model "
           "outputs; the plant clock and the failure log are simulated from NASA's data (see `docs/MONITORING.md`)."),
        md("## Concepts in plain words\n\n"
           "- **Orchestration:** running steps in order, on a schedule, with retries and logs. In Prefect a *task* is one step, a "
           "*flow* is the sequence, a *deployment* runs it on a schedule.\n"
           "- **Idempotent:** running the same shift twice leaves the same rows.\n"
           "- **Label timing:** a prediction can only be scored once the engine has actually failed.\n"
           "- **Data drift:** incoming readings stop looking like the training data. The **KS test** measures the biggest gap "
           "between two distributions; **Holm** corrects for testing 14 sensors at once.\n"
           "- **Retraining flag:** a pre-set rule that says the model should be retrained."),
        code("import json, sys\nfrom pathlib import Path\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\n"
             "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nsys.path.insert(0, str(ROOT))\n"
             "FIG = ROOT / 'docs' / 'figures'\nrule = json.loads((ROOT / 'pipeline/monitoring_rule.json').read_text())\n"
             "chk = json.loads((ROOT / 'results/phase7/phase7_checks.json').read_text())\n"
             "res = json.loads((ROOT / 'results/phase7/results_phase7.json').read_text())\n"
             "health = pd.read_csv(ROOT / 'results/phase7/phase7_health.csv')\ndrift = pd.read_csv(ROOT / 'results/phase7/phase7_drift.csv')\n"
             "print({k: res[k] for k in ('shifts', 'runs', 'forecasts', 'invalid_forecasts', 'work_orders')})"),
        md("## 7.1 The rule, fixed before the test replay"),
        code("pd.DataFrame({'alpha': [rule['drift']['alpha']], 'drift false alarms (validation)': [rule['drift']['false_alarm_chosen']],\n"
             "              'RMSE multiplier': [rule['health']['rmse_multiplier']], 'miss-rate limit': [rule['health']['miss_rate_limit']],\n"
             "              'combined false-alarm estimate': [rule['combined_false_alarm_upper_estimate']]})"),
        md("## 7.2 Model health over the replay\n\nRolling metrics of the 20 most recently labelled engines, with the pre-set limits."),
        code("fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex='col')\n"
             "for j, ds in enumerate(['FD001', 'FD003']):\n"
             "    h = health[health.dataset == ds]\n"
             "    ref = rule['health']['reference'][ds]['reference_rmse']\n"
             "    axes[0, j].plot(h['shift'], h.rmse, label='rolling RMSE (20 engines)')\n"
             "    axes[0, j].plot(h['shift'], h.cum_rmse, label='all labelled engines', alpha=0.7)\n"
             "    axes[0, j].axhline(rule['health']['rmse_multiplier'] * ref, color='red', ls='--', label='flag limit')\n"
             "    axes[0, j].set(title=f'{ds}: RMSE', ylabel='cycles'); axes[0, j].legend()\n"
             "    axes[1, j].plot(h['shift'], h.lower_bound_miss_rate, label='rolling miss rate')\n"
             "    axes[1, j].axhline(rule['health']['miss_rate_limit'], color='red', ls='--', label='flag limit')\n"
             "    fl = h[h.needs_retraining]\n"
             "    axes[1, j].scatter(fl['shift'], fl.lower_bound_miss_rate, color='red', s=10, label='flagged run')\n"
             "    axes[1, j].set(title=f'{ds}: lower-bound miss rate', xlabel='shift'); axes[1, j].legend()\n"
             "fig.tight_layout(); fig.savefig(FIG / 'phase7_health.png', dpi=150, bbox_inches='tight')"),
        md("## 7.3 Drift on the clean test replay"),
        code("t = drift[drift.tested]\nruns = t.groupby(['dataset', 'shift']).drifted.any().reset_index()\n"
             "print('runs with a drifted sensor:', int(runs.drifted.sum()), 'of', len(runs))\n"
             "t[t.drifted][['dataset', 'shift', 'sensor', 'n_recent', 'ks_stat', 'p_holm']]"),
        md("## 7.4 The guide's checks"),
        code("chk['flow_check']"),
        code("pd.DataFrame({k: {'mean': v['mean'], 'weakest': f\"{v['weakest_sensor']} {v['min']:.0%}\"} "
             "for k, v in chk['test_window_power'].items()}).T"),
        md(findings()),
        md(WHY),
    ]
    nbf.write(nb, "notebooks/07_monitoring.ipynb")
    print("wrote notebooks/07_monitoring.ipynb")


if __name__ == "__main__":
    build()
