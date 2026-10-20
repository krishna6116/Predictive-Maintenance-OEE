"""Builds notebooks/06_maintenance.ipynb (Phase 6: maintenance decisions and simulated OEE).

Run from the project root:  python notebooks/build_06_maintenance.py
Then execute it:            jupyter nbconvert --to notebook --execute --inplace notebooks/06_maintenance.ipynb
Reads the saved results of `python -m maintenance.run_phase6` (it re-runs nothing). The findings are generated
from those results and every claim is asserted, so a false claim stops the build.
"""
import json
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
P = ("Run to failure", "Fixed interval", "Predictive")


def findings() -> str:
    r = json.loads(Path("maintenance/results_phase6.json").read_text())
    d1, d3, pool, gaps = r["datasets"]["FD001"], r["datasets"]["FD003"], r["pooled"], r["pooled_gaps"]
    s1, s3 = d1["snapshot"], d3["snapshot"]
    cap = r["settings"]["horizon_days"] * r["settings"]["crews_per_day"]
    assert s1["schedule_status"] == s3["schedule_status"] == "OPTIMAL", "claim: both snapshot schedules are optimal"
    assert max(s1["per_day"].values()) <= r["settings"]["crews_per_day"], "claim: crews respected"
    late1 = sum(v for k, v in s1["outcomes"].items() if "too late" in k or k.startswith("missed"))
    late3 = sum(v for k, v in s3["outcomes"].items() if "too late" in k or k.startswith("missed"))
    assert late1 == late3 == 0, "claim: no snapshot engine is serviced too late or missed"
    o = {p: pool[p]["oee"] for p in P}
    assert o["Predictive"] > o["Fixed interval"] > o["Run to failure"], "claim: predictive > fixed > run to failure"
    assert gaps["Predictive - Fixed interval"]["excludes_zero"] and gaps["Predictive - Run to failure"]["excludes_zero"]
    f = {p: pool[p]["unplanned_failures"] for p in P}
    w = {p: pool[p]["wasted_cycles"] for p in P}
    assert f["Predictive"] < f["Fixed interval"] and w["Predictive"] < w["Fixed interval"], \
        "claim: predictive has fewer failures AND wastes less life than fixed interval"
    sens_lower = all(v["Predictive"]["oee"] > v["Fixed interval"]["oee"] for d in (d1, d3) for v in d["sensitivity"].values())
    assert sens_lower, "claim: predictive beats fixed interval in every sensitivity run"
    import pandas as pd
    snap = pd.read_csv("maintenance/snapshot_decisions_phase6.csv", float_precision="round_trip")
    mach = pd.read_csv("maintenance/machines_phase6.csv", float_precision="round_trip")
    unbooked = snap[snap.flagged & snap.service_day.isna()]
    assert (unbooked.true_rul > r["settings"]["horizon_days"] * r["settings"]["cycles_per_day"]).all(),         "claim: engines left unbooked at the snapshot have more than a week of true life left"
    pf = mach[(mach.policy == "Predictive") & (mach.event == "failure")].merge(
        snap, left_on=["dataset", "machine"], right_on=["dataset", "engine_id"])
    assert len(pf) == f["Predictive"] and (pf.life > pf.cycle).all() and (pf.rul_low > pf.true_rul).all(),         "claim: every predictive failure is after the data ends and follows a lower-bound miss at the last reading"
    g = gaps["Predictive - Fixed interval"]
    rows = [
        f"1. **Decision snapshot.** At each engine's last observed cycle, {s1['flagged']} FD001 and {s3['flagged']} FD003 engines are "
        f"flagged on the lower bound. FD001 needs more services than the {cap} crew slots of the week, so CP-SAT books "
        f"{s1['scheduled']} and leaves {s1['flagged'] - s1['scheduled']} (with more than a week of true life left) for the next plan. "
        "Both plans are optimal (proved again with the Hungarian algorithm) and no booked engine fails first.",
        f"2. **Policies (simulated, 200 machines).** OEE: run to failure {o['Run to failure']:.1%}, fixed interval "
        f"{o['Fixed interval']:.1%}, predictive {o['Predictive']:.1%}. Unplanned failures {f['Run to failure']} / {f['Fixed interval']} / "
        f"{f['Predictive']}; useful life thrown away 0 / {w['Fixed interval']:,} / {w['Predictive']:,} cycles.",
        f"3. **The gain over a fixed interval is small but consistent:** {g['diff'] * 100:+.2f} OEE points (95% interval "
        f"{g['ci'][0] * 100:+.2f} to {g['ci'][1] * 100:+.2f}), and predictive stays ahead in every sensitivity run. Most of its value "
        f"is in failures avoided and life not wasted, which a single OEE figure understates.",
        f"4. **Remaining failures are explained, not hidden:** all {f['Predictive']} predictive-policy failures happen after the "
        "engine's NASA data ends, after the last forecast's lower bound had already missed the truth.",
        "5. **Honest limits:** the plant is a simulation with the guide's assumptions; a higher alert threshold looks better here "
        "but is not adopted, because picking it from these results would tune on the test engines.",
    ]
    return "## What the results say\n\n" + "\n".join(rows)


WHY = """## Why we did these tasks

| # | Task | Why we did it | What we learned | What it changes next |
| --- | --- | --- | --- | --- |
| 1 | Re-checked the Phase 5 forecasts before using them | Decisions are only as good as their inputs | All 200 forecasts are valid; the published model was rebuilt and matched exactly | Phase 7 can score new shifts with the same model |
| 2 | Alert on the interval's lower bound | Acting on the pessimistic edge protects against late predictions | The bound flags engines early enough that no booked engine fails first | Alerts in the Prefect flow use the same rule |
| 3 | CP-SAT schedule with an independent optimality proof | Crews are limited; a solver's word is not evidence | Both snapshot plans are optimal; the guide's cost breaks if the late weight exceeds the unserved penalty | Weights stay as documented assumptions with that caveat |
| 4 | Simulated plant with written assumptions | NASA data has no production records | OEE = availability x performance x quality, checked against good / (rate x planned) | Phase 8 dashboards read these fact_oee rows |
| 5 | Three policies on the same 200 machines | The business case needs a fair comparison | Predictive wins on failures, wasted life and OEE, but its OEE edge over a fixed interval is small | The resume bullet quotes failures and wasted life, not only OEE |
| 6 | Traced engines from raw sensor to OEE | Every displayed action must lead back to its forecast | Values stay consistent at every stage; the database enforces the alert-to-forecast link | Dashboards can drill from an alert to its forecast |
| 7 | Sensitivity runs, not tuning | Assumptions are guesses; the test set must not choose them | Crew count barely matters in the replay; the alert threshold matters most | A threshold choice belongs on validation engines in a later phase |
"""


def build():
    nb = nbf.v4.new_notebook()
    cells = [
        md("# Phase 6: Maintenance decisions and simulated OEE\n\n**Goal:** turn each engine's forecast into a maintenance action "
           "under crew limits, and compare three maintenance policies on a simulated plant.\n\n> **Simulation, not measured results.** "
           "The forecasts are real model outputs; the plant, production and OEE are simulated with the study guide's assumptions.\n\n"
           "Everything is computed by `python -m maintenance.run_phase6`; this notebook reads the saved results."),
        md("## Concepts in plain words\n\n"
           "- **Alert threshold:** the point at which an engine is flagged for service. We use the *lower* edge of the 90% interval, "
           "so we act on the pessimistic estimate.\n"
           "- **CP-SAT:** a solver for yes/no decisions under rules. Here: is engine e serviced on day d? Rules: each engine at most "
           "once, at most 3 services a day. It minimises a cost: late service is expensive, early service wastes some life, "
           "leaving an engine unserved is very expensive.\n"
           "- **OEE:** availability (share of planned time running) x performance (output vs ideal speed) x quality (good units "
           "vs all units).\n"
           "- **Policies:** run to failure (repair after breakdown), fixed interval (service every N cycles), predictive (service "
           "when the forecast says so)."),
        code("import json, sys\nfrom pathlib import Path\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\n"
             "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nsys.path.insert(0, str(ROOT))\n"
             "FIG = ROOT / 'docs' / 'figures'\nCSV = {'float_precision': 'round_trip'}\n"
             "res = json.loads((ROOT / 'maintenance/results_phase6.json').read_text())\n"
             "snap = pd.read_csv(ROOT / 'maintenance/snapshot_decisions_phase6.csv', **CSV)\n"
             "wo = pd.read_csv(ROOT / 'maintenance/work_orders_phase6.csv', **CSV)\n"
             "mach = pd.read_csv(ROOT / 'maintenance/machines_phase6.csv', **CSV)\nprint('run', res['run_id'])\n"
             "res['settings']"),
        md("## 6.1 Decision snapshot\n\nEach test engine at its last observed cycle: forecast, alert, schedule, and what really "
           "happened (NASA's true remaining life)."),
        code("cols = ['dataset', 'engine_id', 'cycle', 'rul_point', 'rul_low', 'rul_high', 'risk', 'service_day', 'true_rul', 'outcome']\n"
             "snap[snap.flagged][cols].sort_values(['dataset', 'rul_low']).round(1)"),
        code("pd.crosstab(snap.dataset, snap.outcome)"),
        md("## 6.2 Three policies on the simulated plant\n\n*Simulation, not measured results.*"),
        code("rows = []\nfor ds, d in list(res['datasets'].items()) + [('Pooled', {'policies': res['pooled']})]:\n"
             "    for p, s in d['policies'].items():\n"
             "        rows.append({'dataset': ds, 'policy': p, **{k: s[k] for k in ('unplanned_failures', 'planned_services', "
             "'downtime_hours', 'wasted_cycles', 'availability', 'performance', 'quality', 'oee')}})\n"
             "tab = pd.DataFrame(rows)\ntab.style.format({k: '{:.1%}' for k in ('availability', 'performance', 'quality', 'oee')})"),
        code("pool = tab[tab.dataset == 'Pooled'].set_index('policy')\nfig, axes = plt.subplots(1, 3, figsize=(14, 4))\n"
             "for ax, col, title in zip(axes, ['oee', 'unplanned_failures', 'wasted_cycles'],\n"
             "                          ['Simulated OEE', 'Unplanned failures', 'Useful life thrown away (cycles)']):\n"
             "    ax.bar(pool.index, pool[col], color=['#C44E52', '#8C8C8C', '#4C72B0'])\n    ax.set_title(title)\n"
             "    ax.tick_params(axis='x', rotation=15)\naxes[0].set_ylim(0.9, 1.0)\n"
             "fig.suptitle('200 NASA test engines as simulated machines (simulation, not measured results)')\n"
             "fig.tight_layout()\nfig.savefig(FIG / 'phase6_policies.png', dpi=150, bbox_inches='tight')"),
        code("pd.DataFrame({k: {'gap (OEE points)': round(v['diff'] * 100, 2), '95% interval': "
             "f\"{v['ci'][0] * 100:+.2f} to {v['ci'][1] * 100:+.2f}\"} for k, v in res['pooled_gaps'].items()}).T"),
        md("## 6.3 Sensitivity (nothing is chosen from this)"),
        code("pd.DataFrame([{'dataset': ds, 'change': k, **{p: round(v[p]['oee'] * 100, 2) for p in v}, "
             "'predictive failures': v['Predictive']['unplanned_failures']}\n"
             "              for ds, d in res['datasets'].items() for k, v in d['sensitivity'].items()])"),
        md("## 6.4 One machine, end to end\n\nThe full traces are in `docs/MAINTENANCE_OEE.md`; here is one work order and its outcome."),
        code("e = int(snap.loc[snap.rul_low.idxmin(), 'engine_id']); ds = snap.loc[snap.rul_low.idxmin(), 'dataset']\n"
             "display(wo[(wo.dataset == ds) & (wo.engine_id == e)].T)\n"
             "mach[(mach.dataset == ds) & (mach.machine == e)].set_index('policy')[['life', 'run_cycles', 'event', 'wasted_cycles', "
             "'downtime_hours', 'good_units', 'planned_hours']]"),
        md(findings()),
        md(WHY),
    ]
    nb["cells"] = cells
    nbf.write(nb, "notebooks/06_maintenance.ipynb")
    print("wrote notebooks/06_maintenance.ipynb")


if __name__ == "__main__":
    build()
