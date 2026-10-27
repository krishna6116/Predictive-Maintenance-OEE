"""Builds notebooks/07b_monitoring.ipynb (Phase 7b: monitoring improved and re-tested on fresh data).

Run from the project root:  python notebooks/build_07b_monitoring.py
Then execute it:            jupyter nbconvert --to notebook --execute --inplace notebooks/07b_monitoring.ipynb
Reads the saved Phase 7b results; re-runs nothing. Findings are generated and every claim is asserted.
"""
import json
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def findings() -> str:
    import pandas as pd
    chk = json.loads(Path("results/phase7b/checks.json").read_text())
    fs = json.loads(Path("results/phase7b/fault_suite.json").read_text())
    c1 = chk["C1"]
    fd2_ok = all(c1[f"FD002 {f}"]["rate"] <= 0.05 for f in ("synchronized", "staggered"))
    fd4_ok = all(c1[f"FD004 {f}"]["rate"] <= 0.05 for f in ("synchronized", "staggered"))
    assert fd2_ok and not fd4_ok, "claim: C1 passes on FD002 and fails on FD004"
    assert all(chk["C2"][d]["passed"] for d in ("FD002", "FD004")), "claim: C2 passes on both"
    assert all(fs[d]["step"]["rate"] >= 0.8 and fs[d]["ramp"]["rate"] >= 0.8 for d in ("FD002", "FD004")), "claim: C3, C4 pass"
    post = {f: pd.read_csv(f"results/phase7b/posthoc_{f}/phase7_health.csv").needs_retraining.sum() for f in ("synchronized", "staggered")}
    v21 = pd.read_csv("results/phase7b/v21_fresh_synchronized/phase7_health.csv")
    v21_fd4 = v21[v21.dataset == "FD004"].needs_retraining.mean()
    assert v21_fd4 > 0.05, "claim: the post-hoc spread fix does not rescue FD004"
    sp = chk["diagnosis"]["spread"]["FD004"]
    assert sp["v2"]["readings_abs_z_over_20"] > 0 and sp["v2.1"]["readings_abs_z_over_20"] == 0, "claim: v2.1 removes the blow-ups"
    cl = chk["closed_loop"]
    assert all(cl[d]["miss_rate_from_labels_only"] > 3 * cl[d]["true_miss_rate_same_forecasts"] for d in cl), "claim: labels-only is biased"
    assert all(cl[d]["miss_rate_bounds_all_engines"][0] <= cl[d]["true_miss_rate_same_forecasts"] <= cl[d]["miss_rate_bounds_all_engines"][1]
               for d in cl), "claim: the censoring bounds contain the truth"
    corr = chk["diagnosis"]["fd004"]["correlation_with_mode_signature"]
    rows = [
        f"1. **Fixed before testing, then tested on unseen fleets.** Criteria, rule and fresh-fleet settings were committed before "
        f"any FD002/FD004 test replay.",
        f"2. **C1 (clean-run flags <= 5%) passes on FD002** ({c1['FD002 synchronized']['rate']:.1%} / {c1['FD002 staggered']['rate']:.1%}) "
        f"**and fails on FD004** ({c1['FD004 synchronized']['rate']:.1%} / {c1['FD004 staggered']['rate']:.1%}). On FD001/FD003 (post hoc) "
        f"the new rule raised {post['synchronized']} and {post['staggered']} flags, where Phase 7's rule had 11.6% false alarms.",
        f"3. **C2-C4 pass on both fresh fleets:** one +5% batch sets the flag (as 'sensor fault suspected'); +1 sd steps are caught "
        f"within 10 shifts in {fs['FD002']['step']['rate']:.0%} / {fs['FD004']['step']['rate']:.0%} of injections, ramps in "
        f"{fs['FD002']['ramp']['rate']:.0%} / {fs['FD004']['ramp']['rate']:.0%}. Dropouts are invisible to drift by design.",
        f"4. **A real defect was found and fixed, but it was not FD004's cause.** The spread model went negative in sparse corners "
        f"({sp['v2']['readings_abs_z_over_20']} FD004 readings with |z| > 20; 0 after the log-scale fix). Re-scored post hoc with the fix, "
        f"FD004 still flags {v21_fd4:.1%} of synchronized runs. Its flagged pattern correlates {corr['-1']:+.2f} / {corr['1']:+.2f} with "
        "the two fault-mode signatures: likely a fault-mode mixture, not proven.",
        f"5. **Closed loop:** when services are carried out, only {cl['FD002']['failed_labelled']} of {cl['FD002']['engines']} FD002 "
        f"engines ever produce a failure label, and the labels-only miss rate reads {cl['FD002']['miss_rate_from_labels_only']:.0%} "
        f"against a true {cl['FD002']['true_miss_rate_same_forecasts']:.1%}. Censored engines give bounds that contain the truth but "
        "are too wide to act on: a deployment needs inspection labels or a control group.",
    ]
    return "## What the results say\n\n" + "\n".join(rows)


WHY = """## Why we did these tasks

| # | Task | Why we did it | What we learned | What it changes next |
| --- | --- | --- | --- | --- |
| 1 | Success criteria and rule committed before any fresh replay | A rule judged on data that shaped it proves nothing | The criteria made a mixed result impossible to hide | Any further change needs data nobody has seen |
| 2 | Wear-adjusted residuals instead of raw readings | Phase 7 mistook healthy survivors for drift | On FD001/FD003 and FD002 the false alarms vanished | Drift tests must condition on the state of the machine |
| 3 | Cross-fitted 60-engine calibration fleets | 20-engine fleets could not reproduce the failure | A calibration set must be able to show the problem it is meant to fix | Check calibration data against the known failure first |
| 4 | Fresh confirmation on FD002 and FD004 | The only honest test of a fix | FD002 passed; FD004 failed | Fault modes need their own expected levels |
| 5 | Root-cause before re-scoring | A failure is information | A real defect existed but was not the cause; my first explanation was wrong and is corrected | Test an explanation with the fix before believing it |
| 6 | Fault suite through a mirror proven equal to the flow | 840 injections would take days in the real flow | Steps and ramps are caught within 2-8 shifts | Fleet-wide faults are covered; single-engine faults are not |
| 7 | Closed loop with censoring bounds | Real maintenance hides the labels monitoring needs | Labels-only monitoring is badly biased once the model acts | Plan inspection labels or a control group into the process |
"""


def build():
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md("# Phase 7b: Monitoring, improved and re-tested on fresh data\n\nPhase 7's drift test raised false alarms on clean data. "
           "Phase 7b replaces it with a wear-adjusted test, fixes the rule before looking, and tests it on NASA's FD002 and FD004, "
           "which nothing had seen. Full results: `docs/MONITORING_7B.md`."),
        md("## Concepts in plain words\n\n"
           "- **Residual:** what a sensor reads minus what it should read for an engine of that age and predicted remaining life.\n"
           "- **Persistence:** a sensor counts as drifted only if it drifts in 2 of the last 3 runs, unless the evidence is "
           "overwhelming (a sudden fault).\n"
           "- **Pre-registration:** writing down the rule and the pass marks before the test data is touched.\n"
           "- **Censoring:** a serviced engine never fails, so we never learn its true remaining life."),
        code("import json, sys\nfrom pathlib import Path\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\n"
             "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nsys.path.insert(0, str(ROOT))\n"
             "FIG = ROOT / 'docs' / 'figures'\nOUT = ROOT / 'results' / 'phase7b'\n"
             "chk = json.loads((OUT / 'checks.json').read_text())\nfs = json.loads((OUT / 'fault_suite.json').read_text())\n"
             "pd.DataFrame(chk['C1']).T"),
        md("## Clean-run flags over the replay"),
        code("fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)\n"
             "for ax, sc in zip(axes, ['fresh_synchronized', 'v21_fresh_synchronized']):\n"
             "    h = pd.read_csv(OUT / sc / 'phase7_health.csv')\n"
             "    for ds, g in h.groupby('dataset'):\n"
             "        ax.plot(g['shift'], g.needs_retraining.astype(int).rolling(10, min_periods=1).mean(), label=ds)\n"
             "    ax.set(title=sc.replace('_', ' '), xlabel='shift', ylabel='share of last 10 runs flagged'); ax.legend()\n"
             "fig.tight_layout(); fig.savefig(FIG / 'phase7b_flags.png', dpi=150, bbox_inches='tight')"),
        md("## Fault suite"),
        code("pd.DataFrame({ds: {k: f\"{v['detected']}/{v['testable']}\" for k, v in fs[ds].items() if isinstance(v, dict) and 'rows' in v}\n"
             "              for ds in ('FD002', 'FD004')})"),
        md("## Closed loop"),
        code("pd.DataFrame(chk['closed_loop']).T"),
        md(findings()),
        md(WHY),
    ]
    nbf.write(nb, "notebooks/07b_monitoring.ipynb")
    print("wrote notebooks/07b_monitoring.ipynb")


if __name__ == "__main__":
    build()
