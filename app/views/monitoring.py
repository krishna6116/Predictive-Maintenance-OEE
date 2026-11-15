"""Monitoring: drift and retraining flags by shift, Phase 7 (rule v1) and Phase 7b (rule v2, pre-registered)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

import common as C

C.header("monitoring", "Each 8-hour shift the flow checks the incoming sensor data for drift and the model's errors once "
                       "failures are known, and raises a retraining flag. On clean data the flag should stay quiet.")

runs = C.load("monitoring_runs.csv")
target, limit = C.fact("p7_drift_target"), C.fact("c1_limit")

# plain-English summary first (review 2026-11-15: the page was hard to read without the project's vocabulary);
# filled once the results below are computed, so every claim in it comes from the data
summary = st.container()
st.caption(C.plain("drift"))
SCEN = {"Phase 7: FD001/FD003, rule v1": "phase7_v1",
        "Phase 7b: FD002/FD004 synchronized, rule v2": "phase7b_v2_synchronized",
        "Phase 7b: FD002/FD004 staggered, rule v2": "phase7b_v2_staggered"}

# The headline results stay visible under every filter: the misses are part of the result
p7 = runs[runs.scenario == "phase7_v1"]
tested = p7[p7.drift_tested.astype(bool)]
fa = tested.any_drifted.astype(bool).mean()
st.subheader("Results against the pre-registered criteria")
c1, c2 = st.columns(2)
c1.metric("Phase 7 drift false-alarm rate on clean test runs", C.pct(fa), help=C.define("drift_false_alarm_rate"))
c1.caption(f"{int(tested.any_drifted.astype(bool).sum())} of {len(tested)} tested runs; target {C.pct(target, 0)}: "
           f"{'met' if fa <= target else '**not met**'}. Cause traced to a reference that matched engine age but not wear.")
c2.metric("Phase 7 runs flagged for retraining", f"{int(p7.needs_retraining.sum())} of {len(p7)}",
          help="Runs with needs_retraining set (drift, error ratio or interval misses) in the Phase 7 clean replay.")
c2.caption("Counted over every run. The false-alarm rate on the left counts only the runs that had a drift test, "
           "so the two figures use different totals.")

rows = []
for fleet in ("synchronized", "staggered"):
    r = runs[runs.scenario == f"phase7b_v2_{fleet}"]
    for ds, g in r.groupby("dataset"):
        rate = g.needs_retraining.mean()
        rows.append({"Criterion": f"C1 clean-run flags <= {C.pct(limit, 0)}", "Dataset": ds, "Fleet": fleet,
                     "Flagged runs": f"{int(g.needs_retraining.sum())} of {len(g)}", "Rate": C.pct(rate),
                     "Result": "PASS" if rate <= limit else "FAIL"})
c1tab = pd.DataFrame(rows)
st.dataframe(c1tab, hide_index=True, width="stretch")
failed = sorted(set(c1tab[c1tab.Result == "FAIL"].Dataset))
passed = sorted(set(c1tab.Dataset) - set(failed))
summary.info(
    "**In short:** the model gets a health check every shift. Two alarm rules were tested on healthy engines, where a "
    f"good alarm should stay quiet. The first rule raised false alarms on {C.pct(fa)} of tested runs against a "
    f"{C.pct(target, 0)} target ({'met' if fa <= target else 'not met'}). The second rule, fixed in advance, stayed "
    f"within its {C.pct(limit, 0)} limit on {', '.join(passed) or 'no dataset'} and failed on {', '.join(failed) or 'no dataset'}. "
    "Every result is shown below as measured, not re-tuned.")
st.markdown(f"Phase 7b's rule was pre-registered (fixed on validation engines before any FD002/FD004 test replay). "
            f"C1 passes on {', '.join(sorted(set(c1tab.Dataset) - set(failed))) or 'no dataset'} and **FAILS on "
            f"{', '.join(failed) or 'no dataset'}**; the likely cause is a mix of two fault modes the expected sensor levels "
            "do not separate (not proven). "
            f"Phase 7's target {C.pct(target, 0)} was {'met' if fa <= target else 'likewise **not met**'} on FD001/FD003. Both are reported, not "
            "re-tuned: a fix needs data no model has seen.")

st.subheader("Retraining flags by shift")
scen = SCEN[st.radio("Scenario", list(SCEN), key="scenario")]
r = runs[runs.scenario == scen]
chosen = st.multiselect("Datasets", sorted(r.dataset.unique()), default=sorted(r.dataset.unique()), key="datasets")
r = r[r.dataset.isin(chosen)].sort_values(["dataset", "shift"])
if C.has_rows(r, "runs"):
    r = r.assign(flag=r.needs_retraining.astype(int))
    r["share"] = 100 * r.groupby("dataset").flag.transform(lambda s: s.rolling(10, min_periods=1).mean())
    DASH = {"FD001": [1, 0], "FD002": [1, 0], "FD003": [6, 3], "FD004": [6, 3]}
    COL = {"FD001": C.OKABE_ITO["blue"], "FD002": C.OKABE_ITO["blue"], "FD003": C.OKABE_ITO["vermillion"], "FD004": C.OKABE_ITO["vermillion"]}
    dom = sorted(r.dataset.unique())
    lines = alt.Chart(r).mark_line(strokeWidth=2).encode(
        x=alt.X("shift:Q", title="Shift (8-hour shift number)"),
        y=alt.Y("share:Q", title="Runs flagged, last 10 (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("dataset:N", title="Dataset", scale=alt.Scale(domain=dom, range=[COL[d] for d in dom])),
        strokeDash=alt.StrokeDash("dataset:N", title="Dataset", scale=alt.Scale(domain=dom, range=[DASH[d] for d in dom])),
        tooltip=[alt.Tooltip("dataset:N", title="Dataset"), alt.Tooltip("shift:Q", title="Shift"),
                 alt.Tooltip("share:Q", title="Runs flagged, last 10 (%)", format=".0f")])
    # no line-end labels: the lines end at the same point and the labels collided at phone width (D5.5);
    # the legend names each line by colour and dash
    layers = [lines]
    if scen != "phase7_v1":  # the C1 limit belongs to the Phase 7b rule; Phase 7 had a drift target instead
        lim = alt.Chart().mark_rule(color=C.OKABE_ITO["black"], strokeDash=[2, 2]).encode(y=alt.datum(100 * limit))
        lim_lab = alt.Chart().mark_text(align="left", dx=4, dy=-6).encode(x=alt.value(0), y=alt.datum(100 * limit),
                                                                         text=alt.value(f"{C.pct(limit, 0)} limit (C1)"))
        layers += [lim, lim_lab]
    st.altair_chart(alt.layer(*layers).properties(height=300), width="stretch")
    # the note matches the scenario shown (review 2026-11-15: it described rule v2 while rule v1 was selected)
    if scen == "phase7_v1":
        st.caption(f"Shift: {C.define('shift')} Phase 7, rule v1: a run is flagged when drift, the error ratio or "
                   f"interval misses trip; its pre-registered target was on drift false alarms ({C.pct(target, 0)}), so "
                   "no flag limit is drawn.")
    else:
        st.caption(f"Shift: {C.define('shift')} Clean-run flag rate: {C.define('clean_run_flag_rate')}")
