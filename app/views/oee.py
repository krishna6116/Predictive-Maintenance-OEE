"""OEE policies (SIMULATED plant): run to failure vs fixed interval vs predictive."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import streamlit as st

import common as C

C.header("oee", "Three maintenance policies replayed on the same simulated machines, each running one real NASA engine life.")
C.simulated_notice()

pol = C.load("oee_policies.csv")
ORDER = ["Run to failure", "Fixed interval", "Predictive"]
scope = st.radio("Scope", ["Pooled", "FD001", "FD003"], horizontal=True, key="scope")
t = pol[(pol.scope == scope) & (pol.kind == "policy")].set_index("policy").reindex(ORDER).reset_index()

if C.has_rows(t.dropna(subset=["oee"]), "policies"):
    cols = st.columns(3)
    for c, (_, r) in zip(cols, t.iterrows()):
        c.metric(f"OEE, {r.policy} ({scope}, simulated)", C.pct(r.oee), help=C.define("OEE"))

    t["oee_pct"] = 100 * t.oee
    t["label"] = t.oee.map(C.pct)
    COLOURS = [C.OKABE_ITO["vermillion"], C.OKABE_ITO["orange"], C.OKABE_ITO["blue"]]
    bars = alt.Chart(t).mark_bar().encode(
        x=alt.X("policy:N", title="Policy", sort=ORDER),
        y=alt.Y("oee_pct:Q", title="OEE (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("policy:N", title="Policy", sort=ORDER, scale=alt.Scale(domain=ORDER, range=COLOURS), legend=None),
        tooltip=[alt.Tooltip("policy:N", title="Policy"), alt.Tooltip("oee_pct:Q", title="OEE (%)", format=".1f")])
    text = bars.mark_text(dy=-8, color=C.OKABE_ITO["black"]).encode(text="label:N", color=alt.value(C.OKABE_ITO["black"]))
    st.altair_chart(alt.layer(bars, text).properties(height=300, title=f"OEE by policy, {scope} (simulated)"),
                    width="stretch")

    show = t[["policy", "availability", "performance", "quality", "oee", "unplanned_failures", "planned_services",
              "downtime_hours", "wasted_cycles"]].copy()
    for c in ("availability", "performance", "quality", "oee"):
        show[c] = show[c].map(C.pct)
    for c in ("unplanned_failures", "planned_services", "downtime_hours", "wasted_cycles"):
        show[c] = show[c].astype(int)
    st.dataframe(show.rename(columns={"policy": "Policy", "availability": "Availability", "performance": "Performance",
                                      "quality": "Quality", "oee": "OEE", "unplanned_failures": "Unplanned failures",
                                      "planned_services": "Planned services", "downtime_hours": "Downtime (hours)",
                                      "wasted_cycles": "Wasted life (cycles)"}), hide_index=True, width="stretch")

gaps = pol[pol.kind == "gap"].set_index("policy")
n_machines = int(pol[(pol.scope == "Pooled") & (pol.policy == "Predictive")].machines.iloc[0])
st.subheader(f"Is the difference real? (pooled, {n_machines} simulated machines)")
c1, c2 = st.columns(2)
for c, g in zip((c1, c2), ("Predictive - Fixed interval", "Predictive - Run to failure")):
    r = gaps.loc[g]
    c.metric(f"OEE gap: {g.replace(' - ', ' minus ')} (simulated)", C.points(r.oee_gap),
             help=f"Difference in pooled OEE, percentage points; 95% bootstrap interval over machines "
                  f"{C.points(r.ci_low)} to {C.points(r.ci_high)}.")
    c.caption(f"95% interval: {C.points(r.ci_low)} to {C.points(r.ci_high)}")
st.caption(f"{C.define('OEE')} Wasted life: cycles of engine life left unused at a planned service.")
with st.expander("Simulation assumptions"):
    for k, v in C.fact("plant_assumptions").items():
        st.markdown(f"- **{k.replace('_', ' ').capitalize()}:** {v}")
