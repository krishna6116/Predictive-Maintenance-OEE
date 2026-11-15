"""Cost what-if (v1.1, ILLUSTRATIVE): the viewer's cost figures applied to the simulated counts of each policy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

import common as C
from costmodel import DEFAULTS, POLICIES, Counts, compare

C.header("cost", "Choose your own cost figures; the counts behind them come from the simulated plant (Phase 6).")
st.warning("**Illustrative:** you choose the costs. The counts and hours come from a simulated plant. "
           "These are not real cost data.")
C.simulated_notice()

pol = C.load("oee_policies.csv")
pooled = pol[(pol.scope == "Pooled") & (pol.kind == "policy")].set_index("policy")
machines = int(pooled.loc["Predictive", "machines"])
counts = {p: Counts(*(float(pooled.loc[p, k]) for k in ("downtime_hours", "planned_services", "unplanned_failures", "wasted_cycles")))
          for p in POLICIES}

st.subheader("Your cost figures")
st.caption("Illustrative defaults, change them: round example values chosen for the demo, not taken from any data.")
i = st.columns(4)
costs = dict(
    c_hour=i[0].number_input("Cost per hour of downtime ($)", min_value=0, value=DEFAULTS["c_hour"], step=100, key="c_hour"),
    c_service=i[1].number_input("Cost per planned service ($)", min_value=0, value=DEFAULTS["c_service"], step=500, key="c_service"),
    c_failure=i[2].number_input("Extra repair cost per unplanned failure ($)", min_value=0, value=DEFAULTS["c_failure"], step=1000,
                                key="c_failure"),
    c_cycle=i[3].number_input("Cost per cycle of wasted engine life ($)", min_value=0, value=DEFAULTS["c_cycle"], step=5, key="c_cycle"),
)
st.markdown("**Formula:** total = downtime hours × cost per hour + planned services × cost per service "
            "+ unplanned failures × extra repair cost + wasted cycles × cost per cycle")
st.caption("Downtime hours already include the downtime of failures and services, so the failure cost here is only the "
           "extra repair cost (parts, call-out), not its lost production time: nothing is counted twice.")

r = compare(counts, **costs)
money = lambda x: f"${x:,.0f}"  # noqa: E731
st.subheader("Estimated total cost per policy (illustrative)")
c = st.columns(3)
for col, p in zip(c, POLICIES):
    col.metric(f"Total cost, {p.lower()} (illustrative)", money(r["totals"][p]),
               help=f"Your cost figures applied to this policy's simulated counts over the {machines} simulated machines.")
s = st.columns(2)
s[0].metric("Saving of predictive vs fixed interval (illustrative)", money(r["saving_vs_fixed"]),
            help="Total cost of the fixed-interval policy minus total cost of predictive maintenance, with your figures.")
s[1].metric("Saving of predictive vs run to failure (illustrative)", money(r["saving_vs_rtf"]),
            help="Total cost of running to failure minus total cost of predictive maintenance, with your figures.")

ORDER = ["Run to failure", "Fixed interval", "Predictive"]
COLOURS = [C.OKABE_ITO["vermillion"], C.OKABE_ITO["orange"], C.OKABE_ITO["blue"]]
d = pd.DataFrame({"policy": ORDER, "cost": [r["totals"][p] for p in ORDER]})
d["label"] = d.cost.map(money)
bars = alt.Chart(d).mark_bar().encode(
    x=alt.X("policy:N", title="Maintenance policy", sort=ORDER),
    y=alt.Y("cost:Q", title="Total cost (illustrative, $)"),
    color=alt.Color("policy:N", title="Policy", sort=ORDER, scale=alt.Scale(domain=ORDER, range=COLOURS), legend=None),
    tooltip=[alt.Tooltip("policy:N", title="Policy"), alt.Tooltip("label:N", title="Total cost (illustrative)")])
text = bars.mark_text(dy=-8, color=C.OKABE_ITO["black"]).encode(text="label:N", color=alt.value(C.OKABE_ITO["black"]))
st.altair_chart(alt.layer(bars, text).properties(height=280), width="stretch")

st.subheader("Simulated counts used")
st.dataframe(pd.DataFrame([{"Policy": p, "Downtime hours": f"{counts[p].downtime_hours:,.0f}",
                            "Planned services": f"{counts[p].planned_services:,.0f}",
                            "Unplanned failures": f"{counts[p].unplanned_failures:,.0f}",
                            "Wasted cycles": f"{counts[p].wasted_cycles:,.0f}"} for p in POLICIES]),
             hide_index=True, width="stretch")
st.caption(f"Pooled over the {machines} simulated machines of Phase 6 (oee_policies.csv). Simulated plant: not measured data.")
