"""Home: what the project built and found, in one screen, with a guide to the other pages (Phase 9; v1.1 at-a-glance)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

import common as C

# Home keeps its header short (title, one line on what the app does, business question) so the results fit on the
# first screen; the longer explanation is in "What this shows / why it matters" below the charts
C.header("home", "Predicts when jet engines need maintenance from sensor data, then plans crew work.", explain=False)

models = C.load("models.csv")
pol = C.load("oee_policies.csv")
pub = C.fact("published_model")
fd3 = models[models.dataset == "FD003"].set_index("model")
pooled = pol[(pol.scope == "Pooled") & (pol.kind == "policy")].set_index("policy")
machines = int(pooled.loc["Predictive", "machines"])
ORDER = ["Predictive", "Fixed interval", "Run to failure"]

# 1. The forecast, in plain English
f = st.columns(2)
f[0].metric(f"{pub} forecast error on engines it never saw (cycles)", C.num(fd3.loc[pub, "rmse"]),
            help=f"{C.define('RMSE')} The published {pub} model on engines no model choice was made on.")
f[1].metric("XGBoost, previous model, same engines (cycles)", C.num(fd3.loc["XGBoost (168 features)", "rmse"]),
            help=f"{C.define('RMSE')} XGBoost on engineered features: the published model until the {pub} beat it on "
                 "a fresh dataset with tests fixed in advance.")
st.caption(f"Lower is better. RMSE is the typical size of a forecast's error, in engine cycles; measured on the "
           f"{int(fd3.loc[pub, 'engines'])} test engines of NASA FD003.")

# 3. The three maintenance policies on the same simulated plant: horizontal bars stay readable in narrow columns
COLOURS = [C.OKABE_ITO["blue"], C.OKABE_ITO["orange"], C.OKABE_ITO["vermillion"]]


def policy_chart(field: str, title: str, x_title: str, fmt) -> alt.LayerChart:
    d = pd.DataFrame({"policy": ORDER, "value": [float(pooled.loc[p, field]) for p in ORDER]})
    d["shown"] = 100 * d.value if field == "oee" else d.value
    d["label"] = d.value.map(fmt)
    bars = alt.Chart(d, title=title).mark_bar().encode(
        y=alt.Y("policy:N", title="Policy", sort=ORDER, axis=alt.Axis(labelOverlap=False, labelFontSize=10)),
        # OEE cannot exceed 100%; the count axes get room for the value labels
        x=alt.X("shown:Q", title=x_title, scale=alt.Scale(domain=[0, 100]) if field == "oee"
                else alt.Scale(domainMax=float(d.shown.max()) * 1.35)),
        color=alt.Color("policy:N", sort=ORDER, scale=alt.Scale(domain=ORDER, range=COLOURS), legend=None),
        tooltip=[alt.Tooltip("policy:N", title="Policy"), alt.Tooltip("label:N", title=title)])
    text = bars.mark_text(align="left", dx=3, color=C.OKABE_ITO["black"]).encode(text="label:N", color=alt.value(C.OKABE_ITO["black"]))
    return alt.layer(bars, text).properties(height=165)


g = st.columns(3)
g[0].altair_chart(policy_chart("oee", "OEE (simulated)", "OEE (%)", C.pct), width="stretch")
g[1].altair_chart(policy_chart("unplanned_failures", "Unplanned failures (simulated, count)", "Failures (count)", lambda v: f"{v:,.0f}"),
                  width="stretch")
g[2].altair_chart(policy_chart("downtime_hours", "Downtime (simulated, hours)", "Downtime (hours)", lambda v: f"{v:,.0f}"), width="stretch")
# 2. Business impact (simulated), the business message, right below the policy charts (option 3, author-approved 2026-11-15). No separate heading (author-approved
#    2026-11-15, V2.1 fold): the table's first column header 'Impact (simulated)' carries the label.
fl = {p: int(pooled.loc[p, "unplanned_failures"]) for p in ORDER}
dt = {p: float(pooled.loc[p, "downtime_hours"]) for p in ORDER}
wc = {p: int(pooled.loc[p, "wasted_cycles"]) for p in ORDER}
# a small table, so each number says what it is compared with (tiles would cut the long labels off)
st.dataframe(pd.DataFrame([
    {"Impact (simulated)": "Failures avoided (simulated)", "vs fixed interval": f"{fl['Fixed interval'] - fl['Predictive']:,}",
     "vs run to failure": f"{fl['Run to failure'] - fl['Predictive']:,}"},
    {"Impact (simulated)": "Downtime hours saved (simulated)", "vs fixed interval": f"{dt['Fixed interval'] - dt['Predictive']:,.0f}",
     "vs run to failure": f"{dt['Run to failure'] - dt['Predictive']:,.0f}"},
    {"Impact (simulated)": "Wasted engine life, cycles (simulated): predictive vs", "vs fixed interval": f"{wc['Predictive']:,} vs {wc['Fixed interval']:,}",
     "vs run to failure": f"{wc['Predictive']:,} vs {wc['Run to failure']:,}"},
]), hide_index=True, width="stretch")
st.caption(f"Predictive maintenance compared with the other two policies, on a simulated plant of {machines} machines.")

C.simulated_notice()
C.explain_why("home")

st.subheader("How to read this app (one minute)")
st.markdown("\n".join(f"- **{name}:** {C.BUSINESS[page]}" for page, name in (
    ("fleet", "Fleet health"), ("plan", "Maintenance plan"), ("oee", "OEE policies (simulated)"),
    ("cost", "Cost what-if (illustrative)"), ("monitoring", "Monitoring"), ("about", "About and method"))))
st.caption("Open a page from the menu on the left (on a phone, the arrow at the top left).")

st.subheader("More")
l1, l2, l3 = st.columns(3)
l1.link_button("Code on GitHub", C.REPO)
l2.link_button("Case study", f"{C.DOCS}/CASE_STUDY.md")
l3.link_button("Author on LinkedIn", C.LINKEDIN)
