"""Maintenance plan: snapshot alerts and the CP-SAT crew schedule, plus the predictive-replay work orders."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

import common as C

C.header("plan", "Engines whose lower 90% bound is at or below the alert threshold, booked onto crews by the CP-SAT "
                 "scheduler at the last observed cycle (Phase 6).")

plan, orders = C.load("plan.csv"), C.load("work_orders.csv")
crews, days, thr = C.fact("crews_per_day"), C.fact("horizon_days"), C.fact("alert_threshold")
st.info(f"Planning assumption (guide value): {crews} crews a day, each able to service one engine, over a {days}-day "
        f"horizon; an engine is flagged when its lower 90% bound is <= {thr} cycles. The crew calendar is an assumption, "
        "the forecasts are real model outputs.")

choice = st.radio("Dataset", ["FD001", "FD003", "Both"], horizontal=True, key="dataset")
datasets = ["FD001", "FD003"] if choice == "Both" else [choice]
risks = st.multiselect("Risk levels in the tables", ["critical", "high", "medium"], default=["critical", "high", "medium"], key="risks")
RISK_COLOURS = {"critical": C.OKABE_ITO["vermillion"], "high": C.OKABE_ITO["orange"], "medium": C.OKABE_ITO["sky"]}

for ds in datasets:
    p = plan[plan.dataset == ds]
    st.subheader(ds)
    booked = p[p.scheduled_day.notna()]
    c = st.columns(5)
    c[0].metric(f"Crew slots in the horizon ({ds})", str(crews * days), help=f"{crews} crews a day x {days} days (assumption).")
    c[1].metric(f"Engines flagged ({ds})", str(len(p)), help=C.define("flagged"))
    c[2].metric(f"Scheduled ({ds})", str(len(booked)), help="Flagged engines booked onto a crew within the horizon.")
    c[3].metric(f"Flagged, no crew in horizon ({ds})", str(int((p.status == "flagged, no crew in horizon").sum())),
                help="Flagged engines left over because every crew slot in the horizon is taken by a more urgent engine.")
    c[4].metric(f"Solver status ({ds})", C.fact(f"solver_status_{ds}"),
                help="CP-SAT status; OPTIMAL means no cheaper plan exists under the cost weights (proved again in Phase 6 "
                     "by a second method).")
    if not C.has_rows(booked, f"scheduled {ds} engines"):
        continue
    per_day = booked.groupby(["scheduled_day", "risk"]).size().rename("engines").reset_index()
    per_day["scheduled_day"] = per_day.scheduled_day.astype(int)
    bars = alt.Chart(per_day).mark_bar().encode(
        x=alt.X("scheduled_day:O", title="Plan day (0 = today)"),
        y=alt.Y("engines:Q", title="Engines serviced (count)", scale=alt.Scale(domain=[0, crews + 1])),
        color=alt.Color("risk:N", title="Risk", scale=alt.Scale(domain=list(RISK_COLOURS), range=list(RISK_COLOURS.values()))),
        order=alt.Order("risk:N"),
        tooltip=[alt.Tooltip("scheduled_day:O", title="Plan day"), alt.Tooltip("risk:N", title="Risk"),
                 alt.Tooltip("engines:Q", title="Engines")])
    text = alt.Chart(per_day).mark_text(dy=10, color=C.OKABE_ITO["black"]).encode(
        x="scheduled_day:O", y=alt.Y("engines:Q", stack="zero"), text=alt.Text("risk:N"), order=alt.Order("risk:N"))
    limit = alt.Chart().mark_rule(color=C.OKABE_ITO["black"], strokeDash=[4, 3]).encode(y=alt.datum(crews))
    lab = alt.Chart().mark_text(align="left", dx=4, dy=-6).encode(x=alt.value(0), y=alt.datum(crews),
                                                                 text=alt.value(f"Crew limit: {crews} per day"))
    st.altair_chart(alt.layer(bars, text, limit, lab).properties(height=260), width="stretch")

    counts = p.risk.value_counts().reindex(["critical", "high", "medium"]).fillna(0).astype(int)
    st.dataframe(pd.DataFrame({"Risk": counts.index, "Engines flagged": counts.values}), hide_index=True)
    rows = p[p.risk.isin(risks)].sort_values(["scheduled_day", "engine_id"])
    if C.has_rows(rows, f"{ds} alerts"):
        st.dataframe(rows[["engine_id", "risk", "reason", "scheduled_day", "status"]].rename(columns={
            "engine_id": "Engine", "risk": "Risk", "reason": "Why flagged", "scheduled_day": "Plan day", "status": "Status"}),
            hide_index=True, width="stretch")

st.subheader("Work orders in the predictive replay (Phase 6)")
o = orders[orders.dataset.isin(datasets)]
if C.has_rows(o, "work orders"):
    st.dataframe(o.groupby(["dataset", "status"]).size().rename("Work orders").reset_index()
                 .rename(columns={"dataset": "Dataset", "status": "Status"}), hide_index=True)
    st.caption("Day-by-day replay over each engine's whole life: an order is created when the lower 90% bound first reaches "
               "the threshold; 'cancelled: failed first' means the engine failed before its booked service.")
