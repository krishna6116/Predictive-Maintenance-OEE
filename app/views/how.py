"""How it works (v1.2): the pipeline from a sensor reading to a maintenance decision, as a diagram and seven short stages.

Every number comes from a snapshot (facts.json, models.csv, plan.csv, oee_policies.csv) and is recomputed independently
by dashboards/verify_v12.py. Numbers that are not in the snapshots (sensors kept, features built) are described in
words, with a link to the docs, instead of being typed in by hand.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import common as C

C.header("how", "From NASA engine sensor readings to a crew schedule, a simulated OEE result and the checks that say when "
                "to retrain.")

models, plan, policies = C.load("models.csv"), C.load("plan.csv"), C.load("oee_policies.csv")
pub = C.fact("published_model")
fd3 = models[models.dataset == "FD003"].set_index("model")
rmse_pub, rmse_xgb = fd3.loc[pub, "rmse"], fd3.loc["XGBoost (168 features)", "rmse"]
thr, crews, days = C.fact("alert_threshold"), C.fact("crews_per_day"), C.fact("horizon_days")
pooled = policies[(policies.scope == "Pooled") & (policies.kind == "policy")].set_index("policy")


def plan_counts(ds: str) -> dict:
    p = plan[plan.dataset == ds]
    return {"flagged": len(p), "booked": int(p.scheduled_day.notna().sum()),
            "left": int((p.status == "flagged, no crew in horizon").sum()), "status": C.fact(f"solver_status_{ds}")}


pc = {ds: plan_counts(ds) for ds in ("FD001", "FD003")}

# ---- the diagram: top to bottom so it stays readable on a phone; Okabe-Ito colours, the simulated stage dashed + labelled
STAGES = ["Data", "Features", "Forecast", "Alert", "Schedule", "OEE (simulated)", "Monitoring"]
SUB = {"Data": "NASA sensor readings", "Features": "rolling statistics", "Forecast": f"{pub} + interval",
       "Alert": "lower bound at threshold", "Schedule": "CP-SAT crew plan", "OEE (simulated)": "3 policies replayed",
       "Monitoring": "drift and error checks"}
blue, verm, black = C.OKABE_ITO["blue"], C.OKABE_ITO["vermillion"], C.OKABE_ITO["black"]
nodes = []
for i, s in enumerate(STAGES):
    style, colour = ("rounded,dashed", verm) if "simulated" in s else ("rounded", blue)
    nodes.append(f'  n{i} [label="{s}\\n{SUB[s]}", style="{style}", color="{colour}", penwidth=2];')
edges = [f"  n{i} -> n{i + 1};" for i in range(len(STAGES) - 1)]
edges.append(f'  n{len(STAGES) - 1} -> n2 [style=dashed, label=" retrain flag", fontsize=11, constraint=false];')
DOT = "\n".join([
    "digraph pipeline {",
    "  rankdir=TB; nodesep=0.25; ranksep=0.25; bgcolor=transparent;",
    f'  node [shape=box, fontname="Helvetica", fontsize=11, fontcolor="{black}", margin="0.15,0.06"];',
    f'  edge [color="{black}", fontname="Helvetica", fontsize=9, fontcolor="{black}"];',
    *nodes, *edges, "}"])

# equal columns: with 2:3 the diagram shrank to about 218 px at a 1,024 px window (review 2026-11-15)
left, right = st.columns(2)
with left:
    st.graphviz_chart(DOT, alt="Pipeline diagram: " + " -> ".join(STAGES) + "; Monitoring feeds a retrain flag back to "
                      "Forecast. The OEE stage is simulated.")
    st.caption("Solid boxes run on real NASA data and real model outputs; the dashed box is the simulated plant. "
               "The dashed arrow is the retraining flag.")
with right:
    st.markdown(f"""
**The one-minute version**

1. Engines report sensor readings every cycle (NASA's C-MAPSS turbofan simulation).
2. A deep learning model ({pub}) forecasts each engine's remaining life, with an uncertainty range.
3. An engine is flagged when even the pessimistic end of that range is close to failure.
4. An optimizer books flagged engines onto the few maintenance crews available, most urgent first.
5. A simulated plant replays three maintenance policies to show the effect on OEE, downtime and failures.
6. Every shift, monitoring checks whether the forecasts can still be trusted.
""")

st.caption(f"{C.plain('MC dropout')} {C.plain('CP-SAT')}")
st.caption(f"{C.plain('OEE')} {C.plain('drift')}")

# ---- the stages, in diagram order: what happens, why it matters, result, where to see it
o = {p: pooled.loc[p] for p in ("Predictive", "Fixed interval", "Run to failure")}
SECTIONS = [
    ("Data", "Engine drill-down",
     "NASA's run-to-failure engine data is ingested into a PostgreSQL star schema. Every reading is validated; faulty rows "
     "are set aside in a quarantine table instead of being loaded.",
     "Forecasts are only as good as the data behind them, and a plant's data is never perfectly clean.",
     f"All four NASA datasets are used. The loaded data was checked cell for cell against NASA's files ([pipeline notes]({C.DOCS}/PIPELINE.md))."),
    ("Features", "Engine drill-down",
     "Sensors that do not change with wear are dropped. For the rest, rolling averages, trends and frequency features are "
     "built from each engine's past readings only, so nothing from the future leaks in.",
     "Raw readings are noisy; trends over recent cycles show wear much more clearly.",
     f"Remaining life used for training is capped at {C.fact('rul_cap')} cycles, because a healthy engine shows no sign of "
     f"how long it will last. Engines are split by engine, never by row ([feature notes]({C.DOCS}/FEATURES.md))."),
    ("Forecast", "Fleet health",
     f"A {pub} network reads each engine's recent history and forecasts its remaining life. MC dropout runs it "
     f"{C.fact('interval_passes')} times to give a {C.pct(C.fact('interval_nominal'), 0)} interval.",
     "A planner needs both a number and how sure the model is; acting on the pessimistic edge avoids surprise failures.",
     f"On {int(fd3.loc[pub, 'engines'])} engines it never saw (NASA FD003), RMSE {C.num(rmse_pub)} cycles for the {pub} against {C.num(rmse_xgb)} for "
     "XGBoost. The model was chosen on validation engines; the test set only confirmed it."),
    ("Alert", "Fleet health",
     f"An engine is flagged when the lower end of its interval is at or below {thr} cycles.",
     "Using the pessimistic edge means a confident but wrong forecast does not let an engine run to failure.",
     f"Engines flagged at the last observed cycle: {pc['FD001']['flagged']} in FD001 and {pc['FD003']['flagged']} in FD003."),
    ("Schedule", "Maintenance plan",
     f"Google OR-Tools' CP-SAT optimizer books flagged engines onto {crews} crews a day over a {days}-day horizon, most "
     "urgent first.",
     "Crews are the real limit in a plant: the question is not only which engines are at risk but which ones get serviced first.",
     f"FD001: {pc['FD001']['booked']} of {pc['FD001']['flagged']} flagged engines booked, {pc['FD001']['left']} left without "
     f"a crew in the horizon (solver {pc['FD001']['status']}). FD003: {pc['FD003']['booked']} of {pc['FD003']['flagged']} "
     f"booked, {pc['FD003']['left']} left (solver {pc['FD003']['status']}). The crew calendar is a stated assumption."),
    ("OEE (simulated)", "OEE policies (simulated)",
     "A simulated plant replays the same engines under three policies: predictive (this model), a fixed service interval, "
     "and running to failure.",
     "OEE, downtime and failures are the language operations managers use to judge maintenance.",
     f"Simulated, pooled over {int(o['Predictive'].machines)} machines: OEE {C.pct(o['Predictive'].oee)} predictive, "
     f"{C.pct(o['Fixed interval'].oee)} fixed interval, {C.pct(o['Run to failure'].oee)} run to failure; unplanned "
     f"failures {int(o['Predictive'].unplanned_failures)}, {int(o['Fixed interval'].unplanned_failures)} and "
     f"{int(o['Run to failure'].unplanned_failures)}. Simulated plant: no real plant was measured."),
    ("Monitoring", "Monitoring",
     "A Prefect flow runs once per shift: it scores new readings, checks the sensors for drift and, once true failures "
     "are known, checks the forecast error, and raises a retrain flag when the rules say so.",
     "A model in production slowly goes stale; the plant needs an honest alarm for when to stop trusting it.",
     "The rules were written down before the test runs. On FD004 the drift rule raised too many false alarms and failed its "
     "pre-registered criterion, and the first version's false-alarm target was not met either. Both are reported, not re-tuned."),
]
st.subheader("The seven stages")
for i, (stage, page_name, what, why, result) in enumerate(SECTIONS, 1):
    with st.container(border=True):
        st.markdown(f"**{i}. {stage}**")
        st.markdown(f"- **What happens:** {what}\n- **Why it matters:** {why}\n- **Result:** {result}\n"
                    f"- **See it:** the *{page_name}* page in the menu")
