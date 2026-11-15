"""Engine drill-down (v1.1): one test engine's sensors, rolling forecast with its 90% interval, first alert and service day."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import streamlit as st

import common as C

C.header("engine", "Pick a test engine: its sensor readings, the forecast remaining life shift by shift, the alert and the "
                   "maintenance plan it led to.")

hist = C.load("engine_history.csv")
fleet = C.load("fleet.csv")
plan = C.load("plan.csv")
thr = C.fact("alert_threshold")
crews = C.fact("crews_per_day")
nominal = C.fact("interval_nominal")
passes = C.fact("interval_passes")

# Sensor names: Saxena, Goebel, Simon and Eklund (2008), Table 2 (dashboards/v11_spec.json engine_page.sensors)
SENSORS = {"s11": ("Static pressure at HPC outlet", "Ps30", "psia"), "s4": ("Total temperature at LPT outlet", "T50", "°R"),
           "s12": ("Ratio of fuel flow to Ps30", "phi", "pps/psi"), "s7": ("Total pressure at HPC outlet", "P30", "psia")}


def default_engine(ds: str) -> int:
    """The flagged engine with the lowest lower bound at its last cycle (ties: lowest engine id)."""
    f = fleet[(fleet.dataset == ds) & fleet.flagged.astype(bool)].sort_values(["rul_low", "engine_id"], kind="mergesort")
    return int(f.engine_id.iloc[0])


ds = st.radio("Dataset", ["FD001", "FD003"], horizontal=True, key="dataset")
ids = sorted(int(e) for e in hist[hist.dataset == ds].engine_id.unique())
# one selector per dataset, so switching dataset opens that dataset's most urgent engine
eng = st.selectbox("Test engine", ids, index=ids.index(default_engine(ds)), key=f"engine_{ds}",
                   help="Opens on the most urgent flagged engine: the lowest lower bound at its last reading.")
h = hist[(hist.dataset == ds) & (hist.engine_id == eng)].sort_values("cycle")
if not C.has_rows(h, "cycles for this engine"):
    st.stop()

last = h.iloc[-1]
fr = fleet[(fleet.dataset == ds) & (fleet.engine_id == eng)].iloc[0]
pr = plan[(plan.dataset == ds) & (plan.engine_id == eng)]
cross = h[h.rul_low <= thr]
if pr.empty:
    service = "Not flagged"
elif pr.scheduled_day.notna().iloc[0]:
    service = f"Service day {int(pr.scheduled_day.iloc[0])}"
else:
    service = "No crew slot in the horizon"

# two per row, so the labels are not cut off at desktop width
m = st.columns(2)
m[0].metric("Forecast remaining life at the last reading (cycles)", f"{last.rul_mc_mean:.1f}",
            help=f"{C.define('RUL')} The shift-by-shift forecast at this engine's last observed cycle.")
m[1].metric(f"{C.pct(nominal, 0)} interval at the last reading (cycles)", f"{last.rul_low:.1f} to {last.rul_high:.1f}",
            help=C.define("interval"))
n = st.columns(2)
n[0].metric("Observed cycles", f"{len(h)}", help="Cycles of sensor data NASA's test file has for this engine before it stops.")
n[1].metric(f"First cycle the lower bound reached {thr} cycles", f"{int(cross.cycle.iloc[0])}" if len(cross) else "Never",
            help=f"The first cycle at which the {C.pct(nominal, 0)} interval's lower edge was at or below the alert threshold.")
o = st.columns(2)
o[0].metric("Flagged for service", "Yes" if bool(fr.flagged) else "No", help=C.define("flagged"))
o[1].metric("Maintenance plan", service,
            help=f"From the CP-SAT crew schedule at the last observed cycle; day 0 is that day. {crews} crews a day, "
                 "so a service day is one of that day's crew slots.")
if service.startswith("Service day"):
    st.caption(f"{service} (one of {crews} crew slots that day). Day 0 is the last observed day.")
st.caption(f"The flag and the plan use the forecast the plan acted on (Fleet health page); the line below is the "
           f"shift-by-shift replay, which runs {passes} MC dropout passes again and can differ slightly at the last cycle "
           "(see About and method). MC dropout: the network is run many times with random parts switched off, and the spread "
           "of its answers gives the interval.")

st.subheader("Forecast remaining life, shift by shift")
base = alt.Chart(h).encode(x=alt.X("cycle:Q", title="Engine cycle (count)"))
band = base.mark_area(opacity=0.35, color=C.OKABE_ITO["sky"]).encode(
    y=alt.Y("rul_low:Q", title="Remaining useful life (cycles)"), y2="rul_high:Q",
    tooltip=[alt.Tooltip("cycle:Q", title="Cycle"), alt.Tooltip("rul_low:Q", title="Lower bound (cycles)", format=".1f"),
             alt.Tooltip("rul_high:Q", title="Upper bound (cycles)", format=".1f")])
line = base.mark_line(color=C.OKABE_ITO["blue"], strokeWidth=2).encode(
    y="rul_mc_mean:Q", tooltip=[alt.Tooltip("cycle:Q", title="Cycle"), alt.Tooltip("rul_mc_mean:Q", title="Forecast (cycles)", format=".1f")])
alert = alt.Chart().mark_rule(color=C.OKABE_ITO["black"], strokeDash=[4, 3]).encode(y=alt.datum(thr))
alert_lab = alt.Chart().mark_text(align="right", dx=-4, dy=-6, color=C.OKABE_ITO["black"]).encode(
    x=alt.value({"expr": "width"}), y=alt.datum(thr), text=alt.value(f"Alert: {thr} cycles"))
st.altair_chart(alt.layer(band, line, alert, alert_lab).properties(height=260), width="stretch")
st.caption(f"Blue line: forecast remaining life. Light band: its {C.pct(nominal, 0)} interval. Dashed line: the alert "
           "threshold; the engine is flagged when the band's lower edge reaches it.")

st.subheader("Sensor readings")
g = st.columns(2)
for i, (code, (name, symbol, unit)) in enumerate(SENSORS.items()):
    ch = alt.Chart(h).mark_line(color=C.OKABE_ITO["blue"]).encode(
        x=alt.X("cycle:Q", title="Engine cycle (count)"),
        y=alt.Y(f"{code}:Q", title=f"{symbol} ({unit})", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("cycle:Q", title="Cycle"), alt.Tooltip(f"{code}:Q", title=f"{symbol} ({unit})")])
    # heading outside the chart: a Vega title was clipped at the top of the second row
    g[i % 2].markdown(f"**{name} ({code})**")
    g[i % 2].altair_chart(ch.properties(height=170), width="stretch")
st.caption("Sensor names: A. Saxena, K. Goebel, D. Simon and N. Eklund, \"Damage Propagation Modeling for Aircraft Engine "
           "Run-to-Failure Simulation\", PHM08, Table 2 (NASA NTRS 20090029214). HPC: high-pressure compressor; LPT: "
           "low-pressure turbine. These four have the strongest correlation with remaining life in the FD001 training data "
           "(docs/EDA_FINDINGS.md).")
