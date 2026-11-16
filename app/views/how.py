"""How it works: the pipeline from sensor readings to a crew plan, one step at a time, with where to see each step.

A walkthrough for a first-time visitor (recruiter or hiring manager). It shows no new results: every number on the page
is read from facts.json (the same snapshot file the other pages use), and the results themselves stay on their pages.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import common as C

C.header("how", "From raw engine sensor readings to a maintenance plan the crews can follow, and how the system checks itself.")

pub = C.fact("published_model")
cap, thr = C.fact("rul_cap"), C.fact("alert_threshold")
crews, days = C.fact("crews_per_day"), C.fact("horizon_days")
passes, nominal = C.fact("interval_passes"), C.fact("interval_nominal")

# The pipeline as a diagram. Colour is paired with a second channel: the simulated step has a dashed border and says
# '(simulated)' in its label; the monitoring loop is a dashed arrow with its own label.
REAL, SIM, CHECK = C.OKABE_ITO["sky"], C.OKABE_ITO["orange"], C.OKABE_ITO["green"]
DOT = f"""
digraph pipeline {{
  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.35;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#000000", fillcolor="{REAL}"];
  edge [color="#000000", fontname="Helvetica", fontsize=9];
  data     [label="1. Sensor data\\nNASA C-MAPSS engines"];
  pipe     [label="2. Data pipeline\\nPostgreSQL, validation"];
  feat     [label="3. Features\\nper-engine history"];
  model    [label="4. Forecast\\n{pub} + interval"];
  alert    [label="5. Alerts\\nlower bound vs threshold"];
  plan     [label="6. Crew plan\\nCP-SAT optimizer"];
  plant    [label="7. Plant impact\\nOEE (simulated)", style="rounded,filled,dashed", fillcolor="{SIM}"];
  monitor  [label="8. Monitoring\\ndrift, error, retrain flag", fillcolor="{CHECK}"];
  data -> pipe -> feat -> model -> alert -> plan -> plant;
  model -> monitor [style=dashed, label="every shift"];
  monitor -> model [style=dashed, label="retrain if needed"];
}}
"""
st.graphviz_chart(DOT)
st.caption("Blue: steps run on the real NASA engine data. Orange with a dashed border: the simulated plant. Green: the "
           "monitoring loop that watches the forecasts every shift. Dashed arrows: the feedback loop.")

st.subheader("The eight steps")

STEPS = [
    ("1. Sensor data", "Data",
     "NASA's C-MAPSS turbofan data: simulated jet engines that each run until they fail, with a reading from every "
     "sensor once per cycle. Training engines run to failure; test engines stop earlier, and the model has to say how "
     "much life they had left.",
     "Real maintenance teams have exactly this: sensor logs, and a question about the future. The files are checked "
     "against pinned checksums, so every result is reproduced on byte-identical data.",
     None, "data/"),
    ("2. Data pipeline", "Data engineering",
     "Readings are loaded into a PostgreSQL star schema. A validation step sends bad rows (missing or impossible values) "
     "to a quarantine table instead of the model, and every load has a run ID, so re-running it never duplicates rows.",
     "A model is only as good as the data feeding it; a quarantine table makes bad data visible instead of silently "
     "skewing forecasts.",
     None, "pipeline/"),
    ("3. Features", "Feature engineering",
     "Each engine's recent history is summarised (rolling averages, trends) using only past readings, never future ones. "
     f"Engines, not rows, are split into training and validation, and remaining life is capped at {cap} cycles because "
     "the sensors barely change while an engine is healthy.",
     "Using only the past keeps the model honest: in a plant you never know tomorrow's readings. Splitting by engine "
     "stops the model from memorising an engine it will be tested on.",
     None, "features/"),
    ("4. Forecast", "Machine learning",
     f"Simple rules, Ridge regression, XGBoost, a 1D-CNN and an {pub} all forecast each engine's remaining life. "
     f"Each forecast comes with a {C.pct(nominal, 0)} interval from {passes} Monte Carlo dropout passes. The {pub} was "
     "published only after it beat XGBoost on a fresh dataset with the tests written down in advance.",
     "A single number is not enough to plan on; the interval says how sure the model is, so a planner can act on the "
     "pessimistic edge.",
     "Fleet health", "models/"),
    ("5. Alerts", "Decision rule",
     f"An engine is flagged when the lower edge of its interval reaches {thr} cycles, and it gets a risk level "
     "(critical, high, medium).",
     "Acting on the lower edge, not the middle, means a confident but wrong forecast does not lead to a failure.",
     "Fleet health · Engine drill-down", "maintenance/alerts.py"),
    ("6. Crew plan", "Operations research",
     f"Google OR-Tools' CP-SAT optimizer books flagged engines onto crews: {crews} a day over a {days}-day horizon "
     "(planning assumptions). The most urgent engines go first; a second method proves the plan is optimal.",
     "This is the trade-off a maintenance planner makes every day. When capacity runs short, the page shows which "
     "engines miss out, rather than hiding it.",
     "Maintenance plan", "maintenance/schedule.py"),
    ("7. Plant impact (simulated)", "Operations analytics",
     "Three policies run on the same simulated machines: predictive (this system), a fixed service interval, and run "
     "to failure. Each is scored on OEE (availability x performance x quality), unplanned failures, downtime and wasted "
     "engine life.",
     "It turns forecast accuracy into the language operations managers use. It is labelled simulated because no real "
     "plant was measured; the engine lives and forecasts behind it are real.",
     "OEE policies (simulated) · Cost what-if (illustrative)", "maintenance/simulate.py"),
    ("8. Monitoring", "MLOps",
     "A Prefect flow runs once per shift: it scores new readings, checks the sensors for drift and the forecasts for "
     "error, and raises a retraining flag when its rules say the model can no longer be trusted. Rules that missed "
     "their targets are shown as misses, not re-tuned.",
     "A model in production needs an honest alarm for when it stops working; reporting the misses is what makes the "
     "other numbers believable.",
     "Monitoring", "pipeline/flow.py"),
]

for title, skill, what, why, page, code in STEPS:
    with st.expander(f"{title}  ·  {skill}", expanded=title.startswith("1.")):
        st.markdown(f"**What happens:** {what}")
        st.markdown(f"**Why it matters:** {why}")
        see = f"Open **{page}** from the menu. " if page else ""
        st.markdown(f"{see}Code: [{code}]({C.REPO}/tree/main/{code})")

st.subheader("What is real and what is simulated")
st.table({
    "Part": ["Engine sensor data and remaining-life labels", f"{pub} forecasts, intervals, alerts and crew plans",
             "Monitoring runs (drift, error, retraining flag)", "Plant, OEE, downtime and failure counts",
             "Crew capacity and the planning calendar", "Costs on the cost page"],
    "Status": ["Real (NASA C-MAPSS)", "Real model outputs", "Real runs on the NASA test engines", "Simulated",
               "Planning assumptions", "Illustrative (your own figures)"],
})
C.simulated_notice()

st.subheader("How the numbers are kept honest")
st.markdown(
    "- **Decided before seeing results:** model choices, tests and each phase's success criteria were committed before "
    "the test data was scored.\n"
    "- **Validation only:** models were chosen on validation engines; the test engines only confirm.\n"
    "- **Fair comparisons:** every model is compared with simple baselines on the same engines.\n"
    "- **Independent checks in CI:** each phase has a separate script that recomputes its published numbers with "
    "different code, on every push. This app reads frozen snapshot files that are checked the same way "
    "(see **About and method**).")

st.subheader("About the builder")
st.markdown(
    f"**{C.AUTHOR}**, MS Business Analytics candidate at the Isenberg School of Management, UMass Amherst (expected "
    "May 2028). B.E. Mechanical Engineering, with industry experience in production, quality control and operations "
    "analytics in manufacturing, which is where the questions in this project come from.\n\n"
    "Open to Summer 2027 internships in data analytics, operations and supply chain analytics, quality analytics, "
    "healthcare analytics and AI product roles.")
b1, b2, b3 = st.columns(3)
b1.link_button("LinkedIn", C.LINKEDIN)
b2.link_button("Case study", f"{C.DOCS}/CASE_STUDY.md")
b3.link_button("Code on GitHub", C.REPO)
