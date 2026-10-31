"""Fleet health: published LSTM forecasts with MC-dropout intervals, FD001 and FD003, and all models compared."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import streamlit as st

import common as C

thr, nominal = C.fact("alert_threshold"), C.fact("interval_nominal")
C.header("fleet", "Each test engine's predicted remaining useful life at its last observed cycle, from the published model, "
                  f"with its {C.pct(nominal, 0)} interval. Real NASA C-MAPSS test engines (RUL definition on the About page).")

fleet, models = C.load("fleet.csv"), C.load("models.csv")
choice = st.radio("Dataset", ["FD001", "FD003", "Both"], horizontal=True, key="dataset")
datasets = ["FD001", "FD003"] if choice == "Both" else [choice]

for ds in datasets:
    f = fleet[fleet.dataset == ds].sort_values(["rul_point", "engine_id"]).reset_index(drop=True)
    if not C.has_rows(f, f"{ds} engines"):
        continue
    st.subheader(f"{ds}: {C.fact('published_model')} forecasts")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Engines ({ds})", str(len(f)), help="Test engines in the dataset, each at its last observed cycle.")
    c2.metric(f"Flagged ({ds})", str(int(f.flagged.sum())), help=C.define("flagged"))
    c3.metric(f"Interval coverage on these engines ({ds})", C.pct(f.inside_interval.mean()), help=C.define("coverage"))
    st.caption(f"Nominal {C.pct(nominal, 0)} interval from {C.fact('interval_passes')} MC-dropout passes. Stated coverage "
               f"for {ds}: {C.pct(C.fact(f'coverage_stated_{ds}'), 0)} ({C.fact(f'coverage_kind_{ds}')}). "
               f"{C.define('interval')}")

    f = f.assign(order=range(1, len(f) + 1), status=f.flagged.map({True: "Flagged", False: "Not flagged"}))
    base = alt.Chart(f).encode(x=alt.X("order:O", title="Engine (sorted by predicted RUL)", axis=alt.Axis(labels=False, ticks=False)))
    bars = base.mark_rule(color=C.OKABE_ITO["sky"]).encode(
        y=alt.Y("rul_low:Q", title="Remaining useful life (cycles)"), y2="rul_high:Q")
    dots = base.mark_point(filled=True, size=45).encode(
        y="rul_point:Q",
        color=alt.Color("status:N", title="Status", scale=alt.Scale(domain=["Flagged", "Not flagged"],
                                                                      range=[C.OKABE_ITO["vermillion"], C.OKABE_ITO["blue"]])),
        shape=alt.Shape("status:N", title="Status", scale=alt.Scale(domain=["Flagged", "Not flagged"], range=["triangle-up", "circle"])),
        tooltip=[alt.Tooltip("engine_id:Q", title="Engine"), alt.Tooltip("rul_point:Q", title="Predicted RUL (cycles)", format=".1f"),
                 alt.Tooltip("rul_low:Q", title="Lower bound (cycles)", format=".1f"),
                 alt.Tooltip("rul_high:Q", title="Upper bound (cycles)", format=".1f"),
                 alt.Tooltip("true_rul_capped:Q", title="True RUL (cycles)"), alt.Tooltip("status:N", title="Status")])
    rule = alt.Chart().mark_rule(color=C.OKABE_ITO["black"], strokeDash=[4, 3]).encode(y=alt.datum(thr))
    label = alt.Chart().mark_text(align="left", dx=4, dy=-6, color=C.OKABE_ITO["black"]).encode(
        x=alt.value(0), y=alt.datum(thr), text=alt.value(f"Alert threshold: lower bound <= {thr} cycles"))
    st.altair_chart(alt.layer(bars, dots, rule, label).properties(height=320), width="stretch")

    flagged = f[f.flagged][["engine_id", "cycle", "rul_point", "rul_low", "rul_high", "risk", "outcome"]]
    st.markdown(f"**Flagged engines ({ds})**: lower 90% bound at or below {thr} cycles")
    if C.has_rows(flagged, f"flagged {ds} engines"):
        st.dataframe(flagged.rename(columns={"engine_id": "Engine", "cycle": "Last cycle", "rul_point": "Predicted RUL (cycles)",
                                             "rul_low": "Lower bound (cycles)", "rul_high": "Upper bound (cycles)",
                                             "risk": "Risk", "outcome": "Outcome"}).round(1),
                     hide_index=True, width="stretch")

m = models[models.dataset.isin(datasets)].copy()
st.subheader(f"All models on the same {' and '.join(str(n) for n in sorted(m.engines.unique()))} test engines")
m["Model"] = m.model + m.published.map({True: " (published)", False: ""})
st.dataframe(m[["dataset", "Model", "rmse", "near_failure_mae", "engines", "near_failure_engines"]]
             .rename(columns={"dataset": "Dataset", "rmse": "RMSE (cycles)", "near_failure_mae": "Near-failure MAE (cycles)",
                              "engines": "Test engines", "near_failure_engines": "Near-failure engines"})
             .sort_values(["Dataset", "RMSE (cycles)"]).round(2), hide_index=True, width="stretch")
st.caption(f"RMSE: {C.define('RMSE')} Near-failure MAE: {C.define('near_failure_MAE')} "
           f"Published model rule: {C.fact('published_rule')}.")
