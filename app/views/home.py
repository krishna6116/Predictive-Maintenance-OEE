"""Home: what the project built and found, in one screen, with a guide to the other pages (Phase 9)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import common as C

C.header("home", "Predicts when jet engines will need maintenance from their sensor data, plans crew work under crew "
                 "limits, and shows the effect on a simulated plant's OEE.")

models = C.load("models.csv")
pol = C.load("oee_policies.csv")
pub = C.fact("published_model")
fd3 = models[models.dataset == "FD003"].set_index("model")
pooled = pol[(pol.scope == "Pooled") & (pol.kind == "policy")].set_index("policy")
ORDER = ["Predictive", "Fixed interval", "Run to failure"]

st.subheader("Headline results")
r1 = st.columns(2)
r1[0].metric(f"{pub} forecast error on unseen engines (FD003 RMSE, cycles)", C.num(fd3.loc[pub, "rmse"]),
             help=f"{C.define('RMSE')} The published model on NASA FD003, a dataset no model choice was made on.")
r1[1].metric("XGBoost forecast error on unseen engines (FD003 RMSE, cycles)", C.num(fd3.loc["XGBoost (168 features)", "rmse"]),
             help=f"{C.define('RMSE')} The strongest non-deep baseline on the same engines.")
r2 = st.columns(2)
r2[0].metric("Simulated OEE: predictive / fixed / run to failure", " / ".join(C.pct(pooled.loc[p, "oee"]) for p in ORDER),
             help=f"{C.define('OEE')} Pooled over the {int(pooled.loc['Predictive', 'machines'])} simulated machines.")
r2[1].metric("Simulated unplanned failures: predictive / fixed / run to failure",
             " / ".join(str(int(pooled.loc[p, "unplanned_failures"])) for p in ORDER),
             help="Simulated machines whose engine failed before it was serviced, out of the same machines under each policy.")
C.simulated_notice()

st.subheader("How to read this app (one minute)")
st.markdown("\n".join(f"- **{name}:** {C.BUSINESS[page]}" for page, name in (
    ("fleet", "Fleet health"), ("plan", "Maintenance plan"), ("oee", "OEE policies (simulated)"),
    ("monitoring", "Monitoring"), ("about", "About and method"))))
st.caption("Open a page from the menu on the left (on a phone, the arrow at the top left).")

st.subheader("More")
l1, l2, l3 = st.columns(3)
l1.link_button("Code on GitHub", C.REPO)
l2.link_button("Case study", f"{C.DOCS}/CASE_STUDY.md")
l3.link_button("Author on LinkedIn", C.LINKEDIN)
