"""Predictive Maintenance & OEE: Phase 8 dashboard (Streamlit). Reads only dashboards/snapshots/.

Run locally:  streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="Predictive Maintenance & OEE", layout="wide")

pages = [st.Page("views/fleet.py", title="Fleet health", default=True),
         st.Page("views/plan.py", title="Maintenance plan"),
         st.Page("views/oee.py", title="OEE policies (simulated)"),
         st.Page("views/monitoring.py", title="Monitoring"),
         st.Page("views/about.py", title="About and method")]
st.navigation(pages).run()
