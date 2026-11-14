"""Predictive Maintenance & OEE: Phase 8 dashboard (Streamlit). Reads only dashboards/snapshots/.

Run locally:  streamlit run app/streamlit_app.py
"""
import streamlit as st

import common as C

st.set_page_config(page_title="Predictive Maintenance & OEE", layout="wide")

pages = [st.Page("views/home.py", title="Home", default=True),
         st.Page("views/fleet.py", title="Fleet health"),
         st.Page("views/plan.py", title="Maintenance plan"),
         st.Page("views/oee.py", title="OEE policies (simulated)"),
         st.Page("views/monitoring.py", title="Monitoring"),
         st.Page("views/about.py", title="About and method")]
nav = st.navigation(pages)

with st.sidebar:
    st.divider()
    st.caption(f"Built by {C.AUTHOR} · [LinkedIn]({C.LINKEDIN}) · [GitHub]({C.REPO})")
    st.caption("Data: NASA C-MAPSS Turbofan Engine Degradation Simulation (NASA Prognostics Center of Excellence).")
    st.caption(f"Snapshot checksum `{C.manifest()['data_checksum'][:12]}` (full value on About and method).")

nav.run()
