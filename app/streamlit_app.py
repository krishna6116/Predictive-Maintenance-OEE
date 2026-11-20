"""Predictive Maintenance & OEE: Phase 8 dashboard (Streamlit). Reads only dashboards/snapshots/.

Run locally:  streamlit run app/streamlit_app.py
"""
import importlib
import sys
from pathlib import Path

import streamlit as st

# A running app re-runs the page scripts after a code update but keeps modules it imported earlier, so a page could
# meet an old app/common.py (the preview crash of 2026-11-15). Re-load the app's own modules on every run; a module of
# the same name from another folder (tests) is dropped so the import below finds this app's copy.
APP = Path(__file__).resolve().parent
for _name in ("common", "costmodel"):
    _mod = sys.modules.get(_name)
    if _mod is not None:
        if Path(getattr(_mod, "__file__", "") or "").resolve().parent == APP:
            importlib.reload(_mod)
        else:
            del sys.modules[_name]

import common as C  # noqa: E402

st.set_page_config(page_title="Predictive Maintenance & OEE", layout="wide")

# Narrow screens (Chrome review 2026-11-20): a metric label wraps onto a second line instead of ending in "...", and on a
# phone the static tables use tighter text so their last column is not cut off. Style only: no label, number or table
# changes, so the verifiers' word-for-word checks are unaffected. st.html with only a <style> block takes no space.
st.html("""<style>
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
/* a row of tiles lines up at the bottom, so a tile whose label wraps keeps its value level with its neighbours' */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) { align-items: flex-end; }
@media (max-width: 640px) {
  [data-testid="stTable"] table { font-size: 0.78rem; }
  [data-testid="stTable"] th, [data-testid="stTable"] td { padding: 0.3rem 0.35rem !important; }
}
</style>""")

pages = [st.Page("views/home.py", title="Home", default=True),
         st.Page("views/how.py", title="How it works"),
         st.Page("views/fleet.py", title="Fleet health"),
         st.Page("views/engine.py", title="Engine drill-down"),
         st.Page("views/plan.py", title="Maintenance plan"),
         st.Page("views/oee.py", title="OEE policies (simulated)"),
         st.Page("views/cost.py", title="Cost what-if (illustrative)"),
         st.Page("views/monitoring.py", title="Monitoring"),
         st.Page("views/about.py", title="About and method")]
nav = st.navigation(pages)

with st.sidebar:
    st.divider()
    st.caption(f"Built by {C.AUTHOR} · [LinkedIn]({C.LINKEDIN}) · [GitHub]({C.REPO})")
    st.caption("Data: NASA C-MAPSS Turbofan Engine Degradation Simulation (NASA Prognostics Center of Excellence).")
    st.caption(f"Snapshot checksum `{C.manifest()['data_checksum'][:12]}` (full value on About and method).")

nav.run()
