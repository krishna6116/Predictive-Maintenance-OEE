"""About / method: where every number comes from, how it is checked, and the snapshot checksum."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

import common as C

C.header("about", "Every number in this app is read from frozen snapshot files and re-computed independently from the raw "
                  "results by a verifier that runs in CI.")

# v1.2 author section (dashboards/v12_spec.json author_section): first on the page; no email address or phone number
st.subheader("About the author")
A = C.AUTHOR_PROFILE
st.markdown(f"**{C.AUTHOR}**, {A['program']}. {A['background']}.\n\n"
            f"[LinkedIn]({C.LINKEDIN}) · [GitHub]({C.GITHUB}) · [Case study]({C.DOCS}/CASE_STUDY.md) · [Source code]({C.REPO})\n\n"
            f"{A['roles']}.")
st.caption(f"How it was built: {A['how_built']}")
st.divider()

man = C.manifest()
fleet = C.load("fleet.csv")
data = "".join(f"{n}:{f['sha256']}\n" for n, f in sorted(man["files"].items()))
checksum = hashlib.sha256(data.encode()).hexdigest()
if checksum != man["data_checksum"]:
    st.error("The snapshot files do not add up to the manifest's data checksum: the snapshot set is inconsistent.")
    st.stop()
st.metric("Snapshot data checksum", checksum[:16], help="sha256 over all snapshot files' sha256 values (manifest.json "
          "data_checksum); equal to the repository's means this app shows exactly the committed data.")
st.code(checksum, language=None, wrap_lines=True)  # wraps on a phone instead of scrolling inside the box (review 2026-11-16)

st.markdown(f"""
**What is real and what is simulated**
- **Real:** NASA C-MAPSS turbofan engines (FD001-FD004), the published {C.fact('published_model')} forecasts, their intervals,
  the alerts, the CP-SAT crew plans and the monitoring runs.
- **Simulated:** the plant behind the OEE page (production, downtime and OEE are not measured data). Crew capacity and the
  calendar are guide assumptions.

**How the numbers are checked**
- Model choices were made on validation engines only; test sets confirm. Comparisons are Holm-corrected.
- Criteria were written down before results were seen; misses (Phase 7 false alarms, Phase 7b FD004) are reported, not re-tuned.
- `python -m dashboards.verify_phase8` recomputes every number on these pages from the raw results files and checks the
  snapshots against their manifest and against PostgreSQL.

**Documents** (in `docs/` of the repository)
- [Model metrics, FD001]({C.DOCS}/MODEL_METRICS_PHASE5.md) and [FD003 confirmation]({C.DOCS}/MODEL_METRICS_PHASE5B.md)
- [Maintenance decisions and simulated OEE]({C.DOCS}/MAINTENANCE_OEE.md)
- [Monitoring, Phase 7]({C.DOCS}/MONITORING.md) and [Phase 7b]({C.DOCS}/MONITORING_7B.md)
- [Dashboards and the Phase 8 scorecard]({C.DOCS}/DASHBOARDS.md)
- Source: {C.REPO}
""")

st.subheader("Monte Carlo noise in the intervals")
diff = (fleet.rul_mc_mean - fleet.rolling_mc_mean).abs()
flips = int(((fleet.rul_low <= C.fact("alert_threshold")) != (fleet.rolling_rul_low <= C.fact("alert_threshold"))).sum())
# one tile per row with short labels: side by side they were cut off below about 1,100 px (review 2026-11-16)
st.metric("Largest change in MC mean (cycles)", C.num(diff.max()),
          help="Same model, same engine, same cycle: the published forecast and the Phase 6 rolling forecast each average "
               f"{C.fact('interval_passes')} dropout passes, with different random draws.")
st.metric("Engines whose flag differs", str(flips),
          help="Engines whose lower bound is on different sides of the alert threshold under the two draw sets.")
st.caption(f"Monte Carlo dropout is random: with {C.fact('interval_passes')} passes the interval itself moves a little between runs. Decisions here "
           "use the published draw set, the one the plan acted on.")

st.subheader("Definitions")
# a list, not a grid: long definitions wrap instead of being cut off (review 2026-11-15)
st.markdown("\n".join(f"- **{k.replace('_', ' ')}:** {v}" for k, v in C.fact("definitions").items()))

st.subheader("Snapshot files")
# a static table wraps, so 'n/a (JSON file)' is not cut off in a narrow Rows column (review 2026-11-15)
st.table(pd.DataFrame([{"File": n, "Rows": "n/a (JSON file)" if f["rows"] is None else f"{f['rows']:,}", "sha256": f["sha256"][:16]}
                       for n, f in sorted(man["files"].items())]).set_index("File"))
