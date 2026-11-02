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

man = C.manifest()
fleet = C.load("fleet.csv")
data = "".join(f"{n}:{f['sha256']}\n" for n, f in sorted(man["files"].items()))
checksum = hashlib.sha256(data.encode()).hexdigest()
if checksum != man["data_checksum"]:
    st.error("The snapshot files do not add up to the manifest's data checksum: the snapshot set is inconsistent.")
    st.stop()
st.metric("Snapshot data checksum", checksum[:16], help="sha256 over all snapshot files' sha256 values (manifest.json "
          "data_checksum); equal to the repository's means this app shows exactly the committed data.")
st.code(checksum, language=None)

st.markdown(f"""
**What is real and what is simulated**
- Real: NASA C-MAPSS turbofan engines (FD001-FD004), the published {C.fact('published_model')} forecasts, their intervals,
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
c1, c2 = st.columns(2)
c1.metric("Largest change in MC mean between two draw sets (cycles)", C.num(diff.max()),
          help="Same model, same engine, same cycle: the published forecast and the Phase 6 rolling forecast each average "
               f"{C.fact('interval_passes')} dropout passes, with different random draws.")
c2.metric("Engines whose flag differs between the two draw sets", str(flips),
          help="Engines whose lower bound is on different sides of the alert threshold under the two draw sets.")
st.caption(f"Monte Carlo dropout is random: with {C.fact('interval_passes')} passes the interval itself moves a little between runs. Decisions here "
           "use the published draw set, the one the plan acted on.")

st.subheader("Definitions")
st.dataframe(pd.DataFrame([{"Term": k.replace("_", " "), "Definition": v} for k, v in C.fact("definitions").items()]),
             hide_index=True, width="stretch")

st.subheader("Snapshot files")
st.dataframe(pd.DataFrame([{"File": n, "Rows": f["rows"], "sha256": f["sha256"][:16]} for n, f in sorted(man["files"].items())]),
             hide_index=True, width="stretch")
