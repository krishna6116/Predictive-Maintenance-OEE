# v1.2 checklist

The v1.2 scorecard (`dashboards/v12_scorecard.json`) has one live check (W4.3). It runs on a local server of branch `v1.2`
before the merge and on the real URL after it; evidence goes in `dashboards/v12_evidence.json`.

## Before the merge

- [x] **Live checks on a local server**: `streamlit run app/streamlit_app.py`, then `python -m dashboards.verify_v12 --live http://localhost:8501`. Done 2026-11-15: pass (no exceptions on 9 pages; titles under 0.5 s; 375 px, no horizontal scroll on How and About).
- [ ] **Author review** of How it works and About on a laptop and a phone.

## After the merge

- [ ] **Smoke check on the real app**: `python -m dashboards.smoke_v11 --live https://predictive-maintenance-oee.streamlit.app`.
- [ ] **Live checks on the real app**: `python -m dashboards.verify_v12 --live https://predictive-maintenance-oee.streamlit.app`; evidence committed.
