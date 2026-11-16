# v1.2 checklist

The v1.2 scorecard (`dashboards/v12_scorecard.json`) has one live check (W4.3). It runs on a local server of branch `v1.2`
before the merge and on the real URL after it; evidence goes in `dashboards/v12_evidence.json`.

## Before the merge

- [x] **Live checks on a local server**: `streamlit run app/streamlit_app.py`, then `python -m dashboards.verify_v12 --live http://localhost:8501`. Done 2026-11-15: pass (no exceptions on 9 pages; titles under 0.5 s; 375 px, no horizontal scroll on How and About).
- [ ] **Author review** of How it works and About on a laptop and a phone.

## After the merge

- [x] **Smoke check on the real app**: `python -m dashboards.smoke_v11 --live https://predictive-maintenance-oee.streamlit.app`. Done 2026-11-16: all 9 pages load with no exception.
- [x] **Live checks on the real app**: `python -m dashboards.verify_v12 --live https://predictive-maintenance-oee.streamlit.app`; evidence committed. Done 2026-11-16: pass (titles 1.6 s How, 1.2 s About; 375 px, no horizontal scroll). CI success on merge commit 6306713 recorded (W4.4).
