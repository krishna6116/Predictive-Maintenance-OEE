# v1.1 manual checklist

The v1.1 scorecard (`dashboards/v11_scorecard.json`) counts a manual item only when its line is ticked (`- [x]`) and its
evidence is committed. Live checks run first on the preview app (branch `v1.1`), then on the real URL after the merge.

## Before the merge (temporary preview app `pm-oee-v11-preview`)

- [x] **Preview app created** from branch `v1.1`, main file `app/streamlit_app.py`, Python 3.14, secrets empty, public.
- [x] **Smoke check (V5.5) before each preview review**: `python -m dashboards.smoke_v11` (committed tree of the pushed branch) and `python -m dashboards.smoke_v11 --live` on the preview address; every page loads with no exception. If the live run shows an exception the committed tree does not, reboot the preview app (a running app can keep an old copy of `app/common.py` after an in-place update).
- [x] **Live checks on the preview** (`python -m dashboards.verify_v11 --live` on the preview address): V2.1, V4.1, V4.3; evidence in `dashboards/v11_evidence.json`, phone screenshots in `docs/figures/v11/`. Done 2026-11-15: all four pass (Home fold ends at 797 of 800 px; 375 px, no horizontal scroll, menu opens; titles in about 1 s; no page shows an exception).
- [ ] **Author review** of every page on a laptop and a phone (Home, cost what-if, engine drill-down, and the existing pages).

## After the merge

- [x] **Deployed checksum re-captured** (Phase 8 D2.4): v1.1 adds engine_history.csv, so the snapshot data checksum changes; `python scripts/capture_phase8_screenshots.py` re-reads it from the live About page, compares it with the manifest and replaces `docs/figures/phase8/deployed_about.png`.
- [x] **Smoke check (V5.5) on the real app**: `python -m dashboards.smoke_v11 --live https://predictive-maintenance-oee.streamlit.app`; every page loads with no exception.
- [x] **Live checks on the real app** (https://predictive-maintenance-oee.streamlit.app, same command); evidence updated.
- [x] **Release v1.1.0** tagged on the merge commit after CI is green on it; release body equals `docs/RELEASE_NOTES_v1.1.0.md`.
- [ ] **Preview app deleted** by the author.
