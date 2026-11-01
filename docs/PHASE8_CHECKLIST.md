# Phase 8 manual checklist

The scorecard (`dashboards/phase8_scorecard.json`) has three checks that code cannot do alone. Each passes only when
its evidence file is committed under `docs/figures/phase8/` AND its line below is ticked (`- [x]`).
`python -m dashboards.verify_phase8` reads this file and the evidence files.

- [ ] **D2.4** Deployed app's data checksum equals the repository's.
  Evidence: `deployed_about.png`, a screenshot of the deployed app's About page showing the full checksum, which must equal
  `data_checksum` in `dashboards/snapshots/manifest.json`.
- [ ] **D5.5** Phone width: each page at 375 px has no horizontal page scroll, readable charts, and appears in under 5 s.
  Evidence: `phone_fleet.png`, `phone_plan.png`, `phone_oee.png`, `phone_monitoring.png`, `phone_about.png`
  (browser developer tools, device toolbar, 375 px wide).
- [x] **D5.6** Power BI report follows the same rules: each page title is its question; Okabe-Ito colours, with meaning
  never carried by colour alone (data labels or a table next to the chart); axis titles with units; "Simulated" on the OEE
  page; FD004's C1 failure and the Phase 7 target miss visible on the monitoring page.
  Evidence: `powerbi_report.pdf` (File > Export > Export to PDF) and one PNG per page (`powerbi_<page>.png`).
