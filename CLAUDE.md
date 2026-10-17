# Predictive Maintenance & OEE

Portfolio project: predict remaining useful life (RUL) of turbofan engines from NASA C-MAPSS sensor data, schedule maintenance under crew limits, and show the effect on OEE. Built phase by phase from the study guide (Claude Doc: https://claude.ai/code/artifact/7dd89f2e-799d-48a0-aa7e-f2e28c2fbd3d; PDF copy in `docs/`, git-ignored).

## Status (2026-10-17)

- Phase 0 (setup): done. Commit `b3c4bb2`. `.venv` on Python 3.14 (torch 2.14 CPU, prefect 3.8, xgboost 3.4, ortools 9.15). `pytest -q` passes.
- Data: C-MAPSS FD001-FD004 downloaded to `data/raw/` (git-ignored); FD001 verified (train 20,631 x 26, test 13,096 x 26, 100 engines each, no nulls).
- Waiting on the user:
  1. PostgreSQL 17 (installed, running): user creates database `pdm` and their own `.env` from `.env.example`. Never type or handle the password.
  2. GitHub: user creates an empty public repo `Predictive-Maintenance-OEE`, then connect `origin` and push.
- Phase 1 (EDA): done. `notebooks/01_eda.ipynb`, `docs/EDA_FINDINGS.md`, figures in `docs/figures/`. Keep 14 sensors: s2 s3 s4 s7 s8 s9 s11 s12 s13 s14 s15 s17 s20 s21.
- Next: Phase 2 (star schema, batch ingest, validation/quarantine, idempotent ETL with run IDs).

## Phases

0 setup · 1 EDA · 2 pipeline (PostgreSQL star schema) · 3 features (RUL cap 125, engine-level split, rolling + FFT) · 4 baselines (fleet mean, Ridge, XGBoost; RMSE + PHM08 score) · 5 PyTorch LSTM/1D-CNN + MC-dropout intervals · 6 alerts + OR-Tools CP-SAT schedule + simulated OEE, 3-policy comparison · 7 Prefect flow + drift/error monitoring · 8 Power BI + Streamlit (snapshot CSVs for the cloud demo) · 9 README, docs, resume bullets.

## Conventions

- Honest evaluation: every model is compared against baselines on the same test engines; publish the winner even if it is not the deep model. Label simulated parts (plant/OEE layer) as simulated.
- Split by engine, never by row; fit scalers on training engines only.
- Tests in `tests/` (pytest, `pythonpath = .`); CI in `.github/workflows/ci.yml`.
- Keep a dated entry in `docs/LEARNING_LOG.md` for each working session.
- The user is learning: explain the concept and the reason for each step, then do it.
- End every phase's notebook (and the phase summary to the user) with a "Why we did these tasks" table: task | why we did it | what we learned | what it changes next.
- Never commit personal files (resumes live in `Documents\Job-Applications`), `.env`, data files or model weights.
