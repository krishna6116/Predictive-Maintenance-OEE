# Predictive Maintenance & OEE

Predicts turbofan engine failures from sensor data, schedules maintenance under crew limits, and shows the effect on OEE.

[![CI](https://github.com/krishna6116/Predictive-Maintenance-OEE/actions/workflows/ci.yml/badge.svg)](https://github.com/krishna6116/Predictive-Maintenance-OEE/actions/workflows/ci.yml)

> **Work in progress.** Phases 0-7 complete, plus a confirmation on a second NASA dataset (5b); Phase 8 (dashboards) next.

## Progress

| Phase | Status | Read more |
| --- | --- | --- |
| 0. Setup: structure, tests, CI | Done | |
| 1. Exploratory data analysis (FD001) | Done | [EDA findings](docs/EDA_FINDINGS.md) · [notebook](notebooks/01_eda.ipynb) |
| 2. Data pipeline: PostgreSQL star schema, batch ingest, validation and quarantine, idempotent run-ID loads | Done | [Pipeline design](docs/PIPELINE.md) |
| 3. Feature engineering: RUL cap, engine-level split, 168 causal features, incremental builds | Done | [Features](docs/FEATURES.md) · [notebook](notebooks/03_features.ipynb) |
| 4. Baseline models: fleet average, Ridge, XGBoost; bootstrap intervals, paired comparisons, ablations | Done | [Model metrics](docs/MODEL_METRICS.md) · [notebook](notebooks/04_baselines.ipynb) |
| 5. Deep learning: LSTM and 1D-CNN on sensor windows, MC-dropout intervals calibrated on validation | Done | [Deep model metrics](docs/MODEL_METRICS_PHASE5.md) · [notebook](notebooks/05_deep.ipynb) |
| 5b. Confirmation on FD003 (fresh test set, second fault mode): frozen recipe, pre-registered tests, better interval rule | Done | [Confirmation metrics](docs/MODEL_METRICS_PHASE5B.md) · [notebook](notebooks/05b_fd003.ipynb) |
| 6. Maintenance alerts on the interval's lower bound, OR-Tools CP-SAT crew schedule, simulated plant and OEE, 3-policy comparison | Done | [Maintenance and OEE](docs/MAINTENANCE_OEE.md) · [notebook](notebooks/06_maintenance.ipynb) |
| 7. Orchestration and monitoring: Prefect shift flow with retries, model health once labels exist, KS drift, pre-registered retraining flag | Done | [Monitoring](docs/MONITORING.md) · [notebook](notebooks/07_monitoring.ipynb) |
| 7b. Monitoring re-tested: wear-adjusted drift rule pre-registered on validation, confirmed on unseen FD002/FD004, fault suite, closed loop | Done (FD004 failed one criterion) | [Phase 7b results](docs/MONITORING_7B.md) · [notebook](notebooks/07b_monitoring.ipynb) |
| 8-9. Dashboards, write-up | Next / planned | |

**Highlights so far**

- 14 of 21 sensors kept; 8 constant or two-valued inputs dropped, each with a stated reason.
- Sensors stay flat for most of an engine's life and drift in the last ~100-150 cycles, which is why remaining life is capped at 125 cycles.
- 20,631 training and 2,000 test readings loaded into PostgreSQL and verified against the NASA files cell by cell (0 mismatches). Injected faults are quarantined with a reason instead of being loaded.
- 168 features built only from past readings; smoothing lifts the median sensor's correlation with remaining life from 0.69 to 0.80. Features built per batch match a full rebuild exactly, and an independent recomputation of all 3.9 million stored feature values matches to within 2e-13 (`python -m features.verify`, 25/25 checks).
- Baseline on NASA's 100 test engines: XGBoost reaches **15.08 RMSE** (95% interval 12.2-17.9). It beats a fleet average (41.9) and an age-only rule (31.2). Near failure (50 or fewer cycles left) its mean error is **4.4 cycles**, against 13.3 for a linear model on the same features; it was chosen on validation data, and the test set confirms it. Feature engineering is the largest real gain; FFT features show no measurable effect, and that is reported. Results reproduce on any machine: XGBoost runs on one fixed thread, and CI re-trains it on Linux and checks it against the published numbers.
- Deep learning (PyTorch) on the same 100 test engines: an LSTM on 40-cycle sensor windows reaches **13.30 RMSE** (95% interval 10.5-16.1) and 3.1 cycles mean error near failure. That is better than XGBoost on paper, but the paired, Holm-corrected comparison does not exclude zero, so on FD001 alone the rule set before testing kept XGBoost. MC-dropout intervals covered only 76% of validation truths at a nominal 90%; one scale factor learned on validation brought test coverage to 82% (88% near failure).
- Confirmation on NASA FD003, a fresh dataset with a second fault mode, with every setting frozen from FD001 and the tests committed before the run: the LSTM reaches **12.66 RMSE** against XGBoost's 15.50. Pooled over 200 test engines its advantage is **2.31 cycles RMSE (95% interval 0.76-4.05, Holm p 0.007)** and 2.76 vs 4.60 cycles near failure, so **the LSTM is now the published model**. A two-factor interval rule, chosen on FD001 validation before FD003 was touched, covers 86% of FD003 test engines (100% of the 29 near failure) at a nominal 90%.
- Maintenance decisions (simulated plant, not measured data): engines are flagged when the lower edge of their 90% interval reaches 20 cycles and scheduled with OR-Tools CP-SAT under 3 crews a day. In a day-by-day replay over the 200 test engines, the predictive policy reaches **98.2% simulated OEE** with 8 unplanned failures, against 95.2% for run to failure (200 failures) and 97.8% for a fixed service interval (21 failures, and 2.7 times as much useful life thrown away). All 8 remaining failures happen after an engine's NASA data ends, where the planner cannot get new readings. Every schedule is re-checked independently, and each snapshot plan is proved optimal with a second algorithm.
- Orchestration and monitoring: a Prefect flow runs each 8-hour shift (ingest, validate, load, predict with the published LSTM, alerts, CP-SAT schedule, model health, drift, retraining flag), with retries, fixed run IDs and resume after failure; 968 runs replay both test fleets. Drift and retraining rules were fixed on validation engines and committed before the replay (KS on per-engine means, Holm over 14 sensors, alpha 0.01; RMSE limit 1.45 x validation, because 1.3 x raised too many false alarms). The guide's +5% sensor fault is caught every time. On clean test data, though, the drift test fired in 11.6% of clean test runs against a 5% target: traced to engines that are healthier than the age-matched training reference, not a sensor fault. It is reported, not re-tuned.
- Monitoring re-tested (Phase 7b): a new drift rule compares each sensor with what it should read for an engine of that age and predicted wear, with persistence and a 'sensor fault suspected' path; criteria and rule were committed before any test replay and judged on NASA FD002/FD004, which nothing had seen. On FD002 clean runs raised flags in 0.0% / 0.9% of runs (synchronized / staggered fleet); on FD004 in 19.5% / 13.9%, against a 5% target, so it **failed** there. Every injected step and ramp fault was caught (median 2 and 8 shifts). A spread-model defect was found and fixed (post hoc) but did not explain FD004, which is most likely a two-fault-mode effect. With services actually carried out, labels-only monitoring read a 24% miss rate where the truth was 3.9%: the model's own decisions hide the labels it needs.

![Sensors drift as engines approach failure](docs/figures/fd001_sensor_trends.png)

## Data

NASA C-MAPSS Turbofan Engine Degradation Simulation (A. Saxena and K. Goebel, NASA Prognostics Center of Excellence, 2008). Download it with one command; every file is checked against the SHA-256 checksums in `data/checksums.sha256`, so results are always reproduced on byte-identical data:

```powershell
python -m pipeline.download_data
```

FD001: 100 training engines (20,631 rows), 100 test engines (13,096 rows), 26 columns. Data files are not committed.

**Continuous verification:** on every push, GitHub Actions downloads the pinned data, runs the pipeline into a fresh PostgreSQL database, builds the features, and runs the test suite plus the Phase 3, 4, 5, 5b, 6 and 7 verifiers (Phase 7 also replays 150 shifts of the flow in a fresh database), which recompute the published numbers with independent code.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt   # exact versions; requirements.txt lists the top-level packages
pytest -q
```

## Shift pipeline and monitoring (Phase 7)

`pipeline/flow.py` is a Prefect flow. Each run is one 8-hour shift: for each test fleet it ingests the readings that arrived, validates them (bad rows are quarantined with a reason), loads them into PostgreSQL, predicts remaining life with the published LSTM and its 90% interval, flags engines whose lower bound reaches 20 cycles, schedules them with OR-Tools under 3 crews a day, scores the model on engines whose failure is already known, tests every sensor for drift, and sets a retraining flag when the pre-registered rule fires. Database and file steps retry automatically; each shift has a fixed run ID, so re-running it replaces its rows and a failed shift is resumed by the next run.

```powershell
prefect server start                                           # terminal 1; dashboard at http://127.0.0.1:4200
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api   # terminal 2, once
python -m pipeline.flow --once          # run the next shift
python -m pipeline.flow --serve         # demo: a new shift every 10 minutes (--interval to change)
python -m pipeline.flow --replay-all    # every shift until all failures are known, then export the results
python -m pipeline.verify_phase7        # independent check of the replay
```

Monitoring, in short: a prediction is scored only after the engine has actually failed (no future labels); drift is a KS test per sensor on per-engine means against training engines of the same age, corrected for 14 tests (Holm); the drift and retraining limits were chosen on validation engines and committed before the replay. Results, including where the rule did not meet its targets on clean test data and why, are in [docs/MONITORING.md](docs/MONITORING.md).
