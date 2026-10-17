# Predictive Maintenance & OEE

Predicts turbofan engine failures from sensor data, schedules maintenance under crew limits, and shows the effect on OEE.

[![CI](https://github.com/krishna6116/Predictive-Maintenance-OEE/actions/workflows/ci.yml/badge.svg)](https://github.com/krishna6116/Predictive-Maintenance-OEE/actions/workflows/ci.yml)

> **Work in progress.** Phases 0-2 complete; Phase 3 (features) next.

## Progress

| Phase | Status | Read more |
| --- | --- | --- |
| 0. Setup: structure, tests, CI | Done | |
| 1. Exploratory data analysis (FD001) | Done | [EDA findings](docs/EDA_FINDINGS.md) · [notebook](notebooks/01_eda.ipynb) |
| 2. Data pipeline: PostgreSQL star schema, batch ingest, validation and quarantine, idempotent run-ID loads | Done | [Pipeline design](docs/PIPELINE.md) |
| 3. Feature engineering | Next | |
| 4-9. Baselines, deep learning, maintenance scheduling and OEE, orchestration, dashboards | Planned | |

**Highlights so far**

- 14 of 21 sensors kept; 8 constant or two-valued inputs dropped, each with a stated reason.
- Sensors stay flat for most of an engine's life and drift in the last ~100-150 cycles, which is why remaining life is capped at 125 cycles.
- 20,631 training and 2,000 test readings loaded into PostgreSQL and verified against the NASA files cell by cell (0 mismatches). Injected faults are quarantined with a reason instead of being loaded.

![Sensors drift as engines approach failure](docs/figures/fd001_sensor_trends.png)

## Data

NASA C-MAPSS Turbofan Engine Degradation Simulation (A. Saxena and K. Goebel, NASA Prognostics Center of Excellence, 2008). Download the zip from NASA's PCoE data repository mirror:

```powershell
curl.exe -L -o cmapss.zip "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
```

It contains `CMAPSSData.zip`; extract the `train_FD00x.txt`, `test_FD00x.txt` and `RUL_FD00x.txt` files into `data/raw/` (not committed). FD001: 100 training engines (20,631 rows), 100 test engines (13,096 rows), 26 columns.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```
