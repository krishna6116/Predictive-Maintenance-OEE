# Predictive Maintenance & OEE

Predicts turbofan engine failures from sensor data, schedules maintenance under crew limits, and shows the effect on OEE.

> Work in progress. Phase 0 (setup) complete.

## Data

NASA C-MAPSS Turbofan Engine Degradation Simulation (NASA Prognostics Center of Excellence). Download `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` into `data/raw/` (not committed).

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```
